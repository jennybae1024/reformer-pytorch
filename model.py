import math, copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Function
from utils.utils import look_back, reverse_sort, expand, expand_gather
from embeddings import Embeddings, PositionEmbeddigs
from lshattention import MultiHeadLSHAttention

class Reversible(Function):

    @staticmethod
    def forward(ctx, layer, x1, x2):
        # layer <- Reformer Block (lsh attention -> ffn) 에 해당
        ctx.layer = layer
        with torch.no_grad():
            y1, y2 = layer(x1, x2)
        Reversible.outputs = (y1.detach(), y2.detach())
        return y1, y2
    
    @staticmethod
    def backward(ctx, y1_grad, y2_grad):
        y1, y2 = Reversible.outputs
        y1.requires_grad = True
        y2.requires_grad = True

        with torch.enable_grad():
            g_y1 = ctx.layer.g_block(y1)
            g_y1.backward(y2_grad)

        with torch.no_grad():
            x2 = y2 - g_y1
            del y2, g_y1
            x1_grad = y1_grad + y1.grad
            del y1_grad
            y1.grad = None
        
        with torch.enable_grad():
            x2.requires_grad = True
            f_x2 = ctx.layer.f_block(x2)
            f_x2.backward(x1_grad)

        with torch.no_grad():
            x1 = y1 - f_x2
            x2_grad = y2_grad + x2.grad
            x2.grad = None

            Reversible.outputs = (x1.detach(), x2.detach())

        return (None, x1_grad, x2_grad)


class ChunkFeedForward(nn.Module):
    def __init__(self, chunks, d_model, d_ff, dropout=0.1):
        super(ChunkFeedForward, self).__init__()
        self.chunks = chunks
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [batch, len, d_,model]
        chunks = torch.chunk(x, chunks=self.chunks, dim=1)
        chunck_outputs = [self.w_2(self.dropout(F.relu(self.w_1(chunk)))) for chunk in chunks]
        output = torch.cat(chunck_outputs, dim=1)

        return output


# layer 을 구성하는 두가지 (attn, ffn) sublayer 처리
class SublayerConnection(nn.Module):
    def __init__(self, func, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(self.func(x)))


class ReformerLayer(nn.Module):
    def __init__(self, args):
        super(ReformerLayer, self).__init__()
        num_heads = args.num_heads
        d_model = args.d_model
        d_ff = args.d_ff
        rounds = args.rounds
        bucket_length = args.bucket_length
        dropout = args.dropout
        causal = args.causal
        chunks = args.chunks
        self.attn = MultiHeadLSHAttention(d_model, num_heads, rounds, bucket_length, dropout, causal)
        self.feedforward = ChunkFeedForward(chunks, d_model, d_ff)

        self.f_block = SublayerConnection(self.attn, d_model, dropout)
        self.g_block = SublayerConnection(self.feedforward, d_model, dropout)

    def forward(self, x1, x2):
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(x1)

        return y1, y2


class Reformer(nn.Module):
    def __init__(self, args):
        super(Reformer, self).__init__()
        N = args.num_layers
        self.layers =  nn.ModuleList([copy.deepcopy(ReformerLayer(args)) for _ in range(N)])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = Reversible.apply(layer, x1, x2)
        return x2


class ReformerLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeddings = Embeddings(args.d_model, args.vocab)
        self.pos_embs = PositionEmbeddigs(args.d_model, args.dropout, args.max_len)
        self.decoder = Reformer(args)
        self.proj = nn.Linear(args.d_model, args.vocab)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        x = self.layers(x, **kwargs)

    def forward(self, input, labels=None):
        x = self.pos_embs(self.embeddings(input))
        dec_output = self.decoder(x, x)
        lm_logits = self.proj(dec_output)
        
        loss = 0.0
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return lm_logits, loss
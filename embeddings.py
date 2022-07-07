import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embs = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embs(x) * math.sqrt(self.d_model)


class PositionEmbeddigs(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEmbeddigs, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)* (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        self.positional_encoding = pe.unsqueeze(0)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_len = x.size()[1]
        x = x + Variable(self.positional_encoding[:, :x_len], requires_grad=False).to(x.device)
        return self.dropout(x)

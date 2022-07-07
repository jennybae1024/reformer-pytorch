import math, copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Function
from utils.utils import look_back, reverse_sort, expand, expand_gather
from embeddings import Embeddings, PositionEmbeddigs


class LSH(nn.Module):
    def __init__(self, d_k, rounds):
        super(LSH, self).__init__()
        self.d_k = d_k
        self.rounds = rounds
        self.rand_matrix = None

    def forward(self, inputs, n_buckets):
        batch, seq_len, _ = inputs.size()
        # 사이즈 1 인 벡터로 정규화 -> angular lsh 
        inputs = F.normalize(inputs, p=2, dim=-1)

        # multi-round hash 만큼 self.rounds
        self.rand_matrix = torch.randn([batch, self.d_k, self.rounds, n_buckets//2],
                                        device=inputs.get_device())
        self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)

        xR = torch.einsum('bld, bdrn->blrn', inputs, self.rand_matrix)

        hashes = torch.argmax(torch.concat([xR, -xR], dim=-1), dim=-1).int() #[batch, seq_len, rounds]

        # hashes 와 같은 device, dtype 을 가진 임의의 tensor 생성
        arange = hashes.new_empty((1, seq_len, 1))
        
        # hashes%seq_len -> pos 정보, hashes//n_buckets->bucekt_id 정보
        hashes = hashes * seq_len + torch.arange(seq_len, out=arange).expand_as(hashes)

        return hashes

class LSHAttention(nn.Module):
    def __init__(self, d_k=64, rounds=4, bucket_length=64, dropout=0.1, causal=False):
        super(LSHAttention, self).__init__()

        self.d_k = d_k
        self.rounds = rounds
        self.lsh = LSH(d_k, rounds)
        self.bucket_length = bucket_length
        self.dropout = nn.Dropout(p=dropout)
        self.causal = causal

    def forward(self, query, value):
        # query: [batch*num_head, length, d_k]
        length = query.size(1)
        n_buckets = length // self.bucket_length

        # sorted_hashes: bucket_id & pos_id 에 따라 정렬된 값, hash_indices: sorting 결과값의 원래 포지션
        sorted_hashes, hash_indice = torch.sort(self.lsh(query, n_buckets), dim=1) # [batch*num_heads, length, rounds]
        # sort 전 상태로 돌아가기 위한 index 값 -> 마지막에 다시 pos_id 순서대로 바꿔야하므로 
        original_indice = reverse_sort(hash_indice, dim=1)
        

        # 원래 Q -> bucket_id & pos_id 순서대로 정렬
        expanded_query = query.unsqueeze(3).expand(-1, -1, -1, self.rounds) # [batch*num_heads, length, d_k, rounds]
        expanded_hash_indice = hash_indice.unsqueeze(2).expand(-1, -1, self.d_k, -1)
        reoredered_query = expanded_query.gather(dim=1, index=expanded_hash_indice) 
        # chunk 로 나누는 작업
        reoredered_query = reoredered_query.reshape(-1, n_buckets, self.bucket_length, self.d_k, self.rounds) # [batch*num_heads, n_buckets, bucket_length, d_k, rounds]
        
        # 이전 chunk 의 같은 bucket_id 를 가진 key attend
        lookback_key = F.normalize(look_back(reoredered_query), p=2, dim=-2) # [batch*num_heads, n_buckets, bucket_length*2, d_k, rounds]
        qk_matmul = torch.einsum('...ijk,...ljk->...ilk', reoredered_query, lookback_key)/math.sqrt(self.d_k)
        # [batch*num_heads, n_buckets, bucket_length, bucket_length*2, rounds]
        

        # MASK: 1) 같은 bucket_id 가 아닌 경우, 2) causal, 3) Idneity mask
        sorted_hashes = sorted_hashes.reshape(-1, n_buckets, self.bucket_length, self.rounds)//length
        # [batch*num_heads, n_buckets, bucket_length, rounds]

        # 1) bucket idx 가 동일한 경우에만 attend
        qk_matmul.masked_fill_(mask=(sorted_hashes[...,None,:] != look_back(sorted_hashes)[..., None, :, :]), value=-1e9)
        
        query_indice = hash_indice.reshape(-1, n_buckets, self.bucket_length, self.rounds).int() # [batch*num_heads, n_buckets, bucket_length, rounds]
        key_indice = look_back(query_indice) # [batch*num_heads, n_buckets, bucket_length*2, rounds]

        # 2) CAUSAL MASK
        if self.causal:
            qk_matmul.masked_fill_(mask=(query_indice[...,None,:] < key_indice[...,None,:,:]), value=-1e10)

        # 3) IDENTITY MASK
        qk_matmul.masked_fill_(mask=(query_indice[...,None,:] == key_indice[...,None,:,:]), value=-1e5)


        # SOFTMAX
        qk_matmul = qk_matmul.flatten(1, 2) # [batch * head, length, bucket_length * 2, rounds]
        logsumexp_qk = torch.logsumexp(qk_matmul, dim=2) # [batch * head, length, rounds]
        softmax_qk = torch.exp(qk_matmul - logsumexp_qk[..., None, :])

        if self.training:
            softmax_qk = self.dropout(softmax_qk)
            # [batch * head, length, bucket_length * 2, rounds]

        # Value 곱하기
        # rounds 만큼 확장 -> hash_indice 에 맞게 재정렬 -> shape 변경
        expanded_val = expand(value, dim=3, num=self.rounds)
        reordered_value = expand_gather(expanded_val, dim=1, index=hash_indice, expand_dim=2, num=self.d_k)
        reordered_value = reordered_value.reshape(-1, n_buckets, self.bucket_length, self.d_k, self.rounds)
        # [batch * head, n_buckets, bucket_length, d_k, rounds]


        softmax_qk = softmax_qk.reshape(-1, n_buckets, self.bucket_length, self.bucket_length * 2, self.rounds)
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', softmax_qk, look_back(reordered_value))
        attention = attention.flatten(1, 2) # [batch * head, length, d_k, rounds]

        # 원래 순서대로 돌리기
        attention = expand_gather(attention, dim=1, index=original_indice, expand_dim=2, num=self.d_k)
        # [batch * head, length, d_k, rounds]
        
        # hash-round 별 Weight 부여
        logsumexp_qk = torch.gather(logsumexp_qk, dim=1, index=original_indice)
        logsumexp_qk = F.softmax(logsumexp_qk, dim=1) # [batch * head, length, rounds]
        attention = torch.einsum('...ij,...j->...i', attention, logsumexp_qk)

        return attention

        
        
class MultiHeadLSHAttention(nn.Module):
    def __init__(self, d_model, num_heads, rounds, bucket_length, dropout, causal):
        super(MultiHeadLSHAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.head = num_heads
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.lshattention = LSHAttention(self.d_k, rounds, bucket_length, dropout, causal)

    def forward(self, input_tensor):
        length = input_tensor.size(1)

        query = self.linear_query(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2) 
        value = self.linear_value(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]
        
        query = query.flatten(0, 1)
        value = value.flatten(0, 1)
        attention = self.lshattention(query, value).reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        attention = attention.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(attention)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy


def look_back(input_tensor: torch.Tensor) -> torch.Tensor:
    shift = torch.cat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets, bucket_length, d_k, rounds]
    concat = torch.cat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
    return concat

def reverse_sort(indice: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    sorting 전 상태로 되돌리기 위한 indice
    '''
    new_size = [1] * indice.dim()
    new_size[dim] = indice.size(dim)
    arange = indice.new_empty(size=new_size)
    torch.arange(new_size[dim], out=arange)
    arange = arange.expand_as(indice)
    new_indice = torch.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice

def expand(input_tensor: torch.Tensor, dim=0, num=1) -> torch.Tensor:
    '''
    dim 에 num 만큼 확장
    '''
    new_size = [-1] * (input_tensor.dim() + 1)
    new_size[dim] = num
    return input_tensor.unsqueeze(dim=dim).expand(new_size)

def expand_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor, expand_dim=0, num=1) -> torch.Tensor:
    '''
    index 를 input_tensor 만큼 expand (expand_dim, num)
    이후 index 순서로 gather 하고 출력
    '''
    expanded_index = expand(index, dim=expand_dim, num=num)
    return input_tensor.gather(dim=dim, index=expanded_index)


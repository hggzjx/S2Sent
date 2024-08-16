import math
import torch
import torch.nn as nn
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr

class NestedSelectiveAttention(nn.Module):
    def __init__(self, num, dim1, dim2, dct_h=6, dct_w=6, freq_sel_method='low4'):
        super(NestedSelectiveAttention, self).__init__()

        self.num = num
        self.dim1 = dim1
        self.dim2 = dim2
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 6) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 6) for temp_y in mapper_y]

        self.squeeze = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, dim1)
        self.dense_z = nn.Sequential(nn.Linear(in_features=dim1, out_features=dim2)
                                ,nn.Tanh())
        self.dense_select_weights = nn.ModuleList([nn.Sequential(nn.Linear(in_features=dim2, out_features=dim1)
                                          , nn.BatchNorm1d(num_features=dim1, eps=1e-6, momentum=0.1, affine=True)
                                          , nn.Sigmoid()) for _ in range(num)])

    def forward(self,hidden_states):
        ##### begin spatial selection #####
        hidden_states = hidden_states[-(self.num+1):-1]
        stacked_out = torch.stack(hidden_states, dim=1).permute(0, 3, 1, 2)
        assert self.dim1 == stacked_out.shape[1]
        n, c, h, w = stacked_out.shape
        stacked_out_pooled = stacked_out
        if h != self.dct_h or w != self.dct_w:
            stacked_out_pooled = torch.nn.functional.adaptive_avg_pool2d(stacked_out, (self.dct_h, self.dct_w))
        # print('stacked_out_pooled',stacked_out.shape)
        #####begin frequency selection #####

        squeezed = self.squeeze(stacked_out_pooled)
        #####end frequency selection #####
        z = self.dense_z(squeezed)
        select_weights = [self.dense_select_weights[i](z) for i in range(self.num)]
        select_weights_norm = [torch.exp(weight) / torch.sum(torch.exp(torch.stack(select_weights))) for weight in
                               select_weights]
        weighted_added_branches = torch.sum(torch.stack([branch * weight.unsqueeze(1) for branch, weight in zip(hidden_states, select_weights_norm)]),dim=0)
        ##### end spatial selection #####
        return weighted_added_branches


def get_freq_indices(method):
    assert method in ['low1', 'low2', 'low4', 'low8', 'low16']
    num_freq = int(method[3:])
    all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
    mapper_x = all_low_indices_x[:num_freq]
    mapper_y = all_low_indices_y[:num_freq]

    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter

if __name__ == "__main__":
    num_tensors = 13
    tensor_size = [3, 64, 768]
    tensors = []
    for _ in range(num_tensors):
        tensor = torch.randn(*tensor_size)
        tensors.append(tensor)
    u = NestedSelectiveAttention(num=6,dim1=768,dim2=64)(tensors)
    print(u.shape)

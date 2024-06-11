import torch
import torch.nn as nn
from src.model.contrast import Contrast
import numpy as np


class MI_NCE(nn.Module):
    def __init__(self, num_feature, mi_hid, big=False, batch=None):
        super(MI_NCE, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_feature, mi_hid),
            nn.ELU(),
            nn.Linear(mi_hid, mi_hid)
        )
        self.big = big
        self.batch = batch

    def forward(self, x):
        x = self.proj(x)
        # if dataset is so big, we will randomly sample part of nodes to perform MI estimation
        if self.big == True:
            idx = np.random.choice(x.shape[0], self.batch, replace=False)
            idx.sort()
            x = x[idx]
        return x


class LayerMI(nn.Module):
    def __init__(self, args):
        super(LayerMI, self).__init__()
        self.args = args
        self.h_dim = args.hidden_size
        self.n_layer = args.n_layer

        self.mi_nce = nn.ModuleList()
        for _ in range(self.args.n_layer):
            layer = nn.ModuleList()
            for __ in range(2):
                layer.append(MI_NCE(self.h_dim, 64))
            self.mi_nce.append(layer)
        if self.args.layer_mi_last:
            self.last_mi = MI_NCE(self.h_dim, 64)

    def forward(self, x):
        outputs = dict()
        for i in range(self.n_layer):
            x_i = x[i]
            z1 = self.mi_nce[i][0](x_i[0])
            z2 = self.mi_nce[i][1](x_i[1])
            outputs[i] = [z1, z2]
        if self.args.layer_mi_last:
            z = self.last_mi(x['h'])
            outputs['z'] = z

        return outputs

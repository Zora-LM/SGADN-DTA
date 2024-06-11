import torch
from torch import nn
import torch.nn.functional as F


class SimpleCrossAttention(nn.Module):
    def __init__(self, args):
        super(SimpleCrossAttention, self).__init__()
        self.d_feat = args.hidden_size
        self.norm = nn.LayerNorm(self.d_feat)
        self.W1 = nn.Linear(self.d_feat, self.d_feat//4)
        self.W2 = nn.Linear(self.d_feat, self.d_feat//4)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        x1_norm, x2_norm = self.norm(x1), self.norm(x2)
        x1_h = self.W1(x1_norm)
        x2_h = self.W2(x2_norm)
        sim = self.softmax(self.cos_sim(x1_h, x2_h))
        x1_new = torch.matmul(sim, x1) + x1
        x2_new = torch.matmul(sim.t(), x2) + x2
        return x1_new, x2_new



class CrossAttention(nn.Module):
    def __init__(self, args):
        super(CrossAttention, self).__init__()
        self.d_feat = args.hidden_size
        self.n_head = args.n_head
        self.scale = self.d_feat ** (-0.5)
        self.norm = nn.LayerNorm(self.d_feat)
        self.Q = nn.Linear(self.d_feat, self.d_feat)
        self.K = nn.Linear(self.d_feat, self.d_feat)
        self.V = nn.Linear(self.d_feat, self.d_feat)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x1, x2):
        q, k, v = self.norm(x1), self.norm(x2), self.norm(x1)
        q_ = self.Q(q)  # .view(-1, self.n_head, self.d_feat // self.n_head)
        k_ = self.K(k)  # .view(-1, self.n_head, self.d_feat // self.n_head)
        v_ = self.V(v)  # .view(-1, self.n_head, self.d_feat // self.n_head)
        attn = self.scale * torch.matmul(q_, k_.permute(1, 0))  # k_.permute(0, 2, 1))
        attn_softmax = self.softmax(attn)
        x = torch.matmul(attn_softmax, v_)  # .view(-1, self.d_feat)
        x_out = v + x
        return x_out

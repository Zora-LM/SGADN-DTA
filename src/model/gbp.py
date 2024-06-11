import torch
from torch import nn
import torch.nn.functional as F



def l2_norm(x, dim=-1, eps=1e-8, keepdim=True):
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    norm = torch.sqrt(norm) + eps
    x_norm = x / norm
    return x_norm

class GBP(nn.Module):
    def __init__(self, d_input, bottleneck=1):
        super(GBP, self).__init__()

        self.d_input = d_input
        self.d_output = d_input
        self.act = nn.LeakyReLU(0.1)
        self.hidden_dim = self.d_input // bottleneck if bottleneck > 1 else self.d_output

        self.norm = nn.LayerNorm(self.d_output)
        # self.x1_down = nn.Linear(d_input, self.hidden_dim, bias=False)
        # self.x2_down = nn.Linear(d_input, self.hidden_dim, bias=False)
        self.x1_out = nn.Linear(self.d_input * 2, self.d_output)
        self.x2_up = nn.Linear(self.d_input, self.d_output, bias=False)

    def forward(self, x1, x2):
        # x1_hidden_rep = self.x1_down(x1)
        # x2_hidden_rep = self.x2_down(x2)
        # x1_norm = l2_norm(x1_hidden_rep, dim=-1)
        # x2_norm = l2_norm(x2_hidden_rep, dim=-1)
        merged = torch.cat([x1, x2], -1)
        x1_h = self.x1_out(merged)
        x1_rep = self.act(x1_h)

        x2_h = self.x2_up(x2)
        x2_rep = x2_h * torch.tanh(l2_norm(x2_h, dim=-1, keepdim=True))

        return x1_rep, x2_rep


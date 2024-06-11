import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.linalg import block_diag
import dgl
from dgl.nn import AvgPooling, GraphConv, MaxPooling
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from src.model.graph_utils import get_batch_id, topk
from src.model.aggregator import MaxPoolAggregator, MeanAggregator, LSTMAggregator
from src.model.bundler import Bundler

class LayerNorm(nn.Module):
    """Construct a layernorm module"""

    def __init__(self, num_features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """

    def __init__(
        self,
        in_dim: int,
        ratio=0.5,
        conv_op=GraphConv,
        non_linearity=torch.tanh,
    ):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = topk(
            score,
            self.ratio,
            get_batch_id(graph.batch_num_nodes()),
            graph.batch_num_nodes(),
        )
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)

        return graph, feature, perm


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """

    def __init__(self, in_dim: int, out_dim: int, pool_ratio=0.8):
        super(ConvPoolBlock, self).__init__()
        self.conv = GraphConv(in_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature):
        out = F.relu(self.conv(graph, feature))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, g_out


class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """

    def __init__(self, in_feats, out_feats, activation, dropout, aggregator_type, bn=False, bias=True):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(in_feats, out_feats, activation, dropout, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats, activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        if self.use_bn and not hasattr(self, 'bn'):
            device = h.device
            self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator, self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop('h')
        return h

class GATLayer(nn.Module):
    def __init__(self, args, in_feats, out_feats, activation, layer_norm=False):
        super(GATLayer, self).__init__()
        self.args = args
        self.layer_norm = layer_norm
        if layer_norm:
            self.head_norm = LayerNorm(in_feats)
            self.tail_norm = LayerNorm(in_feats)
            if args.pool_rel:
                self.rel_norm = LayerNorm(in_feats)
        self.head_W = nn.Linear(in_feats, 1, bias=False)
        self.tail_W = nn.Linear(in_feats, 1, bias=False)
        if args.pool_rel:
            self.rel_W = nn.Linear(in_feats, 1, bias=False)
        self.fc = nn.Linear(in_feats, out_feats)
        self.act = activation

    def forward(self, g, h, r):
        def edge_attention(edges):
            return {'e': self.act(edges.src['eh'] + edges.dst['et'] + edges.data['er'])}
        def edge_attention_no_rel(edges):
            return {'e': self.act(edges.src['eh'] + edges.dst['et'])}

        g = g.local_var()
        eh = et = h
        er = r
        if self.layer_norm:
            eh = self.head_norm(eh)
            et = self.tail_norm(et)
            if self.args.pool_rel:
                er = self.rel_norm(er)
        eh = torch.tanh(self.head_W(eh))
        et = torch.tanh(self.tail_W(et))
        g.ndata.update({'ft': h, 'eh': eh, 'et': et})
        if self.args.pool_rel:
            er = torch.tanh(self.rel_W(er))
            g.edata.update({'er': er})
            g.apply_edges(edge_attention)
        else:
            g.apply_edges(edge_attention_no_rel)
        attn = g.edata.pop('e')
        g.edata['a'] = edge_softmax(g, attn)
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        feat = g.ndata.pop('ft')
        h = self.fc(feat)
        h = F.normalize(h, p=2, dim=1)

        return h

class GCNLayer(nn.Module):
    def __init__(self, args, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.args = args
        self.fc = nn.Linear(in_feats, out_feats)

    def forward(self, g, h, r):
        g.ndata.update({'h': h})
        if self.args.pool_rel:
            g.edata.update({'r': r})
            g.update_all(fn.u_add_e('h', 'r', 'm'), fn.mean('m', 'ft'))
        else:
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'ft'))
        feat = g.ndata.pop('ft')
        h = self.fc(feat)
        # h = F.normalize(h, p=2, dim=1)
        return h


def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
    adj = torch.cat(adj_list, dim=0)
    feat = torch.cat(feat_list, dim=0)

    return feat, adj

class DiffPoolBatchedGraphLayer(nn.Module):

    def __init__(self, args, input_dim, assign_dim, output_feat_dim, activation, dropout, aggregator_type):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.args = args
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        if self.args.pool_gnn == 'graphsage':
            self.pool_gc = GraphSageLayer(input_dim, assign_dim, activation, dropout, aggregator_type)
            # self.feat_gc = GraphSageLayer(input_dim, output_feat_dim, activation, dropout, aggregator_type)
        elif self.args.pool_gnn == 'gat':
            self.pool_gc = GATLayer(args, input_dim, assign_dim, activation, layer_norm=False)
        elif self.args.pool_gnn == 'gcn':
            self.pool_gc = GCNLayer(args, input_dim, assign_dim, activation)

    def forward(self, g, h, r):
        # feat = self.feat_gc(g, h)  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        feat = h
        device = feat.device
        if self.args.pool_gnn == 'graphsage':
            assign_tensor = self.pool_gc(g, h)  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        else:
            assign_tensor = self.pool_gc(g, h, r)  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(*assign_tensor)  # size = (sum_N, batch_size * N_a)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adjacency_matrix(transpose=True, ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        node_per_pool_graph = int(adj_new.size()[0] / len(g.batch_num_nodes()))
        h, adj = batch2tensor(adj_new, h, node_per_pool_graph)

        return adj_new, h


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}


    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log["s"] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log["a"] = anext.cpu().numpy()
        return xnext, anext


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class BatchedGraphSAGE(nn.Module):
    def __init__(
        self, infeat, outfeat, use_bn=True, mean=False, add_self=False
    ):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight, gain=nn.init.calculate_gain("relu")
        )

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, "bn"):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return "BN" + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()

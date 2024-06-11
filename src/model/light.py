import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn import AvgPooling, GraphConv, MaxPooling, SumPooling

import numpy as np
from copy import deepcopy

from src.model.gbp import GBP
from src.model.GNNConv import GNNlayer
from src.model.mi_nce import MI_NCE, LayerMI
from src.model.contrast import Contrast
from src.model.cross_attention import SimpleCrossAttention, CrossAttention
from src.model.layer_pool import SAGPool, DiffPoolBatchedGraphLayer, BatchedDiffPool


feat_name_dict = {'atomic_num': 9, 'hyb': 6, 'heavydegree': 5, 'heterodegree': 5, 'smarts': 32, 'partialcharge': 0}


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data, gain=1.414)
        # module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, act):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Identity() if d_in_feats == d_out_feats else nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, act, d_hidden_feats=d_out_feats*4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = self.in_proj(x) + y
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x

class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, act, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers-2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = act
    
    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers-2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats

class BondAngleBond_graph_Transformer(nn.Module):
    def __init__(self, d_feats, n_heads, n_ffn_dense_layers, feat_drop=0., attn_drop=0., act=nn.GELU()):
        super(BondAngleBond_graph_Transformer, self).__init__()
        self.d_feats = d_feats
        self.n_heads = n_heads
        self.scale = d_feats**(-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats*3)
        self.angle_norm = nn.LayerNorm(d_feats)
        self.angle_attn = nn.Linear(d_feats, n_heads, bias=False)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, act)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = act
        self.leaky_relu = nn.LeakyReLU(0.2)

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, angle_h):
        g = g.local_var()

        # attention computation
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.leaky_relu(self.qkv(new_triplet_h)).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0]*self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})    
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        # attention update
        angle_norm = self.angle_norm(angle_h)
        angle_attn = self.leaky_relu(self.angle_attn(angle_norm)).reshape(-1, self.n_heads, 1)
        g.edata['angle_attn'] = torch.sigmoid(angle_attn)
        g.edata['a'] = g.edata.pop('node_attn') + angle_attn
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata.pop('a')))

        # update node features
        g.ndata['hv'] = v
        # g.apply_edges(fn.u_mul_e('hv', 'angle_attn', 'h_tmp'))
        # g.apply_edges(fn.u_add_e('hv', 'h_tmp', 'he'))
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = (g.edata.pop('he') * g.edata.pop('sa')).view(-1, self.d_feats)
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))

        # feed-forward neural network and residual connection
        out = self.node_out_layer(triplet_h, g.ndata['agg_h'])
        return out

    def _device(self):
        return next(self.parameters()).device


class Bond2Bond_graph_Transformer(nn.Module):
    def __init__(self, d_feats, n_heads, n_ffn_dense_layers, feat_drop=0., attn_drop=0., act=nn.GELU()):
        super(Bond2Bond_graph_Transformer, self).__init__()
        self.d_feats = d_feats
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, act)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = act

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn']
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1, self.d_feats)

        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        out = self.node_out_layer(triplet_h, g.ndata['agg_h'])
        return out

    def _device(self):
        return next(self.parameters()).device


class Attn_Fusion(nn.Module):
    def __init__(self, d_input):
        super(Attn_Fusion, self).__init__()
        self.mlp = MLP(d_in_feats=d_input*2, d_out_feats=2, n_dense_layers=2, d_hidden_feats=128, act=nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h1, h2):
        h = torch.cat([h1, h2], dim=-1)
        attn = self.mlp(h)
        attn = self.softmax(attn)
        x = h1 * attn[:, 0].unsqueeze(-1) + h2 * attn[:, 1].unsqueeze(-1)

        return x


class Feat_Fusion(nn.Module):
    def __init__(self, dim, mode):
        super(Feat_Fusion, self).__init__()
        self.mode = mode

        if mode == 'cat':
            self.proj = nn.Linear(2*dim, dim)
        elif mode == 'attn':
            self.fusion = Attn_Fusion(d_input=dim)

    def forward(self, h1, h2):
        h = None
        if self.mode == 'mean':
            pass
        elif self.mode == 'max':
            pass
        elif self.mode == 'sum':
            pass
        elif self.mode == 'cat':
            h_cat = torch.cat([h1, h2], dim=-1)
            h = self.proj(h_cat)
        elif self.mode == 'attn':
            h = self.fusion(h1, h2)

        return h


class Angle_Attn(nn.Module):
    def __init__(self, d_feats, n_heads, act=nn.ReLU()):
        super(Angle_Attn, self).__init__()
        self.d_feats = d_feats
        self.n_heads = n_heads
        self.act = act

        self.mlp = MLP(d_in_feats=d_feats, d_out_feats=d_feats, n_dense_layers=2, d_hidden_feats=d_feats*2, act=act)
        self.angle_norm = nn.LayerNorm(d_feats)
        self.angle_attn = nn.Linear(d_feats, n_heads, bias=False)
        # self.angle_fc = nn.Linear(d_feats, d_feats)

    def forward(self, angle_h):
        x = self.mlp(angle_h)
        x_norm = self.angle_norm(x)
        attn = self.angle_attn(x_norm)
        # h = self.angle_fc(x_norm)
        return x, attn


class LiGhT(nn.Module):
    def __init__(self, args, d_hidden, n_layer=2, n_heads=4, n_ffn_dense_layers=4, feat_drop=0., attn_drop=0., act=nn.GELU()):
        super(LiGhT, self).__init__()
        self.args = args
        self.graph_type = args.graph_type
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        # Angle Attention
        if self.args.embed_type == 'float':
            self.angle_emb = BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden,
                                               input_drop=self.args.input_drop)
        elif self.args.embed_type == 'int':
            self.angle_emb = BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop)
        elif self.args.embed_type == 'both':
            self.angle_emb = nn.ModuleList()
            self.angle_emb.append(BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop))
            self.angle_emb.append(
                BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden,
                                  input_drop=self.args.input_drop))
        self.angle_attn_layer = MLP(d_in_feats=d_hidden, d_out_feats=d_hidden, n_dense_layers=2, act=act)

        # Molecule Transformer Layers
        if self.args.pool == 'global_SAGPool':
            self.mol_T_layers = nn.ModuleList([
                BondAngleBond_graph_Transformer(d_hidden, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)
                for _ in range(n_layer)
            ])
            self.pool = SAGPool(d_hidden, ratio=args.pool_ratio)
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    self.pool_layers.append(SAGPool(d_hidden, ratio=args.pool_ratio))

        # # Global DiffPool
        elif self.args.pool == 'global_DiffPool':
            self.mol_T_layers = nn.ModuleList([
                BondAngleBond_graph_Transformer(d_hidden, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)
                for _ in range(n_layer)
            ])
            self.pool = DiffPoolBatchedGraphLayer(args=args, input_dim=d_hidden, assign_dim=args.assign_node,
                                                  output_feat_dim=64,
                                                  activation=F.relu, dropout=args.feat_drop,
                                                  aggregator_type='meanpool')
            self.assign_node = args.assign_node
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    assign_node = int(self.assign_node * self.args.pool_ratio)
                    self.pool_layers.append(BatchedDiffPool(d_hidden, assign_node, d_hidden))
                    self.assign_node = assign_node

        else:
            self.mol_T_layers = nn.ModuleList([
                BondAngleBond_graph_Transformer(d_hidden, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)
                for _ in range(n_layer)
            ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        if self.args.graph_pool == 'sum':
            self.readout = SumPooling()
        elif self.args.graph_pool == 'mean':
            self.readout = AvgPooling()
        elif self.args.graph_pool == 'max':
            self.readout = MaxPooling()
        self.act = act

    def forward(self, graph=None, triplet_h=None):
        if self.args.embed_type == 'both':
            angle_h = self.angle_emb[0](graph.edata['bond_angle']) + self.angle_emb[1](graph.edata['bond_angle'])
        else:
            angle_h = self.angle_emb(graph.edata['bond_angle'])
        angle_h = self.angle_attn_layer(angle_h)

        hidden_h = []
        # Global SAGPool
        if self.args.pool == 'global_SAGPool':
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                if i < self.n_layer - 1:
                    feat = self.readout(graph, triplet_h)
                    hidden_h.append(feat)
            graph, triplet_h, _ = self.pool(graph, triplet_h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer - 1):
                    graph, triplet_h, _ = self.pool_layers[p](graph, triplet_h)
            out = self.readout(graph, triplet_h)
            hidden_h.append(out)
        # Global DiffPool
        elif self.args.pool == 'global_DiffPool':
            hidden_h = []
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                if i < self.n_layer - 1:
                    feat = self.readout(graph, triplet_h)
                    hidden_h.append(feat)
            adj, h = self.pool(graph, triplet_h, angle_h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer - 1):
                    adj, h = self.pool_layers[p](adj, h, angle_h)
                    h = self.feat_dropout(h)
            out = torch.mean(h, dim=1)
            hidden_h.append(out)

        else:
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                feat = self.readout(graph, triplet_h)
                hidden_h.append(feat)

        return hidden_h

    def _device(self):
        return next(self.parameters()).device


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """

    def __init__(self, args):
        super(ConvPoolBlock, self).__init__()
        self.args = args
        d_hidden = args.hidden_size
        self.conv = GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=args.n_head, alpha=0.05,
                             hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                             negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
        self.pool = SAGPool(d_hidden, ratio=args.pool_ratio)

    def forward(self, graph, feature, angle_h):
        graph.edata['angle_h'] = angle_h
        out = F.relu(self.conv(graph, feature, angle_h))
        graph, out, _ = self.pool(graph, out)
        return graph, out, graph.edata.pop('angle_h')

class GNNModel(nn.Module):
    def __init__(self, args, d_hidden, n_layer=2, n_heads=4, n_ffn_dense_layers=4, feat_drop=0., attn_drop=0., act=nn.GELU()):
        super(GNNModel, self).__init__()
        self.args = args
        self.graph_type = args.graph_type
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        # Angle Attention
        if self.args.embed_type == 'float':
            self.angle_emb = BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden, input_drop=self.args.input_drop)
        elif self.args.embed_type == 'int':
            self.angle_emb = BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop)
        elif self.args.embed_type == 'both':
            self.angle_emb = nn.ModuleList()
            self.angle_emb.append(BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop))
            self.angle_emb.append(BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden, input_drop=self.args.input_drop))

        # Molecule GNN
        # # Global SAGPool
        if self.args.pool == 'global_SAGPool':
            self.mol_T_layers = nn.ModuleList([
                GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=n_heads, alpha=0.05,
                         hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=feat_drop, attn_drop=attn_drop,
                         negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
                for _ in range(n_layer)
            ])
            self.pool = SAGPool(d_hidden, ratio=args.pool_ratio)
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    self.pool_layers.append(SAGPool(d_hidden, ratio=args.pool_ratio))

        # # Global DiffPool
        elif self.args.pool == 'global_DiffPool':
            self.mol_T_layers = nn.ModuleList([
                GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=n_heads, alpha=0.05,
                         hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=feat_drop, attn_drop=attn_drop,
                         negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
                for _ in range(n_layer)
            ])
            self.pool = DiffPoolBatchedGraphLayer(args=args, input_dim=d_hidden, assign_dim=args.assign_node,
                                                  output_feat_dim=64,
                                                  activation=nn.LeakyReLU(args.leaky_relu), dropout=args.feat_drop,
                                                  aggregator_type='meanpool')
            self.assign_node = args.assign_node
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    assign_node = int(self.assign_node * self.args.pool_ratio)
                    self.pool_layers.append(BatchedDiffPool(d_hidden, assign_node, d_hidden))
                    self.assign_node = assign_node

        # # hierarchical SAGPool
        elif self.args.pool == 'hier_SAGPool':
            self.mol_T_layers = nn.ModuleList([ConvPoolBlock(args=self.args) for _ in range(n_layer)])
        # # hierarchical SAGPool + DiffPool
        elif self.args.pool == 'hier_DiffPool':
            self.mol_T_layers = nn.ModuleList([ConvPoolBlock(args=self.args) for _ in range(n_layer-1)])
            self.mol_T_layers.append(
                GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=n_heads, alpha=0.05,
                         hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=feat_drop, attn_drop=attn_drop,
                         negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
            )
            self.pool = DiffPoolBatchedGraphLayer(input_dim=d_hidden, assign_dim=args.assign_node,
                                                  output_feat_dim=64,
                                                  activation=F.relu, dropout=args.feat_drop,
                                                  aggregator_type='meanpool')
            self.assign_node = args.assign_node
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    assign_node = int(self.assign_node * self.args.pool_ratio)
                    self.pool_layers.append(BatchedDiffPool(d_hidden, assign_node, d_hidden))
                    self.assign_node = assign_node
        # No Pooling operation
        else:
            self.mol_T_layers = nn.ModuleList([
                GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=n_heads, alpha=0.05,
                         hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=feat_drop, attn_drop=attn_drop,
                         negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
                for _ in range(n_layer)
            ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        if self.args.graph_pool == 'sum':
            self.readout = SumPooling()
        elif self.args.graph_pool == 'mean':
            self.readout = AvgPooling()
        elif self.args.graph_pool == 'max':
            self.readout = MaxPooling()
        self.act = act

    def forward(self, graph=None, triplet_h=None):
        # Angle embedding
        if self.args.embed_type == 'both':
            angle_h = self.angle_emb[0](graph.edata['bond_angle']) + self.angle_emb[1](graph.edata['bond_angle'])
        else:
            angle_h = self.angle_emb(graph.edata['bond_angle'])

        hidden_h = []
        # Global SAGPool
        if self.args.pool == 'global_SAGPool':
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                if i < self.n_layer - 1:
                    feat = self.readout(graph, triplet_h)
                    hidden_h.append(feat)
            graph, triplet_h, _ = self.pool(graph, triplet_h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer - 1):
                    graph, triplet_h, _ = self.pool_layers[p](graph, triplet_h)
            feat = self.readout(graph, triplet_h)
            hidden_h.append(feat)
        # Global DiffPool
        elif self.args.pool == 'global_DiffPool':
            hidden_h = []
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                if i < self.n_layer - 1:
                    feat = self.readout(graph, triplet_h)
                    hidden_h.append(feat)
            adj, h = self.pool(graph, triplet_h, angle_h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer - 1):
                    adj, h = self.pool_layers[p](adj, h)
                    h = self.feat_dropout(h)
            feat = torch.mean(h, dim=1)
            hidden_h.append(feat)

        # Hierarchical SAGPool
        elif self.args.pool == 'hier_SAGPool':
            for i in range(self.n_layer):
                graph, triplet_h, angle_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                feat = self.readout(graph, triplet_h)
                hidden_h.append(feat)

        # Hierarchical SAGPool + DiffPool
        elif self.args.pool == 'hier_DiffPool':
            for i in range(self.n_layer-1):
                graph, triplet_h, angle_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                feat = self.readout(graph, triplet_h)
                hidden_h.append(feat)
            triplet_h = self.mol_T_layers[self.n_layer-1](graph, triplet_h, angle_h)
            adj, h = self.pool(graph, triplet_h)
            h = self.feat_dropout(h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer-1):
                    adj, h = self.diffpool_layers[p](adj, h)
                    h = self.feat_dropout(h)
            feat = self.readout(graph, triplet_h)
            hidden_h.append(feat)
        # No pooling operation
        else:
            for i in range(self.n_layer):
                triplet_h = self.mol_T_layers[i](graph, triplet_h, angle_h)
                feat = self.readout(graph, triplet_h)
                hidden_h.append(feat)
        return hidden_h

    def _device(self):
        return next(self.parameters()).device


class AtomEmbedding(nn.Module):
    def __init__(self, args, d_hidden, input_drop):
        super(AtomEmbedding, self).__init__()
        self.args = args
        self.hsize = args.hidden_size

        if self.args.init_emb:
            self.embed_list = nn.ModuleList()
            for name in feat_name_dict.keys():
                if name == 'partialcharge':
                    continue
                embed = nn.Embedding(feat_name_dict[name], self.args.embed_dim)
                self.embed_list.append(embed)

            centers = np.arange(0, 2, 0.1)
            gamma = 10.
            self.rbf = RBF(centers, gamma, device=self.args.device)
            self.charge_linear = nn.Linear(len(centers), self.args.embed_dim)

        self.in_proj = nn.Linear(self.args.embed_dim*2, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)


    def forward(self, pair_node_feats):
        if self.args.init_emb:
            emb_h = 0.
            for i, name in enumerate(feat_name_dict.keys()):
                if name == 'partialcharge':
                    continue
                emb_h += self.embed_list[i](pair_node_feats[name]).view(-1, self.args.embed_dim * 2)
            rbf_x0, rbf_x1 = self.rbf(pair_node_feats['partialcharge'][:, 0]), self.rbf(pair_node_feats['partialcharge'][:, 1])
            rbf_x = torch.stack([rbf_x0, rbf_x1], dim=1)
            rbf_h = self.charge_linear(rbf_x).view(-1, self.args.embed_dim*2)
            pair_node_h = self.in_proj(emb_h + rbf_h)
            h = self.input_dropout(pair_node_h)
        else:
            feat = pair_node_feats.view(pair_node_feats.shape[0], -1)
            pair_node_h = self.in_proj(feat)
            h = self.input_dropout(pair_node_h)
        return h


class BondEmbedding(nn.Module):
    def __init__(self, cut_dist, embed_dim, d_hidden, input_drop):
        super(BondEmbedding, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embed = nn.Embedding(cut_dist + 1, embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, dist_feat):
        x = torch.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).long()
        edge_h = self.dist_embed(x)
        edge_h = self.in_proj(edge_h)
        return self.input_dropout(edge_h)


class BondAngleEmbedding(nn.Module):
    def __init__(self, n_angle, embed_dim, d_hidden, input_drop):
        super(BondAngleEmbedding, self).__init__()
        self.n_angle = n_angle
        self.angle_unit = torch.FloatTensor([np.pi])[0] / n_angle
        self.angle_embed = nn.Embedding(n_angle + 1, embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, angle_feat):
        angle_domain = angle_feat / self.angle_unit
        x = torch.clip(angle_domain.squeeze(), 1.0, self.n_angle-1e-6).long()
        angle_h = self.angle_embed(x)
        angle_h = self.in_proj(angle_h)
        return self.input_dropout(angle_h)


class TripletEmbedding(nn.Module):
    def __init__(self, d_hidden, act=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_hidden*2, d_hidden, 2, act)

    def forward(self, node_h, edge_h):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        return triplet_h


class Bar_Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super(Bar_Predictor, self).__init__()
        self.mlp = MLP(d_in_feats=d_input, d_out_feats=128, n_dense_layers=2, act=nn.ReLU(), d_hidden_feats=512)
        self.act = nn.ReLU()
        self.reg = nn.Linear(128, 1)
        self.cls = nn.Linear(128, d_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_i, h_j):
        h = torch.cat([h_i, h_j], dim=-1)
        h_hidden = self.act(self.mlp(h))
        reg = self.reg(h_hidden)
        cls = self.softmax(self.cls(h_hidden))

        return reg.flatten(), cls


class Blr_Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super(Blr_Predictor, self).__init__()
        self.mlp = MLP(d_in_feats=d_input, d_out_feats=128, n_dense_layers=2, act=nn.ReLU(), d_hidden_feats=512)
        self.act = nn.ReLU()
        self.reg = nn.Linear(128, 1)
        self.cls = nn.Linear(128, d_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_hidden = self.act(self.mlp(x))
        reg = self.reg(x_hidden)
        cls = self.softmax(self.cls(x_hidden))

        return reg.flatten(), cls


class LiGhTPredictor(nn.Module):
    def __init__(self, args, d_hidden=256, n_layer=2, n_heads=4, n_ffn_dense_layers=2, input_drop=0., feat_drop=0.,
                 attn_drop=0., readout_mode='mean'):
        super(LiGhTPredictor, self).__init__()
        self.args = args
        self.graph_type = args.graph_type
        self.d_hidden = d_hidden
        self.readout_mode = readout_mode
        self.act = nn.ReLU()

        # Input
        self.node_emb = AtomEmbedding(args, d_hidden, input_drop)
        if self.args.embed_type == 'float':
            self.bond_len_emb = BondFloatRBF(args=args, bond_float_names=['bond_length'], embed_dim=d_hidden, input_drop=input_drop)
        elif self.args.embed_type == 'int':
            self.bond_len_emb = BondEmbedding(args.inner_cutoff, args.embed_dim, d_hidden, input_drop)
        elif self.args.embed_type == 'both':
            self.bond_len_emb = nn.ModuleList()
            self.bond_len_emb.append(BondEmbedding(args.inner_cutoff, args.embed_dim, d_hidden, input_drop))
            self.bond_len_emb.append(BondFloatRBF(args=args, bond_float_names=['bond_length'], embed_dim=d_hidden, input_drop=input_drop))
        self.triplet_emb = TripletEmbedding(d_hidden, act=self.act)  # GELU

        # Model
        if self.args.transformer:
            self.model = LiGhT(args, d_hidden, n_layer, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act=self.act)
        else:
            self.model = GNNModel(args, d_hidden, n_layer, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act=self.act)

        # Prediction module
        if self.args.readout == 'concat':
            in_dim = out_dim = d_hidden * n_layer
        else:
            in_dim = out_dim = d_hidden
        if self.args.readout == 'gru':
            self.out = nn.GRU(in_dim, out_dim, batch_first=True)
        elif self.args.readout == 'lstm':
            self.out = nn.LSTM(in_dim, out_dim, batch_first=True)
        elif self.args.readout == 'bi-gru':
            self.out = nn.GRU(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
        elif self.args.readout == 'bi-lstm':
            self.out = nn.LSTM(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
        elif self.args.readout == 'linear':
            self.out = nn.Linear(args.n_layer, 1)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # self.apply(lambda module: init_params(module))

    def forward(self, graph):

        # Input
        if self.args.init_emb:
            begin_end_fea = dict()
            for name in feat_name_dict:
                begin_end_fea[name] = graph.ndata[name]
        else:
            begin_end_fea = graph.ndata['begin_end_fea']
        bond_len = graph.ndata['bond_len']
        node_h = self.node_emb(begin_end_fea)
        if self.args.embed_type == 'both':
            bond_len_h = self.bond_len_emb[0](bond_len) + self.bond_len_emb[1](bond_len)
        else:
            bond_len_h = self.bond_len_emb(bond_len)
        triplet_h = self.triplet_emb(node_h, bond_len_h)

        # Model
        feat = self.model(graph=graph, triplet_h=triplet_h)
        # readout
        readout = None
        if self.args.readout == 'sum':
            readout = torch.stack(feat).sum(dim=0)
        elif self.args.readout == 'mean':
            readout = torch.stack(feat).mean(dim=0)
        elif self.args.readout == 'max':
            readout = torch.stack(feat).max(dim=0)[0]
        elif self.args.readout == 'concat':
            readout = torch.concat(feat, dim=-1)
        elif self.args.readout == 'last':
            readout = feat[-1]
        elif self.args.readout == 'linear':
            feat = torch.stack(feat, dim=-1)
            readout = self.out(feat).squeeze(dim=0)
        elif self.args.readout == 'gru':
            feat = torch.stack(feat, dim=1)
            _, hidden = self.out(feat)
            readout = hidden.squeeze(dim=0)
        elif self.args.readout == 'lstm':
            feat = torch.stack(feat, dim=1)
            _, (hidden, _) = self.out(feat)
            readout = self.out(feat).squeeze(dim=0)
        elif self.args.readout == 'bi-gru':
            feat = torch.stack(feat, dim=1)
            _, hidden = self.out(feat)
            readout = hidden.permute(1, 0, 2).reshape(feat.shape[0], -1)
        elif self.args.readout == 'bi-lstm':
            feat = torch.stack(feat, dim=1)
            _, (hidden, _) = self.out(feat)
            readout = hidden.permute(1, 0, 2).reshape(feat.shape[0], -1)

        # Predict
        pred = self.predictor(readout)
        return pred


class RBF(nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, device='cpu'):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.FloatTensor(centers), [1, -1]).to(device)
        self.gamma = torch.FloatTensor([gamma]).to(device)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        # self.canters = self.cent
        return torch.exp(-self.gamma * torch.square(x - self.centers))
    
    
class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, args, bond_float_names, embed_dim, input_drop, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.args = args
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, args.cutoff, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.args.device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), args.hidden_size)
            self.linear_list.append(linear)
        self.input_drop = nn.Dropout(input_drop)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            # x = bond_float_features[name]
            rbf_x = self.rbf_list[i](bond_float_features)
            out_embed += self.linear_list[i](rbf_x)
        return self.input_drop(out_embed)

class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, args, bond_angle_float_names, embed_dim, input_drop=0., rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.args = args
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.args.device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), args.hidden_size)
            self.linear_list.append(linear)
        self.input_drop = nn.Dropout(input_drop)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            # x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](bond_angle_float_features)
            out_embed += self.linear_list[i](rbf_x)
        return self.input_drop(out_embed)


class MI_Projector(nn.Module):
    def __init__(self, args, dim):
        super(MI_Projector, self).__init__()
        # self.cons = Contrast(tau=args.tau)
        self.args = args

        self.mi_nce = MI_NCE(num_feature=dim, mi_hid=args.mi_hid)
        self.mi_nce1 = MI_NCE(num_feature=dim, mi_hid=args.mi_hid)
        self.mi_nce2 = MI_NCE(num_feature=dim, mi_hid=args.mi_hid)

    def forward(self, v1, v2, h):
        vs = dict()
        v = self.mi_nce1(h)
        v_1 = self.mi_nce1(v1)
        v_2 = self.mi_nce2(v2)
        vs['v'], vs['v_1'], vs['v_2'] = v, v_1, v_2
        return vs

        # mi_v1v2 = self.cons.cal(v_1, v_2)
        # mi_vv1 = self.cons.cal(v, v_1)
        # mi_vv2 = self.cons.cal(v, v_2)
        # mi_loss = mi_v1v2 + mi_vv1 + mi_vv2

        # return mi_loss
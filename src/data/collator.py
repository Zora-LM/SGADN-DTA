import dgl
import torch
import numpy as np
import pickle
import random
from torch.utils.data import Dataset

from src.data.featurizer import mol2graph, mask_geognn_graph

class MoleculeDataset(Dataset):
    def __init__(self, graph_path):
        self.graph_path = graph_path
        graphs, labels = self.load()
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return graphs and label. """
        return self.graphs[idx], self.labels[idx]

    def load(self):
        """ Load the generated graphs. """
        print('Loading processed complex data...')
        with open(self.graph_path, 'rb') as f:
            graphs, labels = pickle.load(f)
        return graphs, labels


def preprocess_batch(batch_num, data_list, ssl_tasks=None):
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)

    Ba_bond_i, Ba_bond_j, Ba_bond_angle, Bl_bond, Bl_bond_length = [], [], [], [], []
    for i, data in enumerate(data_list):
        bond_node_count = cs_num[i]
        if 'Bar' in ssl_tasks:
            Ba_bond_i.append(data['Ba_bond_i'] + bond_node_count)
            Ba_bond_j.append(data['Ba_bond_j'] + bond_node_count)
            Ba_bond_angle.append(data['Ba_bond_angle'])
        if 'Blr' in ssl_tasks:
            Bl_bond.append(data['Bl_bond_node'] + bond_node_count)
            Bl_bond_length.append(data['Bl_bond_length'])

    feed_dict = dict()
    if 'Bar' in ssl_tasks:
        feed_dict['Ba_bond_i'] = torch.LongTensor(np.concatenate(Ba_bond_i, 0).reshape(-1))
        feed_dict['Ba_bond_j'] = torch.LongTensor(np.concatenate(Ba_bond_j, 0).reshape(-1))
        feed_dict['Ba_bond_angle'] = torch.FloatTensor(np.concatenate(Ba_bond_angle, 0).reshape(-1, 1))
    if 'Blr' in ssl_tasks:
        feed_dict['Bl_bond'] = torch.LongTensor(np.concatenate(Bl_bond, 0).reshape(-1))
        feed_dict['Bl_bond_length'] = torch.FloatTensor(np.concatenate(Bl_bond_length, 0).reshape(-1, 1))

    # add_factors = np.concatenate([[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)], axis=-1)
    return feed_dict


class Collator_fn(object):
    def __init__(self, args, training=False):
        self.args = args
        self.training = training

    def __call__(self, samples):
        graphs, labels = map(list, zip(*samples))
        pk_values = torch.FloatTensor(labels)

        if self.training & self.args.is_mask:
            masked_graphs = []
            for g in graphs:
                p = random.random()
                if self.args.p > p:
                    masked_g = mask_geognn_graph(g, mask_ratio=self.args.mask_ratio)
                    masked_graphs.append(masked_g)
                else:
                    masked_graphs.append(g)
            batched_graph = dgl.batch(masked_graphs)
        else:
            batched_graph = dgl.batch(graphs)

        return batched_graph, pk_values


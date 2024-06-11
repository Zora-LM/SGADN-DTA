import os
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import dgl
from src.data.featurizer import mol2graph
from src.data.featurizer import Compound3DKit, Featurizer, gen_feature, cons_lig_pock_graph_with_spatial_context, bond_node_prepare

###################################################################################
prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]

fea_name_list = ['atomic_num', 'hyb', 'heavydegree', 'heterodegree', 'partialcharge', 'smarts']


class ComplexDataset:
    def __init__(self, args, mol_path, save_path=None, set_path=None, save_file=True):
        self.args = args
        self.mol_path = mol_path
        self.save_file = save_file
        self.save_path = save_path

        data = pd.read_csv(set_path)
        if args.DEBUG:
            data = data[:10]
        # print('Dataset size: ', len(data))
        self.pdbid = data['pdbid'].tolist()
        self.affinity = data['-logKd/Ki'].tolist()

        self.labels = []
        self.graphs = []

        self.process_data()

    def process_data(self):
        """ Generate complex interaction graphs. """

        # Check cache file
        if os.path.exists(self.save_path):
            print('The prossed dataset is saved in ', self.save_path)
        else:
            print('Processing raw protein-ligand complex data...')
            for i, (pdb_name, pk) in enumerate(tqdm(zip(self.pdbid, self.affinity))):
                # print('pdb name ', pdb_name)
                graph = self.build_graph(pdb_name)
                self.graphs.append(graph)
                self.labels.append(pk)

            self.labels = np.array(self.labels)
            if self.save_file:
                self.save()

    def build_graph(self, name):
        featurizer = Featurizer(save_molecule_codes=False)

        data = dict()
        # atomic feature generation
        lig_poc_dict = gen_feature(self.mol_path, name, featurizer)  # dict of {coords, features, atoms, edges} for ligand and pocket
        ligand = (lig_poc_dict['lig_fea'], lig_poc_dict['lig_fea_dict'], lig_poc_dict['lig_co'], lig_poc_dict['lig_atoms'], lig_poc_dict['lig_eg'])
        pocket = (lig_poc_dict['pock_fea'], lig_poc_dict['pock_fea_dict'], lig_poc_dict['pock_co'], lig_poc_dict['pock_atoms'], lig_poc_dict['pock_eg'])

        # get complex coods, features, atoms
        mol = cons_lig_pock_graph_with_spatial_context(args, ligand, pocket, add_fea=self.args.add_fea, theta=self.args.cutoff,
                                                       theta2=args.inner_cutoff, keep_pock=False, pocket_spatial=True)
        num_atoms, coords, features, features_dict, bond_nodes, bond_len, atoms = mol

        # get bond node (atom-atom edges), bond length
        # # strategy 1: get bond node (atom-atom edges) according to distance
        bond_nodes, bond_len = bond_node_prepare(mol, self.args.cutoff)
        # # strategy 2: use known edges as bond node
        i, j = bond_nodes[:, 0], bond_nodes[:, 1]
        pair_features_dict = dict()
        for n in fea_name_list:
            if n == 'partialcharge':
                pair_features_dict[n] = torch.FloatTensor(np.vstack([features_dict[n][i], features_dict[n][j]]).T)
            else:
                pair_features_dict[n] = torch.LongTensor(np.vstack([features_dict[n][i], features_dict[n][j]]).T)
        feat_dim = features.shape[1]
        pair_features = np.hstack([features[i], features[j]]).reshape(len(bond_nodes), -1, feat_dim)
        pair_features = torch.FloatTensor(pair_features)

        # G: bond-angle-bond graph
        data['atom_pos'] = np.array(coords, dtype='float32')
        data['edges'] = np.array(bond_nodes, dtype="int64")
        BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'], dir_type='HT')
        bab_g = dgl.graph(data=(BondAngleGraph_edges[:, 0], BondAngleGraph_edges[:, 1]), num_nodes=len(bond_nodes))
        bab_g.ndata['bond_len'] = torch.FloatTensor(bond_len)
        bab_g.edata['bond_angle'] = torch.FloatTensor(bond_angles)
        bab_g.ndata['begin_end_feat'] = pair_features
        for n in fea_name_list:
            bab_g.ndata[n] = pair_features_dict[n]

        return bab_g


    def save(self):
        """ Save the generated graphs. """
        print('Saving processed complex data...')
        with open(self.save_path, 'wb') as f:
            pickle.dump((self.graphs, self.labels), f)

def pairwise_atomic_types(path, processed_dict, atom_types, atom_types_):
    keys = [(i, j) for i in atom_types_ for j in atom_types]
    for name in tqdm(os.listdir(path)):
        if len(name) != 4:
            continue
        ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
        pocket = next(pybel.readfile('pdb', '%s/%s/%s_protein.pdb' % (path, name, name)))
        coords_lig = np.vstack([atom.coords for atom in ligand])
        coords_poc = np.vstack([atom.coords for atom in pocket])
        atom_map_lig = [atom.atomicnum for atom in ligand]
        atom_map_poc = [atom.atomicnum for atom in pocket]
        dm = distance_matrix(coords_lig, coords_poc)
        # print(coords_lig.shape, coords_poc.shape, dm.shape)
        ligs, pocks = dist_filter(dm, 12)
        # print(len(ligs),len(pocks))

        fea_dict = {k: 0 for k in keys}
        for x, y in zip(ligs, pocks):
            x, y = atom_map_lig[x], atom_map_poc[y]  # x-atom.atomicnum of ligand, y-atom.atomicnum of pocket
            if x not in atom_types or y not in atom_types_: continue
            fea_dict[(y, x)] += 1

        processed_dict[name]['type_pair'] = list(fea_dict.values())

    return processed_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preprocess for PDBBind")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--DEBUG", action='store_true', default=False, help='Debug mode')
    parser.add_argument("--n_repeat", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='PDBBind2020')
    parser.add_argument('--data_path', default='/media/data2/lm/Experiments/3D_DTI/dataset/', type=str)
    parser.add_argument('--split_type', type=str, default='temporal_split', help='random split or temporal split')
    parser.add_argument("--cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--inner_cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--n_angle", type=int, default=6, help='number of angle domains')
    parser.add_argument("--add_fea", type=int, default=0, help='add feature manner, 1, 2, others')
    parser.add_argument("--save_dir", type=str, default='../dataset/processed/{}/{}/', help='Processed dataset path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.save_dir = args.save_dir.format(args.dataset, args.split_type)
    os.makedirs(args.save_dir, exist_ok=True)

    # preprocess train set and val set
    for i in [2014, 2015, 2016, 2017, 2018]:
        print('This is year ', i)
        for s in ['train', 'val']:
            ComplexDataset(args, mol_path=args.data_path + 'original/pdbbind_refined_core/',
                           set_path=f'{args.data_path}/dataset_split/{args.split_type}/{s}_year_{i}.csv',
                           save_path=args.save_dir + f'year{i}_{s}_{args.cutoff}_{args.n_angle}_graph.pkl')

    # Preprocess core set
    ComplexDataset(args, mol_path=args.data_path + 'original/pdbbind_refined_core/',
                   set_path=f'{args.data_path}/dataset_split/{args.split_type}/test.csv',
                   save_path=args.save_dir + f'test_{args.cutoff}_{args.n_angle}_graph.pkl')



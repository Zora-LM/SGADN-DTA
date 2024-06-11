import numpy as np
import torch
import dgl
import pickle
from openbabel import pybel, openbabel
from scipy.spatial import distance_matrix, distance
from sklearn.metrics import pairwise_distances
from itertools import permutations
import hashlib
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


fea_name_list = ['atomic_num', 'hyb', 'heavydegree', 'heterodegree', 'partialcharge', 'smarts']

def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)

class Featurizer():
    """
    This is from SIGN.

    Calcaulates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns

    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self, atom_codes=None, atom_labels=None, named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None, smarts_labels=None):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).

        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead' % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: %s instead of %s' % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom) if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError('named_properties must be in pybel.Atom attributes,  %s was given at position %s'
                                     % (prop_id, prop))
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            # hybridization, Number of non-hydrogen atoms attached, Number of heteroatoms attached, Partial charge
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree', 'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
            self.smarts_labels = smarts_labels
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s instead of %s' % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given' % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))
        idx_value = np.zeros(len(molecule.atoms))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))), dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
            idx_value[atoms_with_prop] += 2 ** pattern_id
        return features, idx_value


    def get_features(self, molecule, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True, you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given' % type(molcode))

        coords = []
        features = []
        heavy_atoms = []
        feature_dict = {'atomic_num': []}
        props = {prop: [] for prop in self.NAMED_PROPS}
        feature_dict.update(props)

        for i, atom in enumerate(molecule): # iterates over the atoms of the molecule
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:  # atom.atomicnum: Atomic number
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))
                # feature dict
                feature_dict['atomic_num'].append(self.ATOM_CODES[atom.atomicnum])
                for prop in self.NAMED_PROPS:
                    feature_dict[prop].append(atom.__getattribute__(prop))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features, molcode * np.ones((len(features), 1))))
        smarts_result = self.find_smarts(molecule)
        smarts_fea, smarts_idx = smarts_result[0][heavy_atoms], smarts_result[1][heavy_atoms]
        features = np.hstack([features, smarts_fea])
        feature_dict['smarts'] = smarts_idx

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        for name in fea_name_list:
            if name in ['partialcharge']:
                feature_dict[name] = np.array(feature_dict[name], dtype=np.float32)
            else:
                feature_dict[name] = np.array(feature_dict[name], dtype=np.int)

        return coords, features, feature_dict


    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer



## function -- pocket graph
def gen_pocket_graph(pocket):
    edge_l = []
    idx_map = [-1] * (len(pocket.atoms) + 1)
    idx_new = 0
    for atom in pocket:
        edges = []
        a1_sym = atom.atomicnum
        a1 = atom.idx
        if a1_sym == 1:
            continue
        idx_map[a1] = idx_new
        idx_new += 1
        for natom in openbabel.OBAtomAtomIter(atom.OBAtom): # Iterator over the atoms attached to an atom.
            if natom.GetAtomicNum() == 1:
                continue
            a2 = natom.GetIdx()
            bond = openbabel.OBAtom.GetBond(natom, atom.OBAtom)
            bond_t = bond.GetBondOrder()
            edges.append((a1, a2, bond_t))
        edge_l += edges
    edge_l_new = []
    for a1, a2, t in edge_l:
        a1_, a2_ = idx_map[a1], idx_map[a2]
        assert ((a1_ != -1) & (a2_ != -1))
        edge_l_new.append((a1_, a2_, t))
    return np.array(edge_l_new, dtype=np.int64)


def pocket_atom_num_from_mol2(path, name):
    n = 0
    with open('%s/%s/%s_pocket.mol2' % (path, name, name)) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '<TRIPOS>BOND' in line or cont[7] == 'HOH':
                break
            n += int(cont[5][0] != 'H')
    return n


def edge_ligand_pocket(dist_matrix, lig_size, theta=4, theta2=4, keep_pock=False, reset_idx=True):
    def get_edge_list(pos):
        dist_list = dist_matrix[pos]
        ligand_list, pocket_list = pos
        if reset_idx:
            edge_list_ = [(x, node_map[y]) for x, y in zip(ligand_list, pocket_list)]
        else:
            edge_list_ = [(x, y) for x, y in zip(ligand_list, pocket_list)]

        edge_list_ += [(y, x) for x, y in edge_list_]
        dist_list_ = np.concatenate([dist_list, dist_list])
        return edge_list_, dist_list_

    pos = np.where(dist_matrix <= theta)
    ligand_list, pocket_list = pos
    if keep_pock:
        node_list = range(dist_matrix.shape[1])
    else:
        node_list = sorted(list(set(pocket_list)))
    node_map = {node_list[i]: i + lig_size for i in range(len(node_list))}

    edge_list, dist_list = get_edge_list(pos)

    inner_pos = np.where(dist_matrix <= theta2)
    inner_edge_list, inner_dist_list = get_edge_list(inner_pos)

    return dist_list, edge_list, node_map, inner_dist_list, inner_edge_list


def add_identity_fea(lig_fea, pock_fea, comb=1):
    if comb == 1:
        lig_fea = np.hstack([lig_fea, [[1]] * len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[-1]] * len(pock_fea)])
    elif comb == 2:
        lig_fea = np.hstack([lig_fea, [[1, 0]] * len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[0, 1]] * len(pock_fea)])
    # elif comb == 3:
    #     lig_fea = []
    #     np.nonzero(lig_fea[:, :9])[1]
    #     np.nonzero(lig_fea[:, 9:12])[1]
    #     row, col = np.nonzero(lig_fea[:, 13:])
    #
    #     old_idx = 0
    #     new_idx = 0
    #     for i, idx in enumerate(row):
    #         if idx == old_idx:
    #             new_idx += 2 ** col[i]
    #         else:
    #             new_idx += 2 ** col[i]
    #         old_idx = idx



    else:
        lig_fea = np.hstack([lig_fea, [[0] * lig_fea.shape[1]] * len(lig_fea)])
        if len(pock_fea) > 0:
            pock_fea = np.hstack([[[0] * pock_fea.shape[1]] * len(pock_fea), pock_fea])

    return lig_fea, pock_fea


## function -- feature
def gen_feature(path, name, featurizer):
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
    ligand_coords, ligand_features, lig_fea_dict = featurizer.get_features(ligand, molcode=1)  # .get_features: Get coordinates and features for all heavy atoms in the molecule
    pocket = next(pybel.readfile('mol2', '%s/%s/%s_pocket.mol2' % (path, name, name)))
    pocket_coords, pocket_features, poc_fea_dict = featurizer.get_features(pocket, molcode=-1)  # molcode: Molecule type, 1-ligand, -1-protein. do not use in this work
    node_num = pocket_atom_num_from_mol2(path, name)
    pocket_coords = pocket_coords[:node_num]
    pocket_features = pocket_features[:node_num]
    for n in fea_name_list:
        poc_fea_dict[n] = poc_fea_dict[n][:node_num]

    try:
        assert (ligand_features[:, charge_idx] != 0).any()
        assert (pocket_features[:, charge_idx] != 0).any()
        assert (ligand_features[:, :9].sum(1) != 0).all()
    except:
        print(name)
    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1:
            lig_atoms.append(atom.atomicnum)
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum)

    pock_atoms = pock_atoms[:node_num]
    assert len(lig_atoms) == len(ligand_features) and len(pock_atoms) == len(pocket_features)

    ligand_edges = gen_pocket_graph(ligand)
    pocket_edges = gen_pocket_graph(pocket)
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_fea_dict': lig_fea_dict, 'lig_atoms': lig_atoms, 'lig_eg': ligand_edges,
            'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_fea_dict': poc_fea_dict, 'pock_atoms': pock_atoms, 'pock_eg': pocket_edges}


def cons_spatial_gragh(dist_matrix, theta=5):
    pos = np.where((dist_matrix <= theta) & (dist_matrix != 0))
    src_list, dst_list = pos
    dist_list = dist_matrix[pos]
    edges = [(x, y) for x, y in zip(src_list, dst_list)]
    return edges, dist_list


def cons_mol_graph(edges, feas):
    size = feas.shape[0]
    edges = [(x, y) for x, y, t in edges]
    return size, feas, edges

def pocket_subgraph(node_map, edge_list): #, pock_dist):
    edge_l = []
    dist_l = []
    node_l = set()
    for coord in edge_list:
        x, y = coord
        if x in node_map and y in node_map:
            x, y = node_map[x], node_map[y]
            edge_l.append((x, y))
            # dist_l.append(dist)
            # node_l.add(x)
            # node_l.add(y)
    # dist_l = np.array(dist_l)
    return edge_l#, dist_l


def cons_lig_pock_graph_with_spatial_context(args, ligand, pocket, add_fea=2, theta=5, theta2=4, keep_pock=False, pocket_spatial=True):
    lig_fea, lig_fea_dict, lig_coord, lig_atoms_raw, lig_edge = ligand
    pock_fea, pock_fea_dict, pock_coord, pock_atoms_raw, pock_edge = pocket

    # inter-relation between ligand and pocket
    lig_size = lig_fea.shape[0]
    dm = distance_matrix(lig_coord, pock_coord)
    dist_list, edge_list, node_map, inner_dist_list, inner_edge_list = edge_ligand_pocket(dm, lig_size, theta=theta,
                                                                                          theta2=theta2, keep_pock=keep_pock)
    # construct ligand graph & pocket graph
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)

    # construct spatial context graph based on distance
    # dm = distance_matrix(lig_coord, lig_coord)
    # edges, lig_dist = cons_spatial_gragh(dm, theta=theta)
    # if pocket_spatial:
    #     dm_pock = distance_matrix(pock_coord, pock_coord)
    #     edges_pock, pock_dist = cons_spatial_gragh(dm_pock, theta=theta)
    # lig_edge = edges
    # pock_edge = edges_pock


    # map new pocket graph
    pock_size = len(node_map)
    pock_fea = pock_fea[sorted(node_map.keys())]
    pock_edge = pocket_subgraph(node_map, pock_edge) #, pock_dist)
    pock_coord_ = pock_coord[sorted(node_map.keys())]
    for n in fea_name_list:
        pock_fea_dict[n] = pock_fea_dict[n][sorted(node_map.keys())]

    # cooridates
    coords = np.vstack([lig_coord, pock_coord_]) if len(pock_coord_) > 0 else lig_coord

    # distance (bond length)
    dm_lig = distance_matrix(lig_coord, lig_coord)
    lig_edge_arr = np.array(lig_edge)
    lig_i, lig_j = lig_edge_arr[:, 0], lig_edge_arr[:, 1]
    dis_lig = dm_lig[lig_i, lig_j]

    if len(pock_edge) > 0:
        dm_poc = distance_matrix(pock_coord_, pock_coord_)
        poc_edge_arr = np.array(pock_edge) - lig_size
        poc_i, poc_j = poc_edge_arr[:, 0], poc_edge_arr[:, 1]
        dis_poc = dm_poc[poc_i, poc_j]
        dist = np.hstack([dis_lig, dis_poc, inner_dist_list])
    else:
        dist = np.hstack([dis_lig, dist_list])

    size = lig_size + pock_size
    feats_dict = dict()
    for n in fea_name_list:
        feats_dict[n] = np.hstack([lig_fea_dict[n], pock_fea_dict[n]]) if len(pock_fea_dict[n]) > 0 else lig_fea_dict[n]
    assert len(feats_dict['atomic_num']) == size
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea)
    feats = np.vstack([lig_fea, pock_fea]) if len(pock_fea) > 0 else lig_fea
    assert len(feats) == size

    # construct ligand-pocket graph
    edges = []
    edges.extend(lig_edge)
    edges.extend(pock_edge)
    edges.extend(inner_edge_list)
    edges = np.array(edges)

    lig_atoms_raw = np.array(lig_atoms_raw)
    pock_atoms_raw = np.array(pock_atoms_raw)
    pock_atoms_raw = pock_atoms_raw[sorted(node_map.keys())]
    atoms_raw = np.concatenate([lig_atoms_raw, pock_atoms_raw]) if len(pock_atoms_raw) > 0 else lig_atoms_raw

    return size, coords, feats, feats_dict, edges, dist, atoms_raw


# prepare bond nodes #
def bond_node_prepare(mol, cut_dist):
    num_atoms_d, coords, features, features_dict, edges, length, atoms = mol
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    np.fill_diagonal(dist_mat, np.inf)
    num_atoms = len(coords)

    indices = []  # bond nodes
    dist = []
    bond_pair_atom_types = []
    for i in range(num_atoms):  # number of bond atoms
        for j in range(num_atoms):
            a = dist_mat[i, j]
            if a < cut_dist:
                # at_i, at_j = atoms[i], atoms[j]  # ligand atomic num, pocket atomic num
                # if i < num_atoms_d and j >= num_atoms_d and (at_j, at_i) in pair_ids:
                #     bond_pair_atom_types += [pair_ids.index((at_j, at_i))]
                # elif i >= num_atoms_d and j < num_atoms_d and (at_i, at_j) in pair_ids:
                #     bond_pair_atom_types += [pair_ids.index((at_i, at_j))]
                # else:
                #     bond_pair_atom_types += [-1]
                indices.append([i, j])
                dist.append(a)

    indices = np.array(indices, dtype=np.int64)
    dist = np.array(dist, dtype=np.float)

    return indices, dist


def node_pairs2bond_node(mol):
    num_atoms_d, coords, features, edges, atoms = mol
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    i, j = edges[:, 0], edges[:, 1]
    dist = dist_mat[i, j]
    pair_features = np.hstack([features[i], features[j]]).reshape(len(edges), -1, features.shape[1])
    return edges, dist, pair_features


def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    def _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, node_i_index, node_j_index, node_k_index):
        # node_i_indices += [node_i_index, node_k_index]
        # node_j_indices += [node_j_index, node_j_index]
        # node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    bond_ij = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, a1, a0, b1)
                bond_ij.append([edge_i, edge_j])
                bond_ij.append([edge_j, edge_i])
            if a0 == b1:
                _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, a1, a0, b0)
                bond_ij.append([edge_i, edge_j])
                bond_ij.append([edge_j, edge_i])
            if a1 == b0:
                _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, a0, a1, b1)
                bond_ij.append([edge_i, edge_j])
                bond_ij.append([edge_j, edge_i])
            if a1 == b1:
                _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, a0, a1, b0)
                bond_ij.append([edge_i, edge_j])
                bond_ij.append([edge_j, edge_i])

    # node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    # uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    # node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_ij = np.array(bond_ij, dtype=np.int64)
    uniq_bond_ij, uniq_index = np.unique(bond_ij, return_index=True, axis=0)
    bond_i_indices, bond_j_indices = uniq_bond_ij[:, 0], uniq_bond_ij[:, 1]
    bond_angles = np.array(bond_angles)[uniq_index]
    return [bond_i_indices, bond_j_indices, bond_angles]
    # return [node_i_indices, node_j_indices, node_k_indices, bond_angles]


class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)  # relative position
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        #### add self-loop
        for i in range(E):
            super_edges.append([i, i])
            bond_angles.append(0.)
            bond_angle_dirs.append(i == i)

        super_edges = np.array(super_edges, 'int64')
        bond_angles = np.array(bond_angles, 'float32')

        return super_edges, bond_angles, bond_angle_dirs

    @staticmethod
    def get_superedge(num_atoms, edges):
        n_edges = len(edges)
        atomIDPair_to_bondId = np.ones(shape=(num_atoms, num_atoms), dtype=np.int64) * np.nan

        for i, atom_pair in enumerate(edges):
            begin_atom_id, end_atom_id = atom_pair
            atomIDPair_to_bondId[begin_atom_id, end_atom_id] = atomIDPair_to_bondId[end_atom_id, begin_atom_id] = i

        super_edges = []
        for i in range(num_atoms):
            node_ids = atomIDPair_to_bondId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            new_edges = list(permutations(node_ids, 2))
            super_edges.extend(new_edges)

        ## add self-loop
        self_edges = []
        for i in range(n_edges):
            self_edges.append([i, i])
        super_edges = np.concatenate([np.array(super_edges, np.int64), np.array(self_edges, dtype=np.int64)])

        return super_edges


def prepare_ssl_task(data):
    """
    prepare data for self-supervised learning task
    """
    bond_i, bond_j, bond_angles = get_pretrain_bond_angle(data['edges'], data['atom_pos'])
    data['Ba_bond_i'] = bond_i
    data['Ba_bond_j'] = bond_j
    data['Ba_bond_angle'] = bond_angles

    data['Bl_bond_node'] = np.array(range(len(data['edges'])))
    data['Bl_bond_length'] = np.array(data['bond_length'])

    return data


def mol2graph(args, mol_path, cutoff, add_fea, name, training):
    featurizer = Featurizer(save_molecule_codes=False)

    data = dict()
    # atomic feature generation
    lig_poc_dict = gen_feature(mol_path, name, featurizer)  # dict of {coords, features, atoms, edges} for ligand and pocket
    ligand = (lig_poc_dict['lig_fea'], lig_poc_dict['lig_fea_dict'], lig_poc_dict['lig_co'], lig_poc_dict['lig_atoms'], lig_poc_dict['lig_eg'])
    pocket = (lig_poc_dict['pock_fea'], lig_poc_dict['pock_fea_dict'], lig_poc_dict['pock_co'], lig_poc_dict['pock_atoms'], lig_poc_dict['pock_eg'])
    # get complex coods, features, atoms
    mol = cons_lig_pock_graph_with_spatial_context(args, ligand, pocket, add_fea=add_fea, theta=cutoff,
                                                   theta2=args.inner_cutoff, keep_pock=False, pocket_spatial=True)
    num_atoms, coords, features, bond_nodes, bond_len, atoms = mol
    # get bond node (atom-atom edges), bond length
    # # strategy 1: get bond node (atom-atom edges) according distance
    bond_nodes, bond_len = bond_node_prepare(mol, cutoff)
    # # strategy 2: use known edges as bond node
    i, j = bond_nodes[:, 0], bond_nodes[:, 1]
    if args.init_emb:
        pair_features = dict()
        for n in fea_name_list:
            if n == 'partialcharge':
                pair_features[n] = torch.FloatTensor(np.vstack([features[n][i], features[n][j]]).T)
            else:
                pair_features[n] = torch.LongTensor(np.vstack([features[n][i], features[n][j]]).T)
    else:
        feat_dim = features.shape[1]
        pair_features = np.hstack([features[i], features[j]]).reshape(len(bond_nodes), -1, feat_dim)
        pair_features = torch.FloatTensor(pair_features)

    data['num_atoms'] = num_atoms
    data['edges'] = np.array(bond_nodes, dtype="int64")
    data['atom_pos'] = np.array(coords, dtype='float32')
    data['bond_length'] = np.array(bond_len, dtype='float32')


    # get bond_angle graph
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'], dir_type='HT')
    bab_g = dgl.graph(data=(BondAngleGraph_edges[:, 0], BondAngleGraph_edges[:, 1]))
    if args.init_emb:
        for n in fea_name_list:
            bab_g.ndata[n] = pair_features[n]
    else:
        bab_g.ndata['begin_end_fea'] = pair_features
    bab_g.ndata['bond_len'] = torch.FloatTensor(bond_len)
    bab_g.edata['bond_angle'] = torch.FloatTensor(bond_angles)

    masked_bab_g = None
    if (training & args.is_mask):
        masked_bab_g = mask_geognn_graph(bab_g, mask_ratio=args.mask_ratio)

    if training:
        return bab_g, masked_bab_g, data
    else:
        return bab_g


def mask_geognn_graph(g, masked_node_indices=None, mask_ratio=None, mask_value=0):

    g = deepcopy(g)
    N = g.num_nodes()
    E = g.num_edges()
    full_node_indices = np.arange(N)
    full_edge_indices = np.arange(E)

    if masked_node_indices is None:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        masked_node_indices = np.random.choice(full_node_indices, size=masked_size, replace=False)

    masked_edge_indices = []
    for node_index in masked_node_indices:
        left_edge_indices = full_edge_indices[g.edges()[0] == node_index]
        right_edge_indices = full_edge_indices[g.edges()[1] == node_index]
        edge_indices = np.append(left_edge_indices, right_edge_indices)
        masked_edge_indices.append(edge_indices)
    masked_edge_indices = np.concatenate(masked_edge_indices, 0)

    for name in fea_name_list:
        if name == 'partialcharge':
            g.ndata[name][masked_node_indices] = torch.FloatTensor(np.random.choice(g.ndata[name], len(masked_node_indices)))  # mask_value
        else:
            g.ndata[name][masked_node_indices] = torch.LongTensor(np.random.choice(g.ndata[name], len(masked_node_indices)))  # mask_value
    # mask 'feat'
    v = g.ndata['feat'][masked_node_indices]
    v_new = v + 0.1 * torch.normal(mean=0, std=0.01, size=v.shape)
    g.ndata['feat'][masked_node_indices] = v_new

    for name in ['bond_len']:
        g.edata[name][masked_edge_indices] = torch.FloatTensor(np.random.choice(g.edata[name], len(masked_edge_indices)))

    return g

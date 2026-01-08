import os
import torch
import numpy as np
from scipy.spatial import KDTree
from mofdiff.common.data_utils import frac_to_cart_coords, lattice_params_to_matrix_torch
import argparse
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from torch.utils.data import Dataset
import lmdb
import pickle
import pandas as pd

bb_emb_mean = np.array([ 1.9883,  3.4131, -1.2937, -1.0239,  1.1878, -1.1955,  2.8939, -0.6505,
         1.8447, -0.5458,  1.2889, -2.0929, -1.5134,  1.4741, -0.3019, -3.1123,
        -0.0562, -1.1413,  0.0058, -3.3285,  0.9703, -0.9206, -4.2568, -0.5873,
        -0.8195, -0.1195, -0.5928,  2.8217, -0.5937, -2.8954, -0.4469, -3.6022])

bb_emb_std = np.array([2.9326, 8.1418, 7.1256, 3.7709, 3.2484, 2.8743, 4.1241, 3.7445, 2.7989,
        2.7987, 4.9791, 4.0736, 3.9020, 5.0574, 1.5935, 6.4115, 3.6939, 3.7476,
        3.9844, 6.7042, 8.8707, 4.6563, 6.1857, 4.1350, 4.2078, 3.9869, 4.1179,
        4.0798, 8.8324, 8.2764, 4.6471, 9.4339])

class BBDataset(Dataset):

    def __init__(self, lmdb_dir):

        self.processed_env = lmdb.open(lmdb_dir, subdir=False, readonly=True, lock=False)
        self._original_length = pickle.loads(self.processed_env.begin().get('length'.encode()))

    def __len__(self):
        
        return self._original_length

    def __getitem__(self, idx):

        datapoint = pickle.loads(self.processed_env.begin().get(f'{idx}'.encode()))
        return datapoint


def decode_blocks(bb_embs, all_data, kdtree):

    num_bbs = bb_embs.shape[0]
    all_coords, all_types, bb_num_vec = [], [], []
    res = []

    for i in range(num_bbs):
        bb_emb = bb_embs[i]
        target = bb_emb.cpu() * bb_emb_std + bb_emb_mean
        ret_bb = all_data[kdtree.query(target)[1]].cpu().clone()
        res.append(ret_bb)

    return res

METALS = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
          95, 96, 97, 98, 99, 100, 101, 102, 103]

metal_atomic_numbers = torch.tensor(METALS)
non_metal_atomic_numbers = torch.tensor([x for x in np.arange(100) if x not in METALS])

def get_connecting_atom_index(bbs):
    all_connecting_atoms = []
    offset = 0
    for bb in bbs:
        # relying on: cp_index is sorted, edges are double-directed
        cp_index = (bb.atom_types == 2).nonzero().flatten()
        connecting_atom_index = (
            bb.edge_index[1, (bb.edge_index[0].view(-1, 1) == cp_index).any(dim=-1)]
            + offset
        )
        offset += bb.num_atoms
        all_connecting_atoms.append(connecting_atom_index)
    all_connecting_atoms = torch.cat(all_connecting_atoms)
    return all_connecting_atoms

def feasibility_check(cg_mof_bbs):
    """
    check the matched connection criterion.
    """
    if cg_mof_bbs[0] is not None:
        atom_types = torch.cat([x.atom_types for x in cg_mof_bbs])
        connecting_atom_index = get_connecting_atom_index(cg_mof_bbs)
        connecting_atom_types = atom_types[connecting_atom_index]
        n_metal = (
            (connecting_atom_types.view(-1, 1) == metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        n_nonmetal = (
            (connecting_atom_types.view(-1, 1) == non_metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        return n_metal == n_nonmetal
    else:
        return False


def main(args):

    ret = torch.load(args.res_path, map_location='cpu')
    bb_embs = ret['types']
    num_atoms = ret['num_atoms']
    end_atoms = torch.cumsum(num_atoms, dim=0)
    start_atoms = end_atoms - num_atoms
    max_process = len(num_atoms)
    if args.max_process != -1 and args.max_process < max_process:
        max_process = args.max_process

    all_z = torch.load(args.bb_z_path, map_location='cpu')
    all_data = BBDataset(args.bb_blocks_path)
    kdtree = KDTree(all_z)
    check_list = []

    for i in tqdm(range(max_process)):
        cur_bb_embs = bb_embs[start_atoms[i]: end_atoms[i]]
        cur_num_atoms = num_atoms[i]
        ret_bbs = decode_blocks(cur_bb_embs, all_data, kdtree)
        ret_check = feasibility_check(ret_bbs)
        check_list.append(ret_check)
        
    df = pd.DataFrame(check_list, columns=['Feasibility'])

    fea_path = os.path.join(os.path.dirname(args.res_path), 'feasibility.csv')
    df.to_csv(fea_path, index=True)

    print(np.mean(check_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', required=True)
    parser.add_argument('--bb_z_path', required=True)
    parser.add_argument('--bb_blocks_path', required=True)
    parser.add_argument('--max_process', default=-1, type=int)
    args = parser.parse_args()
    main(args)

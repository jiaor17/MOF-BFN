import os
import torch
import numpy as np
from scipy.spatial import KDTree
from mofdiff.common.data_utils import frac_to_cart_coords, lattice_params_to_matrix_torch, cart_to_frac_coords
import argparse
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from torch.utils.data import Dataset
import lmdb
import pickle
from mofdiff.common.optimization import get_bb_bond_and_types, same_component_mask, get_unique_and_index
from mofdiff.common.atomic_utils import (
    compute_distance_matrix,
    frac2cart,
    cart2frac,
    remap_values,
    compute_image_flag,
    mof2cif_with_bonds
)
from torch_geometric.data import Data

from scipy.optimize import minimize, linear_sum_assignment

import pdb

bb_emb_mean = np.array([ 1.9883,  3.4131, -1.2937, -1.0239,  1.1878, -1.1955,  2.8939, -0.6505,
         1.8447, -0.5458,  1.2889, -2.0929, -1.5134,  1.4741, -0.3019, -3.1123,
        -0.0562, -1.1413,  0.0058, -3.3285,  0.9703, -0.9206, -4.2568, -0.5873,
        -0.8195, -0.1195, -0.5928,  2.8217, -0.5937, -2.8954, -0.4469, -3.6022])

bb_emb_std = np.array([2.9326, 8.1418, 7.1256, 3.7709, 3.2484, 2.8743, 4.1241, 3.7445, 2.7989,
        2.7987, 4.9791, 4.0736, 3.9020, 5.0574, 1.5935, 6.4115, 3.6939, 3.7476,
        3.9844, 6.7042, 8.8707, 4.6563, 6.1857, 4.1350, 4.2078, 3.9869, 4.1179,
        4.0798, 8.8324, 8.2764, 4.6471, 9.4339])

METALS = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
          95, 96, 97, 98, 99, 100, 101, 102, 103]

metal_atomic_numbers = torch.tensor(METALS)
non_metal_atomic_numbers = torch.tensor([x for x in np.arange(100) if x not in METALS])

class BBDataset(Dataset):

    def __init__(self, lmdb_dir):

        self.processed_env = lmdb.open(lmdb_dir, subdir=False, readonly=True, lock=False)
        self._original_length = pickle.loads(self.processed_env.begin().get('length'.encode()))

    def __len__(self):
        
        return self._original_length

    def __getitem__(self, idx):

        datapoint = pickle.loads(self.processed_env.begin().get(f'{idx}'.encode()))
        return datapoint

def assemble_coords(local_coords, rotmats, trans, bb_num_vec):
    """
    Returns:
        coords: numpy array of shape (n_atoms, 3), where local coordinates 
            have been assembled via X' = X @ rotmats.T + trans
    """

    start_idx = 0 
    final_coords = []
    for i, num_bb in enumerate(bb_num_vec):
        bb_local_coord = local_coords[start_idx:start_idx+num_bb]
        bb_rotmats = rotmats[i]
        bb_trans = trans[i][None]

        bb_coords = bb_local_coord @ bb_rotmats.transpose(-1, -2) + bb_trans
        final_coords.append(bb_coords)

        start_idx += num_bb

    final_coords = np.concatenate(final_coords, axis=0)

    return final_coords

def _get_equiv_vec(cart_coords, atom_types):

    centroid = np.mean(cart_coords, axis=0)

    # Center of mass weighted by atomic number
    weight = atom_types / atom_types.sum()
    weighted_centroid = np.sum(cart_coords * weight[:, None], axis=0)

    # Equivariant vector
    equiv_vec = weighted_centroid - centroid

    # If v = 0 (symmetric), take the closest non-zero atom
    if np.allclose(equiv_vec, 0):
        dist = np.linalg.norm(cart_coords, axis=1)
        sorted_indices = np.argsort(dist)

        i = 0
        while i < len(sorted_indices) and np.allclose(equiv_vec, 0):
            equiv_vec = cart_coords[sorted_indices[i]]
            i += 1
    
    assert not np.allclose(equiv_vec, 0), "Equivariant vector is zero"
    return equiv_vec
    
def _get_pca_axes(data):
    # Center the data
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    if covariance_matrix.ndim == 0:
        return np.zeros(3), np.eye(3)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors    

def _get_equivariant_axes(cart_coords, atom_types):
    """
    Return:
        R: equivariant rotation matrix
    """

    if cart_coords.shape[0] == 1:
        return np.eye(3)

    equiv_vec = _get_equiv_vec(cart_coords, atom_types)    # v(X)

    _, axes = _get_pca_axes(cart_coords)                   # PCA(X)
    ve = equiv_vec @ axes
    flips = ve < 0 
    axes = np.where(flips[None], -axes, axes)

    right_hand = np.stack(
        [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1
    )
    
    return right_hand

def _rotate_bb(bb_coord, bb_atom_type):
    """
    Returns:
        rotmats: numpy array of shape (3, 3)
        local_coord: numpy array of shape (n_bb_atoms, 3) 
    """
    rotmats = _get_equivariant_axes(bb_coord, bb_atom_type) # f(X)
    local_coord = bb_coord @ rotmats                             # g(X) = X f(X)

    return rotmats, local_coord 

def _get_cart_coords_and_types_from_bb(bb):
    real_nodes = ~bb.is_anchor
    cart_coords = frac_to_cart_coords(
        bb.frac_coords, 
        bb.lengths,
        bb.angles,
        bb.num_atoms
    ).numpy().astype(np.float64)
    atom_types = bb.atom_types.numpy()
    cart_coords_real = cart_coords[real_nodes]
    cart_mean = cart_coords_real.mean(axis = 0)
    cart_coords_real = cart_coords_real - cart_mean
    cart_coords_anchor = cart_coords[~real_nodes] - cart_mean
    return cart_coords_real, cart_coords_anchor, atom_types[real_nodes]

def decode_blocks(bb_embs, all_data, kdtree):

    num_bbs = bb_embs.shape[0]
    all_coords, all_anchors, all_types, bb_num_vec, bb_num_vec_anchor = [], [], [], [], []
    all_edges, all_bonds = [], []
    ret_bbs = []
    tot_num_atoms = 0

    for i in range(num_bbs):
        bb_emb = bb_embs[i]
        target = bb_emb.cpu() * bb_emb_std + bb_emb_mean
        ret_bb = all_data[kdtree.query(target)[1]].cpu().clone()
        cart_coords, anchor_coords, atom_types = _get_cart_coords_and_types_from_bb(ret_bb)
        rotmats, local_coords = _rotate_bb(cart_coords, atom_types)
        local_anchors = anchor_coords @ rotmats
        # atom_types = ret_bb.atom_types_real
        # local_coords = ret_bb.local_coords
        # anchor_e_mask, edge_index, bond_types = get_bb_bond_and_types(ret_bb)
        # edge_index = ret_bb.edge_index
        # bond_types = ret_bb.bond_types
        anchor_e_mask, bb_edge_index, bb_bond_types = get_bb_bond_and_types(ret_bb)
        anchor_edges = ret_bb.edge_index[:, anchor_e_mask]
        bb_edge_index = torch.cat([anchor_edges, bb_edge_index], dim=1)
        bb_bond_types = torch.cat([torch.ones(anchor_e_mask.sum()), bb_bond_types])
        all_edges.append((bb_edge_index + tot_num_atoms).long())
        all_bonds.append((bb_bond_types).long())
        all_coords.append(torch.tensor(local_coords))
        all_anchors.append(torch.tensor(local_anchors))
        all_types.append(torch.LongTensor(atom_types))
        bb_num_vec.append(local_coords.shape[0])
        bb_num_vec_anchor.append(local_anchors.shape[0])
        tot_num_atoms += ret_bb.frac_coords.shape[0]
        ret_bbs.append(ret_bb)

    all_coords = torch.cat(all_coords, dim = 0)
    all_anchors = torch.cat(all_anchors, dim = 0)
    all_types = torch.cat(all_types, dim = 0)
    all_edges = torch.cat(all_edges, dim = 1)
    all_bonds = torch.cat(all_bonds, dim = 0)

    return all_coords, all_anchors, all_types, all_edges, all_bonds, bb_num_vec, bb_num_vec_anchor, ret_bbs

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

def match_cps(
    lengths,
    angles,
    cart_coords,
    bb_num_vec_anchor,
    cp_atom_types=None,
    type_mask=True,
):
    cell = lattice_params_to_matrix_torch(
        lengths.view(1, -1), angles.view(1, -1)
    ).squeeze()
    # cart_coords, _ = get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell)
    dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

    cp_components = torch.cat(
        [
            torch.ones(bb_num_vec_anchor[i]) * i
            for i in range(len(bb_num_vec_anchor))
        ],
        dim=0,
    ).long()
    mask = same_component_mask(cp_components)
    dist_mat[mask] = 1000.0

    if type_mask:
        assert cp_atom_types is not None
        # metal, metal mask -- bond cannot be formed between two metal atoms
        # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
        cp_atom_is_metal = (cp_atom_types.view(-1, 1) == metal_atomic_numbers).any(
            dim=-1
        )
        cp_atom_is_not_metal = (
            cp_atom_types.view(-1, 1) == non_metal_atomic_numbers
        ).any(dim=-1)
        metal_mask = torch.logical_and(
            cp_atom_is_metal.view(-1, 1), cp_atom_is_metal.view(1, -1)
        )
        non_metal_mask = torch.logical_and(
            cp_atom_is_not_metal.view(-1, 1), cp_atom_is_not_metal.view(1, -1)
        )
        dist_mat[metal_mask] = 1000.0
        dist_mat[non_metal_mask] = 1000.0

    row, col = linear_sum_assignment(dist_mat.cpu().numpy())
    cost = dist_mat[row, col]
    return {"row": torch.from_numpy(row), "col": torch.from_numpy(col), "cost": cost}

def main(args):

    ret = torch.load(args.res_path, map_location='cpu')
    # {'frac_coords': frac_coords, 'lattices': pred_lattice, 'rotmats': pred_rotmat, 'types': pred_type, 'num_atoms': num_atoms}
    frac_coords = ret['frac_coords']
    lattices = ret['lattices']
    lengths = lattices[:, :3]
    angles = lattices[:, 3:]
    rotmats = ret['rotmats']
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

    for i in tqdm(range(max_process)):
        cur_frac_coords = frac_coords[start_atoms[i]: end_atoms[i]]
        cur_lattices = lattices[i]
        cur_lengths = lengths[i]
        cur_angles = angles[i]
        cur_rotmats = rotmats[start_atoms[i]: end_atoms[i]]
        cur_bb_embs = bb_embs[start_atoms[i]: end_atoms[i]]
        cur_num_atoms = num_atoms[i]
        cur_trans = frac_to_cart_coords(
            cur_frac_coords, 
            cur_lengths.view(1,-1),
            cur_angles.view(1,-1),
            cur_num_atoms
        )
        all_coords, all_anchors, all_types, edge_index, bond_types, bb_num_vec, bb_num_vec_anchor, ret_bbs = decode_blocks(cur_bb_embs, all_data, kdtree)
        feas = feasibility_check(ret_bbs)
        if not feas:
            continue
        final_coords = assemble_coords(all_coords.numpy().astype(np.float64), cur_rotmats.numpy().astype(np.float64), cur_trans.numpy().astype(np.float64), bb_num_vec)
        final_anchors = assemble_coords(all_anchors.numpy().astype(np.float64), cur_rotmats.numpy().astype(np.float64), cur_trans.numpy().astype(np.float64), bb_num_vec_anchor)
        final_frac_coords = cart_to_frac_coords(
            torch.tensor(final_coords).float(),
            cur_lengths.view(1,-1),
            cur_angles.view(1,-1),
            sum(bb_num_vec)
        )
        cell = lattice_params_to_matrix_torch(cur_lengths.view(1,-1), cur_angles.view(1,-1)).squeeze()
        connecting_atom_index = get_connecting_atom_index(ret_bbs)
        atom_types = torch.cat([x.atom_types for x in ret_bbs])
        connecting_atom_types = atom_types[connecting_atom_index]

        is_anchors = torch.cat([bb.is_anchor for bb in ret_bbs])
        anchor_index = is_anchors.nonzero().T
        anchor_s_mask = (edge_index[0].view(-1, 1) == anchor_index).any(dim=1)
        anchor_neighs = torch.unique(edge_index[:, anchor_s_mask], dim=1)[1]

        cp_match = match_cps(
            cur_lengths,
            cur_angles,
            torch.tensor(final_anchors).float(),
            bb_num_vec_anchor,
            connecting_atom_types
        )
        row, col = cp_match["row"], cp_match["col"]
        inter_BB_edges = torch.cat(
            [
                torch.stack([anchor_neighs[row], anchor_neighs[col]]),
                torch.stack([anchor_neighs[col], anchor_neighs[row]]),
            ],
            dim=1,
        )
        anchor_t_mask = (edge_index[1].view(-1, 1) == anchor_index).any(dim=1)
        anchor_e_mask = torch.logical_or(anchor_s_mask, anchor_t_mask)
        edge_index = edge_index[:, ~anchor_e_mask]
        bond_types = bond_types[~anchor_e_mask]

        # pdb.set_trace()

        edge_index = torch.cat([edge_index, inter_BB_edges], dim=1)
        bond_types = torch.cat(
            [bond_types, torch.ones(inter_BB_edges.shape[1])], dim=0
        ).long()
        atom_index = torch.cat([~bb.is_anchor for bb in ret_bbs], dim = 0).nonzero().flatten()
        remapping = atom_index, torch.arange(sum(bb_num_vec))
        edge_index = remap_values(remapping, edge_index)
        edge_index, unique_index = get_unique_and_index(edge_index, dim=1)
        bond_types = bond_types[unique_index]


        to_jimages = compute_image_flag(
            cell, final_frac_coords[edge_index[0]], final_frac_coords[edge_index[1]]
        )

        res = Data(
            frac_coords=final_frac_coords,
            atom_types=all_types,
            num_atoms=sum(bb_num_vec),
            cell=cell,
            lengths=cur_lengths,
            angles=cur_angles,
            edge_index=edge_index,
            to_jimages=to_jimages,
            bond_types=bond_types,
        )

        # pdb.set_trace()


        # pred_structure = Structure(
        #     lattice=Lattice.from_parameters(*cur_lattices.tolist()),
        #     species=all_types,
        #     coords=final_coords,
        #     coords_are_cartesian=True
        # )
        # Write predicted structure
        # writer = CifWriter(pred_structure)
        sample_dir = os.path.join(os.path.dirname(args.res_path), 'bond_samples/cif')
        os.makedirs(sample_dir, exist_ok=True)
        # writer.write_file(os.path.join(sample_dir, f'pred_{i}.cif'))
        mof2cif_with_bonds(res, os.path.join(sample_dir, f'pred_{i}.cif'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', required=True)
    parser.add_argument('--bb_z_path', required=True)
    parser.add_argument('--bb_blocks_path', required=True)
    parser.add_argument('--max_process', default=-1, type=int)
    args = parser.parse_args()
    main(args)

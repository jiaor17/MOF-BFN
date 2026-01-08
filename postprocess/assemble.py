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
    return cart_coords[real_nodes], atom_types[real_nodes]

def decode_blocks(bb_embs, all_data, kdtree):

    num_bbs = bb_embs.shape[0]
    all_coords, all_types, bb_num_vec = [], [], []

    for i in range(num_bbs):
        bb_emb = bb_embs[i]
        target = bb_emb.cpu() * bb_emb_std + bb_emb_mean
        ret_bb = all_data[kdtree.query(target)[1]].cpu().clone()
        # cart_coords, atom_types = _get_cart_coords_and_types_from_bb(ret_bb)
        # rotmats, local_coords = _rotate_bb(cart_coords, atom_types)
        atom_types = ret_bb.atom_types_real
        local_coords = ret_bb.local_coords
        all_coords.append(local_coords)
        all_types.append(atom_types)
        bb_num_vec.append(local_coords.shape[0])

    all_coords = np.concatenate(all_coords, axis = 0)
    all_types = np.concatenate(all_types, axis = 0)

    return all_coords, all_types, bb_num_vec

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
        all_coords, all_types, bb_num_vec = decode_blocks(cur_bb_embs, all_data, kdtree)
        final_coords = assemble_coords(all_coords, cur_rotmats.numpy().astype(np.float64), cur_trans.numpy().astype(np.float64), bb_num_vec)
        pred_structure = Structure(
            lattice=Lattice.from_parameters(*cur_lattices.tolist()),
            species=all_types,
            coords=final_coords,
            coords_are_cartesian=True
        )
        # Write predicted structure
        writer = CifWriter(pred_structure)
        sample_dir = os.path.join(os.path.dirname(args.res_path), 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        writer.write_file(os.path.join(sample_dir, f'pred_{i}.cif'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', required=True)
    parser.add_argument('--bb_z_path', required=True)
    parser.add_argument('--bb_blocks_path', required=True)
    parser.add_argument('--max_process', default=-1, type=int)
    args = parser.parse_args()
    main(args)

from functools import cache
import os
import pickle
import gzip
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from pymatgen.core.lattice import Lattice
from mofdiff.common.data_utils import frac_to_cart_coords, lattice_params_to_matrix_torch
import lmdb
from data.utils import frac2cart, compute_distance_matrix

class MOFDataset(Dataset):
    def __init__(
            self,
            *,
            cache_path,
            dataset_cfg,
            is_training,
            split_idx=None,
            do_preprocess=True,
            n_samples=-1,
            n_samples_from_split=-1,
            use_block_emb=False,
            bb_encoder=None,
            bb_emb_clipping=100,
            device='cpu',
            max_atoms=200,
            max_cps=20,
            max_bbs=20,
        ):

        self._log = logging.getLogger(__name__)
        self._cache_path = cache_path
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._do_preprocess = do_preprocess
        self._use_block_emb = use_block_emb
        if bb_encoder is not None:
            self.bb_encoder = bb_encoder.to(device)
            self.bb_encoder.type_mapper.match_device(self.bb_encoder)
        else:
            self.bb_encoder = None
        self.device = device
        self.bb_emb_clipping = bb_emb_clipping
        self.max_atoms = max_atoms
        self.max_cps = max_cps
        self.max_bbs = max_bbs
        self.processed_env = None


        # Check if processed data exists
        if self._do_preprocess:
            processed_path = self._cache_path.replace('.lmdb', f'{"_bb" if self._use_block_emb else ""}_preprocessed.lmdb')
            if os.path.exists(processed_path):
                print(f"INFO:: Loading processed data from {processed_path}")
                # with gzip.open(processed_path, 'rb') as f:
                self.processed_env = lmdb.open(processed_path, subdir=False, readonly=True, lock=False)
                self.lmdb_path = processed_path
            else:
                print(f"INFO:: No processed data found at {processed_path}")
                self._preprocess()
        else:
            self.processed_env = lmdb.open(self._cache_path, subdir=False, readonly=True, lock=False)


        with self.processed_env.begin() as txn:
            self._original_length = pickle.loads(txn.get('length'.encode()))


        self._n_samples = n_samples
        self._n_samples_from_split = n_samples_from_split
        self._split_idx = split_idx
        if split_idx is not None:
            with open(split_idx, 'rb') as f:
                self._split_idx = pickle.load(f)



    @cache
    def __len__(self):
        
        if self._n_samples != -1:
            return self._n_samples
        if self._split_idx is not None:
            if self._n_samples_from_split != -1:
                return self._n_samples_from_split
            return len(self._split_idx)
        return self._original_length

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    @property
    def data_conf(self):
        return self._data_conf
        
    @staticmethod
    def _recenter_mof(data, c_rotmat):
        """
        Returns:
            gt_coords: numpy array of shape (n_atoms, 3)
                Ground truth coordinates with PBC considered
            gt_trans: numpy array of shape (n_bbs, 3)
                Ground truth translation vectors
            local_coords: list of numpy array
                CoM-free building block coordinates with gt rotations
        """
        def _get_cart_coords_from_bb(bb):
            cart_coords = frac_to_cart_coords(
                bb.frac_coords, 
                bb.lengths,
                bb.angles,
                bb.num_atoms
            ).numpy().astype(np.float64)
            return cart_coords

        # Compute translation and centered coordinates
        gt_trans = []
        local_coords = []
        for bb in data.pyg_mols:
            bb_coords = _get_cart_coords_from_bb(bb)
            bb_coords = bb_coords @ c_rotmat.T
            gt_trans.append(bb_coords.mean(axis=0))
            local_coords.append(bb_coords - bb_coords.mean(axis=0))
        
        gt_trans = np.array(gt_trans)
        gt_trans = gt_trans - gt_trans.mean(axis=0)
        
        # Compute global coordinates
        gt_coords = np.concatenate(
            [coords + trans for coords, trans in zip(local_coords, gt_trans)], 
            axis=0
        )
        return gt_coords, gt_trans, local_coords

    @staticmethod
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
    
    @staticmethod
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

    def _get_equivariant_axes(self, cart_coords, atom_types):
        """
        Return:
            R: equivariant rotation matrix
        """

        if cart_coords.shape[0] == 1:
            return np.eye(3)

        equiv_vec = self._get_equiv_vec(cart_coords, atom_types)    # v(X)

        _, axes = self._get_pca_axes(cart_coords)                   # PCA(X)
        ve = equiv_vec @ axes
        flips = ve < 0 
        axes = np.where(flips[None], -axes, axes)

        right_hand = np.stack(
            [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1
        )
        
        return right_hand
    
    def _rotate_bb(self, bb_coord, bb_atom_type):
        """
        Returns:
            rotmats: numpy array of shape (3, 3)
            local_coord: numpy array of shape (n_bb_atoms, 3) 
        """
        rotmats = self._get_equivariant_axes(bb_coord, bb_atom_type) # f(X)
        local_coord = bb_coord @ rotmats                             # g(X) = X f(X)

        return rotmats, local_coord 

    def _get_canonical_rotmat(self, lattice):
        """
        Returns:
            rotmat: numpy array of shape (3, 3)
                Rotation matrix to apply to Cartesian coordinates
            lattice_std: pymatgen.core.lattice.Lattice
                Standardized lattice
        """
        lattice_std = Lattice.from_parameters(*lattice.parameters)
        rotmat = np.linalg.inv(lattice_std.matrix) @ lattice.matrix
        
        return rotmat, lattice_std

    def embed_bb(self, all_bbs):
        """
        get all building block embeddings.
        """
        def bb_criterion(bb):
            bb.num_cps = bb.is_anchor.long().sum()
            if (bb.num_atoms > self.max_atoms) or (bb.num_cps > self.max_cps):
                return None, False

            cart_coords = frac_to_cart_coords(
                bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
            )
            pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)

            # detect BBs with problematic bond info.
            edge_index = bb.edge_index
            j, i = edge_index
            bond_dist = (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()

            success = (
                pdist.min() > 0.25
                and bond_dist.max() < 5.0
                and (bb.num_atoms <= self.max_atoms)
                and (bb.num_cps <= self.max_cps)
            )

            return cart_coords, success

        bb_all_success = True

        for bb in all_bbs:
            cart_coords, success = bb_criterion(bb)
            if success:
                bb.num_nodes = bb.num_atoms
                bb.diameter = torch.pdist(cart_coords).max()
            else:
                bb_all_success = False
                break
        
        if not bb_all_success:
            raise Exception("Invalid Building Blocks")
            # return None

        batch = Batch.from_data_list(
            all_bbs
        ).to(self.device)
        batch.atom_types = self.bb_encoder.type_mapper.transform(batch.atom_types)
        with torch.no_grad():
            bb_emb = self.bb_encoder.encode(batch).cpu()
        all_bb_emb = torch.clamp(
            bb_emb, -self.bb_emb_clipping, self.bb_emb_clipping
        )

        return bb_emb
        
    def mof_criterion(self, mof):
        if mof.num_components > self.max_bbs:
            return False

        cell = lattice_params_to_matrix_torch(mof.lengths, mof.angles).squeeze()
        distances = compute_distance_matrix(
            cell, frac2cart(mof.cg_frac_coords, cell)
        ).fill_diagonal_(5.0)
        return (not (distances < 1.0).any()) and mof.num_components <= self.max_bbs

    def _process_one(self, datapoint):

        if not self.mof_criterion(datapoint):
            raise Exception("Invalid Building Blocks")

        # Get Niggli cell
        lattice = Lattice(datapoint.cell.squeeze())
        lattice = lattice.get_niggli_reduced_lattice()
        
        # Get canonical rotation matrix
        c_rotmat, lattice = self._get_canonical_rotmat(lattice)
        
        # Recenter MOF
        gt_coords, gt_trans, centered_bb_coords = self._recenter_mof(datapoint, c_rotmat)

        # Rotate MOF bbs
        gt_rotmats = []
        local_coords = []
        
        for i, bb_coord in enumerate(centered_bb_coords):
            bb_atom_type = datapoint.pyg_mols[i].atom_types.numpy()
            rotmats, local_coord = self._rotate_bb(bb_coord, bb_atom_type)
            gt_rotmats.append(rotmats)
            local_coords.append(local_coord)

        gt_rotmats = np.stack(gt_rotmats, axis=0)
        local_coords = np.concatenate(local_coords, axis=0)

        # Create masks
        bb_num_vec = np.array([bb.num_atoms for bb in datapoint.pyg_mols])
        res_mask = np.ones_like(bb_num_vec)
        diffuse_mask = np.ones_like(bb_num_vec)

        

        # Convert numpy arrays to torch tensors
        feats = {
            'rotmats_1': torch.tensor(gt_rotmats).float(),
            'trans_1': torch.tensor(gt_trans).float(),
            'res_mask': torch.tensor(res_mask).int(),
            'diffuse_mask': torch.tensor(diffuse_mask).int(),
            'local_coords': torch.tensor(local_coords).float(),
            'gt_coords': torch.tensor(gt_coords).float(),
            'bb_num_vec': torch.tensor(bb_num_vec).int(),
            'atom_types': torch.cat([bb.atom_types for bb in datapoint.pyg_mols], dim=0).int(),
            'lattice_1': torch.tensor(lattice.parameters).float(),
            'cell_1': torch.tensor(lattice.matrix).float()
        }


        # Create bb Embeddings
        if self._use_block_emb:
            bb_emb = self.embed_bb(datapoint.bbs)
            if bb_emb is not None:
                feats.update({
                    'bb_emb_1': bb_emb.float(),
                })
            else:
                raise Exception("Invalid Building Blocks")

        return feats

    def _preprocess(self):
        # Load cached data
        print(f"INFO:: Loading cached data from {self._cache_path}")
        cached_env = lmdb.open(self._cache_path, subdir=False, readonly=True, lock=False)
        tot = pickle.loads(cached_env.begin().get('length'.encode()))
        processed_path = self._cache_path.replace('.lmdb', f'{"_bb" if self._use_block_emb else ""}_preprocessed.lmdb')
        processed_env = lmdb.open(processed_path, subdir=False, readonly=False, lock=False, map_size=1099511627776)

        # Process data
        print(f"INFO:: Processing data")
        results = []
        errs = []
        for idx in tqdm(range(tot)):
            try:
                datapoint = pickle.loads(cached_env.begin().get(f'{idx}'.encode()))
                post_datapoint = self._process_one(datapoint)
            except:
                print(f"WARNING::Datapoint {idx} cannot be processed")
                errs.append(idx)
                post_datapoint = None
            if post_datapoint is not None:
                txn = processed_env.begin(write=True)
                post_idx = idx - len(errs)
                txn.put(f'{post_idx}'.encode(), pickle.dumps(post_datapoint, protocol=-1))
                txn.commit()              

        txn = processed_env.begin(write=True)
        post_tot = tot - len(errs)
        txn.put('length'.encode(), pickle.dumps(post_tot, protocol=-1))
        txn.commit()        
        txn = processed_env.begin(write=True)
        txn.put('errs'.encode(), pickle.dumps(errs, protocol=-1))
        txn.commit()
        processed_env.sync()
        processed_env.close()

        # Save processed data

        self.processed_env = lmdb.open(processed_path, subdir=False, readonly=True, lock=False)
    
    def __getitem__(self, idx):
        """
        Returns: dictionary with following keys:
            - rotmats_1: [M, 3, 3]
            - trans_1: [M, 3]
            - res_mask: [M,]
            - diffuse_mask: [M,]
            - local_coords: [N, 3]
            - gt_coords: [N, 3]
            - bb_num_vec: [M,]
            - bb_emb: [M, 3]
            - atom_types: [N,]
            - lattice: [6,]
            - cell: [3, 3]
        """

        if self._split_idx is not None:
            idx = self._split_idx[idx]

        datapoint = pickle.loads(self.processed_env.begin().get(f'{idx}'.encode()))

        if not self._do_preprocess:
            datapoint = self._process_one(datapoint)


        return datapoint

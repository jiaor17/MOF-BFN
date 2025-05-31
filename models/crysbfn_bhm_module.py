import math, copy

from data.so3_utils import matrix_to_quaternion, quaternion_to_matrix, rotquat_to_rotmat
import numpy as np
from p_tqdm import p_map
import pymatgen as pmg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import wandb

from crysbfn.common.utils import PROJECT_ROOT
from crysbfn.pl_modules.base_model import BaseModule
from crysbfn.common.data_utils import lattices_to_params_shape
from crysbfn.common.data_utils import (_make_global_adjacency_matrix, cart_to_frac_coords, frac_to_cart_coords, remove_mean)
from crysbfn.common.data_utils import SinusoidalTimeEmbeddings, lattices_to_params_shape
from crysbfn.common.data_utils import PeriodHelper as p_helper
from torch_geometric.data import InMemoryDataset, Data, DataLoader, Batch
from models.crysbfn_bhm import CrysBFN
MAX_ATOMIC_NUM=100

import torch.distributed as dist

import pandas as pd
import json
import time, os
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher


class CrysBFN_PL_Model(BaseModule):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.cfg = cfg.model

        self._exp_cfg = cfg.experiment
        self._data_cfg = cfg.data
        self.BFN:CrysBFN = hydra.utils.instantiate(self.cfg.BFN, hparams=self.cfg, device=device,
                                           _recursive_=False)
        self.time_dim = self.cfg.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)


        # atom type map
        self.T_min = eval(str(self.cfg.T_min))
        self.T_max = eval(str(self.cfg.T_max))
        
        self.train_loader = None
        self._inference_dir = None

        self.results = []
        self.matcher = StructureMatcher(
            stol=cfg.matcher.stol, 
            angle_tol=cfg.matcher.angle_tol, 
            ltol=cfg.matcher.ltol
        )

        self.inference_fg = self.cfg.inference_fg if hasattr(self.cfg, 'inference_fg') else False

        self.ignore_so3_weight = self.cfg.ignore_so3_weight if hasattr(self.cfg, 'ignore_so3_weight') else False

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir
        
    def forward(self, batch):
        batch = batch.to(self.device)

        lengths, angles = batch.lattice_1[:, :3], batch.lattice_1[:, 3:]

        frac_coords = cart_to_frac_coords(
            batch.trans_1,
            lengths,
            angles,
            batch.num_atoms
        )

        rot_vecs = matrix_to_quaternion(batch.rotmats_1)

        bb_embs = batch.bb_emb_1

        lattice_loss, coord_loss, rot_loss, type_loss = self.BFN.loss_one_step(
            t = None,
            bb_embs=bb_embs,
            frac_coords=frac_coords,
            rot_vecs = rot_vecs,      
            lengths=lengths,
            angles = angles,
            num_atoms = batch.num_atoms,
            segment_ids= batch.batch,
            ignore_so3_weight=self.ignore_so3_weight
        )
        
        loss = (
            self.cfg.cost_lattice * lattice_loss +
            self.cfg.cost_coord * coord_loss +
            self.cfg.cost_rot * rot_loss + 
            self.cfg.cost_type * type_loss
            )

        return {
            'loss' : loss,
            'loss_lattice' : lattice_loss,
            'loss_coord' : coord_loss,
            'loss_rot': rot_loss,
            'loss_type' : type_loss,
            'batch_size' : batch.num_graphs
        }


    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(
            params=self.parameters(),
            **self._exp_cfg.optimizer
        )
        if not self._exp_cfg.use_lr_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self._exp_cfg.lr_scheduler
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    

    def lattice_matrix_to_params(self, lattice_matrix):
        pmg_lattice =  pmg.Lattice(lattice_matrix).get_niggli_reduced_lattice()
        lengths = pmg_lattice.lengths
        angles = pmg_lattice.angles
        return lengths, angles

    @torch.no_grad()
    def sample(self, batch, sample_steps=None, segment_ids=None, show_bar=False, samp_acc_factor=1.0, **kwargs):
        sample_batch = batch
        atom_types = sample_batch.atom_types.to(self.device)
        local_coords = sample_batch.local_coords.to(self.device)
        bb_num_vec = sample_batch.bb_num_vec.to(self.device)
        segment_ids = sample_batch.batch.to(self.device)
        num_atoms = sample_batch.num_atoms.to(self.device)
        sample_steps = self.cfg.BFN.dtime_loss_steps if sample_steps is None else sample_steps
        get_rej = False if 'return_traj' not in kwargs else kwargs['return_traj']

        rot_vecs = matrix_to_quaternion(batch.rotmats_1)
        kwargs['rotquat'] = rot_vecs

        sample_res = self.BFN.sample(
                atom_types,
                num_atoms,
                local_coords,
                bb_num_vec,
                sample_steps=sample_steps,
                segment_ids=segment_ids,
                show_bar=show_bar,
                samp_acc_factor=samp_acc_factor,
                batch = batch,
                strategy= 'end_back',
                **kwargs
            )
        if get_rej:
            coord_pred_final, lattice_pred_final, rot_pred_final, traj = sample_res
        else:
            coord_pred_final, lattice_pred_final, rot_pred_final = sample_res

        
        frac_coords = p_helper.any2frac(coord_pred_final,eval(str(self.T_min)),eval(str(self.T_max)))

        
        pred_lattice_pre = lattice_pred_final

        pred_log_lengths, pred_tan_angles = pred_lattice_pre[..., :3], pred_lattice_pre[..., 3:]

        pred_lengths = torch.exp(pred_log_lengths)
        pred_angles = torch.rad2deg(torch.atan(pred_tan_angles) + math.pi / 2)


        pred_lattice = torch.cat([pred_lengths, pred_angles], dim=-1)

        pred_rotmat = quaternion_to_matrix(rot_pred_final)
        
        output_dict = {'frac_coords': frac_coords, 'lattices': pred_lattice, 'rotmats': pred_rotmat}
        return output_dict

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss, batch_size = self.compute_stats(output_dict, prefix='train')

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size = batch_size
        )

        if loss.isnan():
            hydra.utils.log.info(f'loss is nan at step {self.global_step}')
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss, batch_size = self.compute_stats(output_dict, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size = batch_size
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss, batch_size = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_rot = output_dict['loss_rot']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']
        batch_size = output_dict['batch_size']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_rot_loss': loss_rot,
            f'{prefix}_type_loss': loss_type,
        }

        return log_dict, loss, batch_size


    def to_datalist(self, lattice, coords, atom_types, bb_num_vec, node2graph, num_graphs):

        num_atoms = scatter(bb_num_vec, node2graph, reduce='sum', dim_size=num_graphs)
        ret = []
        end_idx = torch.cumsum(num_atoms, dim=0)
        start_idx = end_idx - num_atoms
        for i in range(num_graphs):
            ret.append((
                lattice[i].detach().cpu().numpy(),
                coords[start_idx[i]:end_idx[i]].detach().cpu().numpy(),
                atom_types[start_idx[i]:end_idx[i]].detach().cpu().numpy(),
            ))
        return ret

    def to_datalist_fg(self, lattice, coords_fg, num_blocks, num_graphs):

        ret = []
        end_idx = torch.cumsum(num_blocks, dim=0)
        start_idx = end_idx - num_blocks
        for i in range(num_graphs):
            ret.append((
                lattice[i].detach().cpu().numpy(),
                coords_fg[start_idx[i]:end_idx[i]].detach().cpu().numpy(),
                np.arange(int(num_blocks[i].detach().cpu())) + 1,
            ))
        return ret

    def _assemble_coords(self, local_coords, rotmats, trans, bb_num_vec):
        """
        Returns:
            coords: numpy array of shape (n_atoms, 3), where local coordinates 
                have been assembled via X' = X @ rotmats.T + trans
        """

        start_idx = 0 
        final_coords = []
        device = self._device
        for i, num_bb in enumerate(bb_num_vec):
            bb_local_coord = local_coords[start_idx:start_idx+num_bb].to(device)
            bb_rotmats = rotmats[i].to(device)
            bb_trans = trans[i][None].to(device)

            bb_coords = bb_local_coord @ bb_rotmats.t() + bb_trans
            final_coords.append(bb_coords)

            start_idx += num_bb

        final_coords = torch.cat(final_coords, dim=0)

        return final_coords



    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'

        now = time.time()

        ret = self.sample(batch, show_bar=True)

        pred_lattice = ret['lattices']

        pred_lengths, pred_angles = pred_lattice[:, :3], pred_lattice[:, 3:]


        pred_frac_block = ret['frac_coords']
        pred_cart_block = frac_to_cart_coords(
            pred_frac_block,
            pred_lengths,
            pred_angles,
            batch.num_atoms
        )

        pred_rotmat = ret['rotmats']

        pred_coords = self._assemble_coords(batch.local_coords, pred_rotmat, pred_cart_block, batch.bb_num_vec)

        elapsed = time.time() - now

        gt_ret_block = self.to_datalist_fg(batch.lattice_1, batch.trans_1, batch.num_atoms, batch.num_graphs)
        pred_ret_block = self.to_datalist_fg(pred_lattice, pred_cart_block, batch.num_atoms, batch.num_graphs)
        res_block = self.get_res(gt_ret_block, pred_ret_block, batch.num_graphs, batch_idx, 'block')

        gt_ret_atom = self.to_datalist(batch.lattice_1, batch.gt_coords, batch.atom_types, batch.bb_num_vec, batch.batch, batch.num_graphs)
        pred_ret_atom = self.to_datalist(pred_lattice, pred_coords, batch.atom_types, batch.bb_num_vec, batch.batch, batch.num_graphs)
        res_atom = self.get_res(gt_ret_atom, pred_ret_atom, batch.num_graphs, batch_idx, 'atom')

        for k in res_block:
            # Save results
            self.results.append({
                'sample_idx': k,
                'rms_dist_block': res_block[k],
                'rms_dist_atom': res_atom[k],
                'time': elapsed / batch.num_graphs
            })



    def get_res(self, gt_ret, pred_ret, num_graphs, batch_idx, suffix):

        ans = {}

        for i in range(num_graphs):

            # Get global index
            global_idx = (batch_idx * dist.get_world_size() + dist.get_rank()) * 300 + i
            
            # Set directory
            sample_dir = os.path.join(
                self.inference_dir,
                f'sample_{global_idx}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            gt_lat, gt_c, gt_atom_types = gt_ret[i]

            # Create structure files
            gt_structure = Structure(
                lattice=Lattice.from_parameters(*gt_lat),
                species=gt_atom_types,
                coords=gt_c,
                coords_are_cartesian=True
            )
            # Write ground truth structure
            writer = CifWriter(gt_structure)
            writer.write_file(os.path.join(sample_dir, f'gt_{global_idx}_{suffix}.cif'))

            pred_lat, pred_c, pred_atom_types = pred_ret[i]
            pred_structure = Structure(
                lattice=Lattice.from_parameters(*pred_lat),
                species=pred_atom_types,
                coords=pred_c,
                coords_are_cartesian=True
            )
            # Write predicted structure
            writer = CifWriter(pred_structure)
            writer.write_file(os.path.join(sample_dir, f'pred_{global_idx}_{suffix}.cif'))

            
            # Compute RMSD with structure matcher
            rms_dist = self.matcher.get_rms_dist(gt_structure, pred_structure)
            rms_dist = None if rms_dist is None else rms_dist[0]

            ans[global_idx] = rms_dist

        return ans

        
    def on_predict_epoch_end(self):
        # Gather results
        all_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_results, self.results)

        if dist.get_rank() == 0:
            all_results = [item for sublist in all_results for item in sublist]
            all_df = pd.DataFrame(all_results)
            all_df = all_df.sort_values(by='sample_idx')
        
            # Compute average metrics
            results = {}
            for suf in ['block', 'atom']:
                rms_dist = all_df[f'rms_dist_{suf}'].dropna()
                match_rate = len(rms_dist) / len(all_df) * 100
                results.update({
                    f'match_rate_{suf}': match_rate,
                    f'rms_dist_{suf}': rms_dist.mean() if len(rms_dist) > 0 else None,
                    'avg_time': all_df['time'].dropna().mean()
                })

            # Save average metrics to JSON
            print(f"INFO:: {results}")
            with open(os.path.join(self.inference_dir, 'average.json'), 'w') as f:
                json.dump(results, f)

            # Save results to CSV
            all_df.to_csv(os.path.join(self.inference_dir, 'results.csv'), index=False)

import math, copy

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
from crysbfn.pl_modules.base_model import build_mlp, BaseModule
from crysbfn.common.data_utils import lattices_to_params_shape
from crysbfn.common.data_utils import (_make_global_adjacency_matrix, cart_to_frac_coords, remove_mean)
from crysbfn.common.data_utils import SinusoidalTimeEmbeddings, lattices_to_params_shape
from crysbfn.common.data_utils import PeriodHelper as p_helper
from torch_geometric.data import InMemoryDataset, Data, DataLoader, Batch
from crysbfn.pl_modules.crysbfn import CrysBFN
MAX_ATOMIC_NUM=100

class CrysBFN_PL_Model(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.BFN:CrysBFN = hydra.utils.instantiate(self.hparams.BFN, hparams=self.hparams,
                                           _recursive_=False)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None
        self.cart_scaler = None

        # atom type map
        self.atom_type_map = self.hparams.data.atom_type_map
        self.inv_atom_type_map = {v: k for k, v in self.atom_type_map.items()}
        self.num_atom_types = len(self.atom_type_map.keys())
        self.K = self.num_atom_types
        self.atomic_nb = [e[1] for e in sorted(self.inv_atom_type_map.items(),key=lambda x:x[0])]
        self.T_min = eval(str(self.hparams.T_min))
        self.T_max = eval(str(self.hparams.T_max))
        
        self.train_loader = None
        
    def forward(self, batch):
        batch = batch.to(self.device)
        # 按照self.atom_type_map把atom_type重新映射
        mapped_atom_types = torch.tensor([self.atom_type_map[atom_type.item()] 
                                            for atom_type in batch.atom_types], 
                                                device=self.device)
        one_hot_x = F.one_hot(mapped_atom_types, num_classes=self.num_atom_types)\
                                                    .float().to(self.device)
        charges = batch.atom_types.unsqueeze(-1).float().to(self.device)
        type_loss, lattice_loss, coord_loss = self.BFN.loss_one_step(
            t = None,
            atom_type = one_hot_x,
            frac_coords = batch.frac_coords,
            lengths = batch.lengths,
            angles = batch.angles,
            num_atoms = batch.num_atoms,
            segment_ids= batch.batch,
            edge_index = batch.fully_connected_edge_index,
        )
        length_loss = 0
        angle_loss = 0
        
        loss = (
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss
            )

        return {
            'loss' : loss,
            'loss_lattice' : lattice_loss,
            'loss_coord' : coord_loss,
            'loss_type' : type_loss,
            'loss_length' : length_loss,
            'loss_angle' : angle_loss,
        }
    
    def initiate_sample_batch(self, batch_size):
        """
        Initiate a batch for sample, which will generate data from prior distribution with n_node_histogram
        """
        num_atoms_dist = self.hparams.data.num_atoms_dist
        # 从 num_atoms_dist中采样batch_size个num_atom
        num_atoms_freq = np.array(list(num_atoms_dist.values()))
        num_atoms_freq = num_atoms_freq / num_atoms_freq.sum()
        num_atoms_choices = np.array(list(num_atoms_dist.keys()))
        n_atoms = torch.tensor(np.random.choice(num_atoms_choices, 
                                size=batch_size, p=num_atoms_freq), 
                                device=self.device)
        full_adj, diag_bool = _make_global_adjacency_matrix(max(num_atoms_dist.values())+1)
        
        make_adjacency_matrix = lambda x: full_adj[:, :x, :x].reshape(2, -1)[
            :, ~(diag_bool[:x, :x].reshape(-1))
        ]

        def _evaluate_transform(params):
            data, num_atoms = params
            # sample n_nodes from n_node_histogram
            data.edge_index = make_adjacency_matrix(num_atoms)
            data.num_atoms = num_atoms
            data.num_nodes = num_atoms
            return data
    
        assert batch_size == n_atoms.shape[0], 'batch_size should be equal to num_molecules'
        data_list = [(Data(), num_atom) for _, num_atom in zip(range(batch_size), n_atoms.cpu())]
        data_list = list(map(_evaluate_transform, data_list))
        return Batch.from_data_list(data_list)

    def charge_decode(self, charge):
        """
        charge: [n_nodes, 1]
        """
        anchor = torch.tensor(
            [
                (2 * k - 1) / max(self.atomic_nb) - 1
                for k in self.atomic_nb[False:]
            ], # TODO: remove_h debug here
            dtype=torch.float32,
            device=charge.device,
        )
        atom_type = (charge - anchor).abs().argmin(dim=-1)
        one_hot = torch.zeros(
            [charge.shape[0], self.num_atom_types], dtype=torch.float32
        )
        one_hot[torch.arange(charge.shape[0]), atom_type] = 1
        atom_types = torch.argmax(one_hot, dim=-1)
        return atom_types


    def lattice_matrix_to_params(self, lattice_matrix):
        pmg_lattice =  pmg.Lattice(lattice_matrix).get_niggli_reduced_lattice()
        lengths = pmg_lattice.lengths
        angles = pmg_lattice.angles
        return lengths, angles

    @torch.no_grad()
    def sample(self,  num_samples=256, sample_steps=None, segment_ids=None, show_bar=False, samp_acc_factor=1.0, **kwargs):
        sample_loader = self.train_loader
        print('use train_loader', sample_loader is not None)
        sampling_batch = next(iter(sample_loader)).to(self.device) if sample_loader is not None else None      
        
        batch_size = self.hparams.data.datamodule.batch_size.val if num_samples is None else num_samples
        sample_batch = self.initiate_sample_batch(batch_size=batch_size)
        edge_index = sample_batch.edge_index.to(self.device)
        segment_ids = sample_batch.batch.to(self.device)
        num_atoms = sample_batch.num_atoms.to(self.device)
        # sample_steps = sample_steps
        # if self.hparams.BFN.dtime_loss:
        #     sample_steps = self.hparams.BFN.dtime_loss_steps
        # else:
        #     assert sample_steps is not None
        sample_steps = self.hparams.BFN.dtime_loss_steps if sample_steps is None else sample_steps
        get_traj = False if 'return_traj' not in kwargs else kwargs['return_traj']
        sample_res = self.BFN.sample(
                num_atoms,
                edge_index,
                sample_steps=sample_steps,
                edge_attr=None,
                segment_ids=segment_ids,
                show_bar=show_bar,
                # return_traj=get_traj,
                samp_acc_factor=samp_acc_factor,
                batch = sampling_batch,
                **kwargs
            )
        if get_traj:
            type_final, coord_pred_final, lattice_pred_final, traj = sample_res
        else:
            type_final, coord_pred_final, lattice_pred_final = sample_res
        
        # 把 [-pi, pi) 转为 [0, 1)
        
        frac_coords = p_helper.any2frac(coord_pred_final,eval(str(self.T_min)),eval(str(self.T_max)))
        
        assert frac_coords.min() >= 0 and frac_coords.max() < 1
        # 把one-hot的type_final转为索引
        if 'charge_loss' in self.hparams.BFN and self.hparams.BFN.charge_loss:
            type_final = self.charge_decode(type_final[:, :1])
        else:
            type_final = type_final.argmax(dim=-1)
        
        inverse_map = {v: k for k, v in self.atom_type_map.items()}
        # 把type_final转为原来的atom_type
        
        atom_types = torch.tensor([inverse_map[type.item()] for type in type_final], device=self.device)
        # 处理lattices
            
        lattices = lattice_pred_final.reshape(-1, 3, 3).cpu()
        lattices = lattices * self.hparams.data.lattice_std + self.hparams.data.lattice_mean
        
        lengths, angles = lattices_to_params_shape(torch.tensor(lattices))
        # lengths_angles = list(map(self.lattice_matrix_to_params, lattices))
        # lengths, angles = zip(*lengths_angles)
        
        lengths = torch.tensor(lengths, device=self.device)
        angles = torch.tensor(angles, device=self.device)
        
        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': frac_coords, 'atom_types': atom_types, 'traj': traj if get_traj else None,
                       'is_traj': False, 'segment_ids': segment_ids}
        return output_dict

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)        
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'type_loss': loss_type,
            'coord_loss': loss_coord,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            hydra.utils.log.info(f'loss is nan at step {self.global_step}')
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
            # f'{prefix}_length_loss': loss_length,
            # f'{prefix}_angle_loss': loss_angle,
        }

        return log_dict, loss

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    device = 'cuda'
    batch = next(iter(datamodule.train_dataloader())).to(device)
    model: CrysBFN_PL_Model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    ).to(device)
    model.train_loader = datamodule.train_dataloader()
    model.BFN.device = device
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    model.cart_scaler = datamodule.cart_scaler.copy()
    # loss = model.forward(batch)
    result_dict = model.initiate_sample_batch(
        batch_size=4
    )
    result_dict = model.training_step(batch, batch_idx=0)
    print(result_dict)
    res = model.sample(show_bar=True,samp_acc_factor=1)
    print(res)
    return result_dict

if __name__ == "__main__":
    main()
    print("success")

    
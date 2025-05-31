import logging
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
import torch.distributed as dist
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import gc

from torch_geometric.data import Data, Batch

class MOFDatamodule(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, predict_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        if hasattr(self.sampler_cfg, 'batch_strategy') and self.sampler_cfg.batch_strategy == 'TimeBatch':
            batch_sampler = TimeBatcher(
                    sampler_cfg=self.sampler_cfg,
                    dataset=self._train_dataset,
                    rank=rank,
                    num_replicas=num_replicas,
                )
        
        elif hasattr(self.sampler_cfg, 'batch_strategy') and self.sampler_cfg.batch_strategy == 'simple':
            batch_sampler = BatchSampler(
                    sampler = DistributedSampler(self._train_dataset, shuffle=True),
                    batch_size = self.sampler_cfg.max_batch_size,
                    drop_last = False
                )
        else:
            batch_sampler=DynamicBatcher(
                    sampler_cfg=self.sampler_cfg,
                    dataset=self._train_dataset,
                    rank=rank,
                    num_replicas=num_replicas,
                )
        if hasattr(self.sampler_cfg, 'collate_fn') and self.sampler_cfg.collate_fn == 'pyg':    
            collate_fn = collate_mof_data_pyg
        else:
            collate_fn = collate_mof_data
        return DataLoader(
            self._train_dataset,
            batch_sampler = batch_sampler,
            collate_fn = collate_fn,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            # persistent_workers=False
        )

    def val_dataloader(self):
        if hasattr(self.sampler_cfg, 'collate_fn') and self.sampler_cfg.collate_fn == 'pyg':    
            collate_fn = collate_mof_data_pyg
        else:
            collate_fn = collate_mof_data   
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            collate_fn = collate_fn,
            batch_size = 300,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        if hasattr(self.sampler_cfg, 'collate_fn') and self.sampler_cfg.collate_fn == 'pyg':    
            collate_fn = collate_mof_data_pyg
        else:
            collate_fn = collate_mof_data  
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            batch_size = 300,
            collate_fn = collate_fn,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=False,
        )


def stack_w_padding_1d(tensors):
    max_len = max([t.shape[0] for t in tensors])
    suffix = tensors[0].shape[1:]
    ret = torch.zeros(len(tensors), max_len, *suffix).to(tensors[0])
    for i in range(len(tensors)):
        t = tensors[i]
        len_t = t.shape[0]
        ret[i, :len_t] = t
    return ret


def collate_mof_data(batch):

            # - rotmats_1: [M, 3, 3]
            # - trans_1: [M, 3]
            # - res_mask: [M,]
            # - diffuse_mask: [M,]
            # - local_coords: [N, 3]
            # - gt_coords: [N, 3]
            # - bb_num_vec: [M,]
            # - bb_emb: [M, 3]
            # - atom_types: [N,]
            # - lattice: [6,]
            # - cell: [3, 3]
    specific_keys = ['atom_types', 'local_coords', 'gt_coords']
    keys = [k for k in batch[0].keys() if k not in specific_keys]
    batch_col = {k:stack_w_padding_1d([b[k] for b in batch]) for k in keys}
    for k in specific_keys:
        batch_col[k] = torch.cat([b[k] for b in batch], dim=0)
    return batch_col

def collate_mof_data_pyg(batch):

            # - rotmats_1: [M, 3, 3]
            # - trans_1: [M, 3]
            # - res_mask: [M,]
            # - diffuse_mask: [M,]
            # - local_coords: [N, 3]
            # - gt_coords: [N, 3]
            # - bb_num_vec: [M,]
            # - bb_emb: [M, 3]
            # - atom_types: [N,]
            # - lattice: [6,]
            # - cell: [3, 3]
    datalist = []
    cluster_idx = []
    tot_clusters = 0
    for b in batch:
        b['lattice_1'] = b['lattice_1'].reshape(1,6)
        b['cell_1'] = b['cell_1'].reshape(1,3,3)
        data = Data.from_dict(b)
        n_nodes = b['bb_num_vec'].shape[0]
        data.num_nodes = n_nodes
        data.num_atoms = n_nodes
        if hasattr(data, 'num_clusters'):
            cluster_idx.append(data.cluster_index + tot_clusters)
            tot_clusters = tot_clusters + data.num_clusters.sum()
        datalist.append(data)
    
    batch = Batch.from_data_list(datalist)
    if hasattr(batch, 'cluster_index'):
        cluster_idx = torch.cat(cluster_idx, dim=0)
        batch.cluster_index = cluster_idx
    return batch
        


class DynamicBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            dataset,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        
        self._sampler_cfg = sampler_cfg
        self._dataset_indices = np.arange(len(dataset))
        self._dataset = dataset

        # self._num_batches = math.ceil(len(dataset) / self.num_replicas)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._dataset_indices
        if self.shuffle: 
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = np.array([indices[i] for i in new_order])

        collected_batches = []
        idx_iter = indices
        if self.rank == 0:
            idx_iter = tqdm(idx_iter)

        cur_batch = []
        cur_square = 0
        for idx in idx_iter:
            num_atoms = self._dataset[idx]['atom_types'].shape[0]
            num_atoms_square = num_atoms ** 2
            if num_atoms_square >= self._sampler_cfg.max_num_res_squared:
                collected_batches.append([idx])
                continue
            if cur_square + num_atoms_square > self._sampler_cfg.max_num_res_squared:
                collected_batches.append(cur_batch)
                cur_batch = [idx]
                cur_square = num_atoms_square
            else:
                cur_batch.append(idx)
                cur_square += num_atoms_square
                if len(cur_batch) == self.max_batch_size:
                    collected_batches.append(cur_batch)
                    cur_batch = []
                    cur_square = 0
        collected_batches.append(cur_batch)

        if len(collected_batches) % self.num_replicas != 0:
            padding_len = self.num_replicas - len(collected_batches) % self.num_replicas
            collected_batches = collected_batches + collected_batches[:padding_len]

        if len(collected_batches) > self.num_replicas:
            collected_batches = collected_batches[self.rank::self.num_replicas]
        else:
            collected_batches = collected_batches

        return collected_batches

    def _create_batches(self):
        all_batches = self._replica_epoch_batches()
        self.sample_order = all_batches
    
    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)
    
    def __len__(self):
        return len(self.sample_order)


class TimeBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            dataset,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        
        self._sampler_cfg = sampler_cfg
        self._dataset_indices = np.arange(len(dataset))
        self._dataset = dataset

        self._num_batches = math.ceil(len(dataset) / self.num_replicas)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._dataset_indices
        if self.shuffle: 
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = np.array([indices[i] for i in new_order])
        
        if len(indices) > self.num_replicas:
            replica_indices = indices[self.rank::self.num_replicas]
        else:
            replica_indices = indices
        
        # Dynamically determine max batch size
        repeated_indices = []
        for idx in replica_indices:
            num_atoms = self._dataset[idx]['atom_types'].shape[0]
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // num_atoms**2 + 1,
            )
            repeated_indices.append([idx] * max_batch_size)

        return repeated_indices

    def _create_batches(self):
        all_batches = []
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches
    
    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)
    
    def __len__(self):
        return len(self.sample_order)
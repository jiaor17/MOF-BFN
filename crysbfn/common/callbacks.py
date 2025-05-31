from typing import Any, Optional, Union, Dict
from func_timeout import FunctionTimedOut, func_timeout
import hydra
from p_tqdm import p_map,p_umap,t_map
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
import torch.nn.functional as F
from absl import logging
import time
import os
import glob
from torch.optim import Optimizer
from copy import deepcopy
from overrides import overrides
from tqdm import tqdm
import crysbfn.evaluate
from crysbfn.pl_modules.crysbfn_csp_plmodel import CrysBFN_CSP_PL_Model
from pytorch_lightning.utilities import rank_zero_only
from tqdm import trange
from hydra.core.hydra_config import HydraConfig

from crysbfn.pl_modules.crysbfn_plmodel import CrysBFN_PL_Model

class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


class Gradient_clip(Callback):
    # gradient clupping for
    def __init__(self, Q=Queue(3000), maximum_allowed_norm=1e15,use_queue_clip=False) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        self.maximum_allowed_norm = maximum_allowed_norm
        self.use_queue_clip = use_queue_clip

    @overrides
    def on_after_backward(self, trainer, pl_module) -> None:
        # zero graidents if they are not finite
        if not all([torch.isfinite(t.grad if t.grad is not None else torch.tensor(0.)).all() for t in pl_module.parameters()]):
            hydra.utils.log.info("Gradients are not finite number")
            pl_module.zero_grad()
            return
        if not self.use_queue_clip:
            parameters = [p for p in pl_module.parameters() if p.grad is not None]
            device = parameters[0].grad.device
            parameters = [p for p in parameters if p.grad is not None]
            norm_type = 2.0
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
            wandb.log({"grad_norm": total_norm})
            return
        minimum_queue_length = 10
        optimizer = trainer.optimizers[0]
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )
        if len(self.gradnorm_queue) < minimum_queue_length:
            self.gradnorm_queue.add(float(grad_norm))
        elif float(grad_norm) > self.maximum_allowed_norm:
            optimizer.zero_grad()
            hydra.utils.log.info(
                f"Too large gradient with value {grad_norm:.1f}, NO UPDATE!"
            )
        elif float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))
        # if float(grad_norm) > max_grad_norm:
        #     hydra.utils.log.info(
        #         f"Clipped gradient with value {grad_norm:.1f} "
        #         f"while allowed {max_grad_norm:.1f}",
        #     )
        wandb.log({"grad_norm": grad_norm})
        # pl_module.log_dict(
        #     {"grad_norm": grad_norm},
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     # batch_size=pl_module.hparams.data.datamodule.batch_size.train,
        # )



class EMACallback(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        ema_device: Optional[Union[torch.device, str]] = None,
        pin_memory=True,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device: str = (
            f"{ema_device}" if ema_device else None
        )  # perform ema on different device from the model
        self.ema_pin_memory = (
            pin_memory if torch.cuda.is_available() else False
        )  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        # Update EMA weights
        # start_step = 1500
        start_step = 5
        if trainer.global_step > start_step:
            if not self._ema_state_dict_ready and pl_module.global_rank == 0:
                self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
                if self.ema_device:
                    self.ema_state_dict = {
                        k: tensor.to(device=self.ema_device)
                        for k, tensor in self.ema_state_dict.items()
                    }

                if self.ema_device == "cpu" and self.ema_pin_memory:
                    self.ema_state_dict = {
                        k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                    }

                self._ema_state_dict_ready = True
            with torch.no_grad():
                for key, value in self.get_state_dict(pl_module).items():
                    ema_value = self.ema_state_dict[key]
                    ema_value.copy_(
                        self.decay * ema_value + (1.0 - self.decay) * value,
                        non_blocking=True,
                    )
        
    def load_ema(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # print('here is the EMA call back start!!!')
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        # trainer.strategy.broadcast(self.ema_state_dict, 0)

        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), (
            f"There are some keys missing in the ema static dictionary broadcasted. "
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        )
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    def unload_ema(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # print('here is the EMA call back end!!!')
        
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ):
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready
        # return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
        # callback_state: Dict[str, Any]
    ) -> None:
        callback_state = checkpoint
        if callback_state is None:
            self._ema_state_dict_ready = False
            print('no ema state!!!')
        else:
            self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
            self.ema_state_dict = callback_state["ema_state_dict"]
            print('load ema state!!!')
            if self.ema_device:
                    self.ema_state_dict = {
                        k: tensor.to(device=self.ema_device)
                        for k, tensor in self.ema_state_dict.items()
                    }
            # pl_module.load_state_dict(self.ema_state_dict, strict=False)

from crysbfn.common.data_utils import Crystal,ray_crys_map
from torch_geometric.data import Data, Batch
from crysbfn.evaluate.compute_metrics import RecEval,GenEval
from scipy.spatial.distance import cdist
import wandb
import crysbfn
import ray

# @ray.remote(num_gpus=1)
@ray.remote
def crys_map(x,check_comp=True, use_fingerprint=True):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen.analysis.local_env')
    return Crystal(x, species_tolerence=9,check_comp=check_comp, use_fingerprint=use_fingerprint)

def compute_structure_dist(crys, gt_crys):
    struc_fps = [c.struct_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
    }

    return combined_dist_dict

class GenEvalCallback(pl.Callback):
    def __init__(self, cfg = None, num_samples=50, sample_steps=1000, datamodule=None, use_ray=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_ray = False
        if use_ray and cfg.logging.run_gen:
            if not ray.is_initialized():
                self.context = ray.init(runtime_env={"py_modules": [crysbfn,crysbfn.evaluate]},num_cpus=20)
        self.num_samples = cfg.logging.gen_check.num_samples
        
        if cfg.model.BFN.dtime_loss:
            self.sample_steps = cfg.model.BFN.dtime_loss_steps
        else:
            self.sample_steps = sample_steps
        self.val_dataset = datamodule.val_datasets[0]
        self.ref_crys_list = self.compute_dist('val')
        self.show_bar = True
        self.run_func_timeout = cfg.logging.gen_check.run_func_timeout
        self.wait_count = 0
        self.ray_futures = []
        
        self.best_score = 1
        self.monitor_metric = 'metric_amsd_precision'
        self.best_dir = None
        self.newest_dir = None

    def log_image(self, trainer:pl.Trainer, pl_module:pl.LightningModule, crys_list, num_log=5):
        epoch_idx = pl_module.current_epoch
        # 从crys_list中选择num_log个crys
        log_images = []
        cnt = 0
        for crys in crys_list:
            res = crys.visualize()
            if res != None:
                log_images.append(res)
                cnt += 1
            if cnt >= num_log:
                break
        # wandb.log({f"epoch {epoch_idx} images:": [wandb.Image(i) for i in log_images]})
        pl_module.log_dict({f"epoch {epoch_idx} images:": [wandb.Image(i) for i in log_images]})
    
    def run_function(self, func, *args, **kwargs):
        try:
            func_timeout(self.run_func_timeout+120, func, args=args, kwargs=kwargs)
            return True
        except FunctionTimedOut:
            print(f"Function {func} Time out!")
        return False
    
    def run_callback(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not isinstance(pl_module, CrysBFN_PL_Model):
            return
        DEBUG_MODE = self.cfg.logging.debug_mode
        if (not (self.cfg.logging.run_gen and \
            (pl_module.current_epoch % self.cfg.logging.gen_check_interval == 0) and \
                pl_module.current_epoch > self.cfg.logging.gen_check.start_epoch)) and (not DEBUG_MODE):
            return
        # prev_stat = deepcopy(pl_module.state_dict())
        # 手动开一下ema
        if self.cfg.train.ema.enable:
            ema_callback = [e for e in trainer.callbacks if isinstance(e, EMACallback)][0]
            ema_callback.load_ema(trainer, pl_module)
        model: CrysBFN_PL_Model = pl_module
        
        try:
            sampled_output = model.sample(self.num_samples,sample_steps=self.sample_steps,show_bar=self.show_bar,return_traj=DEBUG_MODE,back_sampling=False)

            crys_dict_list = []
            num_molecules = len(sampled_output['num_atoms'])
            run_dir = HydraConfig.get().run.dir
            cur_epoch = pl_module.current_epoch
            cur_steps = pl_module.global_step
            if self.newest_dir != None:
                os.remove(self.newest_dir)
            filename = f'newest-epoch={cur_epoch}-step={cur_steps}.ckpt'
            file_dir = os.path.join(run_dir, filename)
            trainer.save_checkpoint(file_dir)
            self.newest_dir = file_dir             
            if DEBUG_MODE:
                log_acc_trajs = [sampled_output['traj'][i]['log_acc'] for i in range(len(sampled_output['traj']))]
                traj = torch.stack(log_acc_trajs,dim=-1).cuda() if DEBUG_MODE else None
                metrices = self.analyze_traj(sampled_output['traj'])
            
            for idx in range(num_molecules):
                select_indices = torch.where(sampled_output['segment_ids'] == idx)[0]
                frac_coords = torch.index_select(sampled_output['frac_coords'], dim=0, index=select_indices)
                atom_types = torch.index_select(sampled_output['atom_types'], dim=0, index=select_indices)
                num_atoms = sampled_output['num_atoms'][idx].cpu().detach().numpy()
                lengths = sampled_output['lengths'][idx].cpu().detach().numpy()
                angles = sampled_output['angles'][idx].cpu().detach().numpy()
                acc_traj = torch.index_select(traj,dim=0,index=select_indices) if DEBUG_MODE else None
                
                crys_dict = {
                    'frac_coords': frac_coords.cpu().detach().numpy(),
                    'atom_types': atom_types.cpu().detach().numpy(),
                    'num_atoms': num_atoms,
                    'lengths': lengths,
                    'angles': angles,
                    'acc_traj': acc_traj
                }
                crys_dict_list.append(crys_dict)
            if self.use_ray:
                crys_list = ray.get([crys_map.remote(x) for x in crys_dict_list])
            else:
                crys_list = p_map(lambda x: Crystal(x, species_tolerence=10),crys_dict_list)
            gen_eval = GenEval(pred_crys=crys_list, 
                            gt_crys=self.ref_crys_list, 
                            eval_model_name=self.cfg.data.eval_model_name)

            metrics = gen_eval.get_metrics(get_prop=False)
            metrics['success_rate'] = len(crys_list)/len(crys_dict_list)
            print(metrics)
            monitor_metrics = ['amsd_precision','amcd_precision','wdist_num_elems','wdist_density']
            avg_monitor_values = np.prod([metrics[e]*10 for e in monitor_metrics])
            model.log_dict(
                # metrics,
                {f'metric_{k}':v for k,v in metrics.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            
            if avg_monitor_values < self.best_score:
                if self.best_dir != None:
                    os.remove(self.best_dir)
                self.best_score = avg_monitor_values
                filename = f'ema_avgs={self.best_score:.3f}-epoch={cur_epoch}-step={cur_steps}.ckpt'
                file_dir = os.path.join(run_dir, filename)
                trainer.save_checkpoint(file_dir)
                self.best_dir = file_dir
        except Exception as e:       
            print('metrics is None')
        # 手动关一下ema
        if self.cfg.train.ema.enable:
            ema_callback.unload_ema(trainer, pl_module)
        return

    def compute_dist(self,mode='val'):
        if os.path.exists(os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt')):
            crys_list = torch.load(os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt'))
            return crys_list
    
        crys_dict_list = []
        for idx, data in enumerate(self.val_dataset):
            frac_coords = data['frac_coords'].numpy()
            atom_types = data['atom_types'].numpy()
            num_atoms = data['num_atoms'].numpy()
            lengths = data['lengths'][0].numpy()
            angles = data['angles'][0].numpy()
            crys_dict = {
                'frac_coords': frac_coords,
                'atom_types': atom_types,
                'num_atoms': num_atoms,
                'lengths': lengths,
                'angles': angles
            }
            crys_dict_list.append(crys_dict)
        if self.use_ray:
            crys_list = ray.get([crys_map.remote(x) for x in crys_dict_list])
        else:
            crys_list = p_map(lambda x: Crystal(x),crys_dict_list)
        # 保存val_crys_list到self.cfg.data.root_path
        torch.save(crys_list, os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt'))
        return crys_list 
    
    @overrides
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # return self.run_callback(trainer,pl_module)
        return self.run_function(self.run_callback, trainer,pl_module)

    @overrides
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ray.shutdown()
        return


class CSPEvalCallback(pl.Callback):
    def __init__(self, cfg = None, num_samples=50, sample_steps=1000, datamodule=None, use_ray=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_ray = use_ray
        if use_ray and cfg.logging.run_gen:
            if not ray.is_initialized():
                self.context = ray.init(runtime_env={"py_modules": [crysbfn,crysbfn.evaluate]},num_cpus=20)
        self.num_samples = cfg.logging.gen_check.num_samples
        
        if cfg.model.BFN.dtime_loss:
            self.sample_steps = cfg.model.BFN.dtime_loss_steps
        else:
            self.sample_steps = sample_steps
        self.val_dataset = datamodule.val_datasets[0]
        self.ref_crys_list = self.compute_dist(mode='val')
        self.show_bar = True
        self.run_func_timeout = cfg.logging.gen_check.run_func_timeout
        self.wait_count = 0
        self.ray_futures = []
        
        self.best_score = 100
        self.best_dir = None
    
    def run_function(self, func, *args, **kwargs):
        try:
            func_timeout(self.run_func_timeout+120, func, args=args, kwargs=kwargs)
            return True
        except FunctionTimedOut:
            print(f"Function {func} Time out!")
        return False
    
    def output2crys(self, sampled_outputs):
        crys_dict_list = []
        for sampled_output in sampled_outputs:
            num_molecules = len(sampled_output['num_atoms'])
            for idx in range(num_molecules):
                select_indices = torch.where(sampled_output['segment_ids'] == idx)[0]
                frac_coords = torch.index_select(sampled_output['frac_coords'], dim=0, index=select_indices)
                atom_types = torch.index_select(sampled_output['atom_types'], dim=0, index=select_indices)
                num_atoms = sampled_output['num_atoms'][idx].cpu().detach().numpy()
                lengths = sampled_output['lengths'][idx].cpu().detach().numpy()
                angles = sampled_output['angles'][idx].cpu().detach().numpy()
                crys_dict = {
                    'frac_coords': frac_coords.cpu().detach().numpy(),
                    'atom_types': atom_types.cpu().detach().numpy(),
                    'num_atoms': num_atoms,
                    'lengths': lengths,
                    'angles': angles,
                }
                crys_dict_list.append(crys_dict)
        # crys_list = ray.get([crys_map.remote(x,check_comp=False) for x in crys_dict_list])
        crys_list = p_map(lambda x: Crystal(x,check_comp=False, use_fingerprint=False), crys_dict_list)
        return crys_list
    
    def run_callback(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not isinstance(pl_module, CrysBFN_CSP_PL_Model):
            return
        DEBUG_MODE = self.cfg.logging.debug_mode
        if (not (self.cfg.logging.run_gen and \
            (pl_module.current_epoch % self.cfg.logging.gen_check_interval == 0) and \
                pl_module.current_epoch > self.cfg.logging.gen_check.start_epoch)) and (not DEBUG_MODE):
            return
        # 手动开一下ema
        if self.cfg.train.ema.enable:
            ema_callback = [e for e in trainer.callbacks if isinstance(e, EMACallback)][0]
            ema_callback.on_validation_start(trainer, pl_module)
        
        model: CrysBFN_CSP_PL_Model = pl_module
        crys_dict_list = []
        for idx, batch in enumerate(tqdm(trainer.datamodule.test_dataloader()[0],desc='CSP Eval')):
            sampled_output = model.sample(batch=batch, sample_steps=self.sample_steps,show_bar=False,return_traj=DEBUG_MODE,back_sampling=False)
            crys_dict_list.append(sampled_output)
            if idx >= self.cfg.logging.gen_check.num_samples//len(batch.num_atoms):
                break
        crys_list = self.output2crys(crys_dict_list)
        torch.save(crys_list, os.path.join(self.cfg.data.root_path, f'ep_{pl_module.current_epoch}_csp_crys_list.pt'))
        recon_eval = RecEval(pred_crys=crys_list, gt_crys=self.ref_crys_list[:len(crys_list)])
        # self.log_image(trainer, pl_module, crys_list)
        try:
            metrics = recon_eval.get_metrics()
            print(metrics)
            avg_monitor_values = (1-metrics['match_rate']) * metrics['rms_dist']
            model.log_dict(
                {f'metric_{k}':v for k,v in metrics.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            if avg_monitor_values < self.best_score:
                if self.best_dir != None:
                    os.remove(self.best_dir)
                self.best_score = avg_monitor_values
                run_dir = HydraConfig.get().run.dir
                cur_epoch = pl_module.current_epoch
                cur_steps = pl_module.global_step
                filename = f'ema_avgs={self.best_score:.3f}-epoch={cur_epoch}-step={cur_steps}.ckpt'
                file_dir = os.path.join(run_dir, filename)
                trainer.save_checkpoint(file_dir)
                self.best_dir = file_dir
        except Exception as e:       
            print('metrics is None')
        # 手动关一下ema
        if self.cfg.train.ema.enable:
            ema_callback.on_validation_end(trainer, pl_module)
        return

    def compute_dist(self,mode='val'):
        if os.path.exists(os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt')):
            val_crys_list = torch.load(os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt'))
            return val_crys_list
    
        crys_dict_list = []
        for idx, data in enumerate(self.val_dataset):
            frac_coords = data['frac_coords'].numpy()
            atom_types = data['atom_types'].numpy()
            num_atoms = data['num_atoms'].numpy()
            lengths = data['lengths'][0].numpy()
            angles = data['angles'][0].numpy()
            crys_dict = {
                'frac_coords': frac_coords,
                'atom_types': atom_types,
                'num_atoms': num_atoms,
                'lengths': lengths,
                'angles': angles
            }
            crys_dict_list.append(crys_dict)
        val_crys_list = ray.get([crys_map.remote(x, use_fingerprint=False) for x in crys_dict_list])
        # 保存val_crys_list到self.cfg.data.root_path
        torch.save(val_crys_list, os.path.join(self.cfg.data.root_path, f'{mode}_crys_list.pt'))
        return val_crys_list
    
    @overrides
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # return self.run_callback(trainer,pl_module)
        return self.run_function(self.run_callback, trainer,pl_module)

    @overrides
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ray.shutdown()
        return


        
                    
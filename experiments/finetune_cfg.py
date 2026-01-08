import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data.datasetv4 import MOFDataset
from data.dataloader_ori import MOFDatamodule
from models.crysbfn_bhm_cond_module import CrysBFN_PL_Model
from common.utils import PROJECT_ROOT
from experiments import utils as eu
# from common.callbacks import RecoveryCallback
from common.callbacks import EMACallback, LMDBCallback
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('highest')


class ResetOptimizerCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        # Clear optimizer and lr_scheduler
        trainer.strategy._optimizers = []
        trainer.strategy._lightning_optimizers = []
        trainer.strategy.lr_scheduler_configs = []
        
        # Reconfigure optimizer and lr_scheduler
        optimizer_dict = pl_module.configure_optimizers()
        trainer.optimizers = [optimizer_dict['optimizer']]
        trainer.lr_schedulers = [optimizer_dict['lr_scheduler']]

class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._cond_cfg = cfg.conditions
        self._task = self._data_cfg.task
        self._setup_dataset()
        self._datamodule: LightningDataModule = MOFDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        self._train_device_ids = eu.get_available_device(self._exp_cfg.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(local_rank, self._train_device_ids, self._train_device_ids[local_rank])
        self.running_device = f'cuda:{self._train_device_ids[local_rank]}'
        self._module: LightningModule = CrysBFN_PL_Model(self._cfg, device=self.running_device)

        if self._exp_cfg.seed is not None:
            log.info(f'Setting seed to {self._exp_cfg.seed}')
            self._set_seed(self._exp_cfg.seed)

    def _setup_dataset(self):
        self._train_dataset = MOFDataset(
            cache_path=os.path.join(self._data_cfg.cache_dir, 'MetalOxo.lmdb'),
            dataset_cfg=self._data_cfg,
            is_training=True,
            use_block_emb=True,
            use_prop_dict=True,
            split_idx=os.path.join(self._data_cfg.cache_dir, 'train_bb_prop.idx'),
            used_conditions=self._cond_cfg
        )
        self._valid_dataset = MOFDataset(
            cache_path=os.path.join(self._data_cfg.cache_dir, 'MetalOxo.lmdb'),
            dataset_cfg=self._data_cfg,
            is_training=False,
            use_block_emb=True,
            use_prop_dict=True,
            split_idx=os.path.join(self._data_cfg.cache_dir, 'val_bb_prop.idx'),
            used_conditions=self._cond_cfg
        )

    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Ensuring deterministic behavior in CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
           
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            logger.watch(
                self._module,
                log=self._exp_cfg.wandb_watch.log,
                log_freq=self._exp_cfg.wandb_watch.log_freq
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Recovery when crash
            # callbacks.append(RecoveryCallback())
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            # Learning rate monitor
            callbacks.append(LearningRateMonitor(logging_interval='step'))

            if hasattr(self._cfg, 'ema') and self._cfg.ema.enable:
                callbacks.append(EMACallback(decay=self._cfg.ema.decay, start_step=self._cfg.ema.start_step, ema_device=self.running_device))

            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg, allow_val_change=True)
        
        if self._exp_cfg.pretrained_ckpt is not None:
            pretrained_ckpt = torch.load(self._exp_cfg.pretrained_ckpt)
            pretrained_dict = pretrained_ckpt['state_dict']
            scratch_dict = self._module.state_dict()
            scratch_dict.update(
                (k, pretrained_dict[k]) for k in scratch_dict.keys() & pretrained_dict.keys()
            )
            self._module.load_state_dict(scratch_dict, strict=True)
            if not self._exp_cfg.full_finetuning:
                for name, param in self._module.named_parameters():
                    if name in set(pretrained_dict.keys()):
                        param.requires_grad_(False)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="bfn_base_bhm_cond.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()

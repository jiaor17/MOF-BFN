import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Any, Optional, Union, Dict
from copy import deepcopy
from pytorch_lightning.utilities import rank_zero_only
from overrides import overrides

class RecoveryCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.detect_crash(trainer, pl_module):
            print(f"Training crashed at epoch {trainer.current_epoch}")
            self.restore_checkpoint(trainer, pl_module)

    def detect_crash(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("valid/loss", None)
        epoch = pl_module.current_epoch
        if epoch >= 50 and val_loss > 2.0:
            return True
        return False

    def restore_checkpoint(self, trainer, pl_module):
        device = f'cuda:{torch.cuda.current_device()}'    
        last_path = os.path.join(pl_module.checkpoint_dir, 'last.ckpt')
        print(f'Load state from {last_path}')
        checkpoint = torch.load(last_path, map_location = device)
        pl_module.load_state_dict(checkpoint["state_dict"])  # 恢复模型参数
        trainer.optimizers[0].load_state_dict(checkpoint["optimizer_states"][0])  # 恢复优化器


class LMDBCallback(Callback):

    
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.datamodule._train_dataset.processed_env is not None:
            trainer.datamodule._train_dataset.processed_env.close()
            trainer.datamodule._train_dataset.processed_env = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.datamodule._valid_dataset.processed_env is not None:
            trainer.datamodule._valid_dataset.processed_env.close()
            trainer.datamodule._valid_dataset.processed_env = None  


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
        start_step = 10000,
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
        self.start_step = start_step

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
        # start_step = 5
        start_step = self.start_step
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
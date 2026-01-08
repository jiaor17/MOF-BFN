"""Script for running inference and evaluation."""

import os
import time
import random
import numpy as np
from dotenv import load_dotenv
import hydra
import torch
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
# from data.dataset import MOFDataset
from data.dataloader import MOFDatamodule
from models.crysbfn_bhm_cond_module import CrysBFN_PL_Model
from common.utils import PROJECT_ROOT
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

train_dist = [0.0,
 0.0,
 0.00010963822810918597,
 0.004539707882645982,
 0.5992825548448105,
 1.0278583885236185e-05,
 0.0027169723403307647,
 0.00017473592604901514,
 0.10874056511654201,
 0.10723646567466912,
 0.0,
 1.0278583885236185e-05,
 0.07465678095309883,
 0.0,
 0.0014390017439330658,
 0.0,
 0.052890166478797,
 0.0,
 0.047058783221239665,
 0.0,
 0.0011340704220043924]

class SampleDataset(Dataset):

    def __init__(self, total_num, prop_dict):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.prop_dict = prop_dict

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        for k, v in self.prop_dict.items():
            data[k] = v
        return data

class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._data_cfg = cfg.data
        self._cond_cfg = cfg.conditions
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        if self._infer_cfg.seed is not None:
            log.info(f'Setting seed to {self._infer_cfg.seed}')
            self._set_seed(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            # Create output directory.
            inference_dir = self._infer_cfg.inference_dir
            self._exp_cfg.inference_dir = inference_dir
            os.makedirs(inference_dir, exist_ok=True)
            log.info(f'Saving results to {inference_dir}')

            # Save config.
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')


        # Read checkpoint and initialize module.
        self._flow_module = CrysBFN_PL_Model.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            device=f'cuda:{local_rank}'
        )
        # self.load_ema(ckpt_path)
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg

    def load_ema(self, ckpt_path):
        model = torch.load(ckpt_path)
        if 'ema_state_dict' in model:
            print("Loaded EMA")
            self._flow_module.load_state_dict(model['ema_state_dict'])

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

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
    
    def run_sampling(self):
        devices = [0]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')

        prop_dict = {}
        for cond in self._cond_cfg:
            key = cond.key
            value = cond.target_value
            prop_dict[key] = value

        print(prop_dict)
        test_set = SampleDataset(500, prop_dict=prop_dict)
        dataloader = DataLoader(test_set, batch_size = 512)

        self._flow_module.change_weight(2.0)


        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="bfn_infer_bhm_cond")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    load_dotenv()
    print(str(PROJECT_ROOT))
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
# MOF-BFN: Metal-Organic Frameworks Structure Prediction via Bayesian Flow Networks (NeurIPS 2025)

Official Implementation of MOF-BFN: Metal-Organic Frameworks Structure Prediction via Bayesian Flow Networks.

![Overview](fig/overview.png "Overview")

## Setup

### Setup Conda Environment

```bash
mamba env create -f env.yml
pip install git+https://github.com/microsoft/MOFDiff.git
pip install -e .
mamba activate mof-bfn
```

### Download Zeo++ for Property Evaluation

Please Install `Zeo++` [here](https://www.zeoplusplus.org/download.html).

### Setup dotenv File

Rename the `.env.template` file into `.env` and specify the following variables.

```bash
export PROJECT_ROOT=<abs_dir_of_this_repo>
export DATASET_ROOT=<abs_dir_of_sp_data>
export DATASET_DNG_ROOT=/<abs_dir_of_dng_data>
export CHECKPOINT_ROOT=<abs_dir_of_ckpts>
export ZEO_PATH=<abs_dir_of_zeo++> # for property prediction
export MOFDIFF_PATH=<abs_dir_of_mofdiff> # for dng evaluation
```

## Datasets

For MOF structure prediction, we follow the data splits and benchmark used in [MOFFlow](https://github.com/nayoung10/MOFFlow). The datasets are processed into `lmdb` format. For de novo generation, we follow the splits in [MOFDiff](https://github.com/microsoft/MOFDiff) and pre-computed the block embeddings. The processed datasets are given [here](https://doi.org/10.6084/m9.figshare.31029235). The processed datasets are listed as follows:

```bash
|-- sp # DATASET_ROOT
    |-- train_preprocessed.lmdb
	|-- val_preprocessed.lmdb
    |-- test_preprocessed.lmdb
|-- dng # DATASET_DNG_ROOT
    |-- MetalOxo_bb_prop_preprocessed.lmdb
    |-- train_bb_prop.idx
    |-- val_bb_prop.idx
    |-- bb_emb_space_tot_blocks_processed.lmdb
    |-- bb_emb_space_tot_z.pt
```

## Training

```bash
# MOF structure prediction
python experiments/train_csp.py experiment.wandb.name=<expname>

# De Novo Generation
python experiments/train.py experiment.wandb.name=<expname>
```

We have provided the pretrained checkpoints [here](https://doi.org/10.6084/m9.figshare.31029511).

## Inference & Evaluation

### MOF structure prediction

```bash
python -m experiments.infer_csp experiment.wandb.name=infer_csp inference.ckpt_path=<csp_model_dir>/ckpt/last.ckpt inference.inference_dir=<csp_infer_dir> inference.num_gpus=1
python -m evaluation.rmsd --cif_dir <csp_infer_dir> # RMSD Evaluation
python -m evaluation.property --cif_dir <csp_infer_dir> # Property Evaluation
```


### De Novo Generation

```bash
python -m experiments.infer experiment.wandb.name=infer_dng inference.ckpt_path=<dng_model_dir>/ckpt/last.ckpt inference.inference_dir=<dng_infer_dir>

# w/o LAMMPS relaxation

python -m postprocess.assemble --res_path <dng_infer_dir>/raw_results.pt --bb_z_path <dng_data_dir>/bb_emb_space_tot_z.pt --bb_blocks_path <dng_data_dir>/bb_emb_space_tot_blocks_processed.lmdb --max_process 1000
python -m evaluation.connect_check --res_path <infer_dir>/raw_results.pt --bb_z_path <dng_data_dir>/bb_emb_space_tot_z.pt --bb_blocks_path <dng_data_dir>/bb_emb_space_tot_blocks_processed.lmdb --max_process 1000
python -m evaluation.validity_check --res_path <infer_dir>/samples --max_process 1000

# w/ LAMMPS relaxation

python -m postprocess.assemble_w_local_edge --res_path <infer_dir>/raw_results.pt --bb_z_path <dng_data_dir>/bb_emb_space_tot_z.pt --bb_blocks_path <dng_data_dir>/bb_emb_space_tot_blocks.lmdb --max_process 1000
python -m postprocess.uff_relax --input <infer_dir>/bond_samples
python -m evaluation.validity_check --res_path <infer_dir>/bond_samples/relaxed --max_process 1000
```

## Conditional Fine-tuning

### Training

Please specify the targeted condition(s) under the `configs/conditions` folder similar to the example `vf.yaml`.

```bash
python experiments/finetune_cfg.py experiment.wandb.name=<expname> experiment.pretrained_ckpt=<dng_model_dir>/ckpt/last.ckpt conditions=<condition>
```

### Inference

```bash
python -m experiments.infer_cfg experiment.wandb.name=infer_cfg inference.ckpt_path=<cfg_model_dir>/ckpt/last.ckpt inference.inference_dir=<cfg_dir> conditions.0.target_value=<cond_value>
python -m postprocess.assemble_w_local_edge --res_path <cfg_dir>/raw_results.pt --bb_z_path <dng_data_dir>/bb_emb_space_tot_z.pt --bb_blocks_path <dng_data_dir>/bb_emb_space_tot_blocks.lmdb --max_process 500

# w/o LAMMPS relaxation
python -m evaluation.property --cif_dir <cfg_dir>/bond_samples/cif

# w/ LAMMPS relaxation
python -m postprocess.uff_relax --input <cfg_dir>/bond_samples
python -m evaluation.property --cif_dir <cfg_dir>/bond_samples/relaxed
```

## Acknowledgments

The main framework of this codebase is build upon [CrysBFN](https://github.com/wu-han-lin/CrysBFN) and [MOFFlow](https://github.com/nayoung10/MOFFlow). Raw datasets, benchmarks, and evaluations are partially sourced from [MOFDiff](https://github.com/microsoft/MOFDiff) and [MOFFlow](https://github.com/nayoung10/MOFFlow).

## Citation

Please consider citing our work if you find it helpful:

```
@inproceedings{
    jiao2025mofbfn,
    title={{MOF}-{BFN}: Metal-Organic Frameworks Structure Prediction via Bayesian Flow Networks},
    author={Rui Jiao and Hanlin Wu and Wenbing Huang and Yuxuan Song and Yawen Ouyang and Yu Rong and Tingyang Xu and Pengju Wang and Hao Zhou and Wei-Ying Ma and Jingjing Liu and Yang Liu},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=pNwiFucAtA}
}
```

## Contact

If you have any questions, feel free to reach us at:

Rui Jiao: [jiaor21@mails.tsinghua.edu.cn](mailto:jiaor21@mails.tsinghua.edu.cn)
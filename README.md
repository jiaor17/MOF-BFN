## Environment 

```bash
mamba env create -f env.yml
pip install git+https://github.com/microsoft/MOFDiff.git
pip install -e .
mamba activate mof-bfn
```


## Training


```bash
# MOF structure prediction
python experiments/train_csp.py experiment.wandb.name=<expname>

# De Novo Generation
python experiments/train.py experiment.wandb.name=<expname>
```

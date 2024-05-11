# Master thesis - Permutation Tree Invariant Graph Neural Networks
## Information
Author: Johannes P. Urban

## Installation procedure:
(python 3.9)
```bash
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg==2.5.2 -c pyg
pip install rdkit==2023.9.5
conda install jupyterlab numpy pandas matplotlib
pip install chainer-chemistry==0.7.1
conda install -c conda-forge torchmetrics==1.3.2
pip install "ray[tune]==2.9.3" torch torchvision
pip install hyperopt==0.2.7
pip install multiprocess==0.70.16
```

## Content information:

- notebooks: Experiments and notebooks, TODO: rename to experiments in the future
- ptgnn: Related code realizing model, framework, etc. TODO: rename to ptignn
- .gitignore: Prevent pushing of datasets to GitHub repo
- ReadMe.md: General information and installation instructions
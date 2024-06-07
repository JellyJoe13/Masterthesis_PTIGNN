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

- experiments: Experiments and notebooks
- ptgnn: Related code realizing model, framework, etc. Model name is technically Permutation Tree Invariant Graph Neural Network (PTIGNN),
however, at the beginning of the project the general term PTGNN was chosen which does not specify whether th emodel is to be invariant or
equivariant. (Changing it now would change almost every file which is guaranteed to cause errors, except everything is retested)
- .gitignore: Prevent pushing of datasets to GitHub repo
- ReadMe.md: General information and installation instructions

## Notebook information:

Notebooks have been opened and used with DataSpell and not directly with jupyter notebook/lab. Notebooks may require the following addition (should be included in most of them) to successfully import pt(i)gnn package from root folder:

```python
import sys
sys.path.append("../") # or "../../" or more incase notebook is contained in a deeper subfolder structure 
```

## Notice on foreign sources

Other works have been used in some parts to recreate functionality and to realize experiment setups to test the methods
proposed by this thesis. This mainly refers to contents from ChiENN (https://github.com/gmum/ChiENN). Relevant parts are marked and linked to their 
source, these contents are licenced under an MIT licence, see https://github.com/gmum/ChiENN/blob/master/LICENSE.
# CellDART-pytorch

An unofficial implementation of 'CellDART: cell type inference by domain adaptation of single-cell and spatial transcriptomics data' ([link](https://github.com/mexchy1000/CellDART)) based on pytorch-lightning.

## Requirements
The major requirements of establishing the environment are listed as follow.
```
pytorch-lightning==1.5.10
scanpy==1.8.2
pandas=1.3.4=py38h8c16a72_0
numpy=1.21.2
scipy=1.7.1
pytorch=1.8.0
```
You may refer to the [pytorch official website](https://pytorch.org/) for installing proper distribution with CUDA on your machine.

We also combined [scvi-tools](https://scvi-tools.org/) analysis within the preprocessing step.
```
scvi-tools=0.15.0
```

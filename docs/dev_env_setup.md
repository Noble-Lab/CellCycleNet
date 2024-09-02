# CellCycleNet Dev Env Setup

## Installation of GitHub repo package via conda

1. Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Install Mamba to facilitate installation of GPU-enabled dependencies:

```
$ conda install -n base -c conda-forge mamba
```

3. Clone this repo, then create the provided [conda environment](../envs/dev_env.yml).

```
$ cd CellCycleNet
$ mamba env create -f envs/dev_env.yml
```

4. Activate the new conda environment and install `cellcyclenet`:

```
$ conda activate ccn_dev_env
$ pip install .
```

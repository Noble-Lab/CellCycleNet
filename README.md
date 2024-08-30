# CellCycleNet

![CellCycleNet Diagram](./docs/img/CellCycleNet_diagram.png)

## Overview

TODO

## Installation

1. Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Install Mamba to facilitate installation of GPU-enabled dependencies:

```
$ conda install -n base -c conda-forge mamba
```

3. Clone this repo, then create the provided [conda environment](./environment.yml).

```
$ mamba env create -f environment.yml
$ conda activate ccn_dev_env
```

4. Activate the new conda environment and install `cellcyclenet`:

```
$ conda activate ccn_dev_env
$ cd CellCycleNet
$ pip install .
```

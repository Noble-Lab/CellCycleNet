# CellCycleNet

[**Installation**](#installation)
| [**Enviroment**](#enviroment)
| [**Example**](#example)
| [**Citation**](#citation)
| [**Contact**](#contact)

The cell cycle governs the proliferation, differentiation, and regeneration of all eukaryotic cells. Profiling cell cycle dynamics is therefore central to basic and biomedical research spanning development, health, aging, and disease. However, current approaches to cell cycle profiling involve complex interventions that may confound experimental interpretation. To facilitate more efficient cell cycle annotation of microscopy data, we developed CellCycleNet, a machine learning (ML) workflow designed to simplify cell cycle staging with minimal experimenter intervention and cost.
CellCycleNet accurately predicts cell cycle phase using only a fluorescent nuclear stain (DAPI) in fixed interphase cells. Using the Fucci2a cell cycle reporter system as ground truth, we collected two benchmarking image datasets and trained two ML models---a support vector machine (SVM) and a deep neural network---to classify nuclei as being in either the G1 or S/G2 phases of the cell cycle.
Our results suggest that CellCycleNet outperforms state-of-the-art SVM models on each dataset individually.
When trained on two image datasets simultaneously, CellCycleNet achieves the highest classification accuracy, with an improvement in AUROC of 0.08--0.09.
The model also demonstrates excellent generalization across different microscopes, achieving an AUROC of 0.95.
Overall, using features derived from 3D images, rather than 2D projections of those same images, significantly improves classification performance.
We have released our image data, trained models, and software as a community resource.

![CellCycleNet Diagram](https://raw.githubusercontent.com/Noble-Lab/CellCycleNet/main/docs/img/CellCycleNet_diagram.png)

## Installation<a id="installation"></a>

**Option 1:** Install via pip: `pip install cellcyclenet`

**Option 2:** Install via conda or build the development conda environment: [see documentation](https://github.com/Noble-Lab/CellCycleNet/blob/main/docs/dev_env_setup.md)

## Running the included examples <a id="examples"></a>

1. After installation, download `example_data.zip` from [here](https://beliveau-shared.s3.us-east-2.amazonaws.com/cellcyclenet/data/example_data.zip).

2. Try running the included examples on these example files using the included example notebooks:

	1. [Example #1](https://github.com/Noble-Lab/CellCycleNet/blob/main/notebooks/3D_prediction_demo.ipynb): Predict cell cycle stage from segmented 3D DAPI images
	2. [Example #2](https://github.com/Noble-Lab/CellCycleNet/blob/main/notebooks/3D_fine_tune_training_demo.ipynb): Fine tune pre-trained 3D model with additional training
	3. [Example #3](https://github.com/Noble-Lab/CellCycleNet/blob/main/notebooks/2D_prediction_demo.ipynb): Predict cell cycle stage from segmented 2D DAPI images
	4. [Example #4](https://github.com/Noble-Lab/CellCycleNet/blob/main/notebooks/2D_fine_tune_training_demo.ipynb): Fine tune pre-trained 2D model with additional training

 ## Contact<a id="contact"></a>
In case you have questions, reach out to `gangliuw@uw.edu` and/or 'eknich@uw.edu'.


## Citation<a id="citation"></a>
[Predicting cell cycle stage from 3D single-cell nuclear-stained images](https://doi.org/10.1101/2024.08.30.610553)

If you have found our work useful, please consider citing us:

```
Predicting cell cycle stage from 3D single-cell nuclear-stained images
Gang Li, Eva K. Nichols, Valentino E. Browning, Nicolas J. Longhi, Conor Camplisson, Brian J. Beliveau, William Stafford Noble
bioRxiv 2024.08.30.610553; doi: https://doi.org/10.1101/2024.08.30.610553

```


# GarmentNets Tracking

This is a fork of GarmentNets for the goal of extending GarmentNets' single prediction to tracking deformables. The code in this repository was written by Dylan Colli, Yating Lin, Abhinav Kumar, and Cheng Chi (Original paper author) for the University of Michigan ROB 599, "DeepRob: Deep Learning for Robot Perception" course.

Table of Contents
- [GarmentNets Tracking](#garmentnets-tracking)
  - [Setup](#setup)
    - [Python Environment](#python-environment)
    - [Data Download](#data-download)
    - [Model Download](#model-download)
    - [Configuring the Evaluation](#configuring-the-evaluation)
  - [Running Evaluation Script](#running-evaluation-script)
  - [Simulation](#simulation)
- [Original GarmentNets Documentation](#original-garmentnets-documentation)
  - [Cite this work](#cite-this-work)
  - [Datasets](#datasets)
  - [Sample Datasets](#sample-datasets)
  - [Pretrained Models](#pretrained-models)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Evaluation](#evaluation)
    - [Training](#training)

## Setup

### Python Environment

We provide a conda environment configuration for installing necessary packages in `environment.yml`. This environment can be installed by executing the following:
```
conda env create -n garmentnets --file environment.yml
conda activate garmentnets
```

### Data Download

We provide a minimal test set that can be [downloaded at this link](https://drive.google.com/drive/folders/1MIcf4LjCprNpu2Wf8OQojtkNcPEWQkms?usp=sharing).

### Model Download

Please reference [Pretrained Models](#pretrained-models) to obtain pretrained models.

### Configuring the Evaluation

We provide a hydra yaml file for running our evaluation script. This yml file can be found at `config/dynamics_test.yaml`. There are two fields that must be changed before the evaluation script can be run. The first is `zarr_path`. This should be set to the path of the downloaded evaluation data. Please remember to include `/TShirt` at the end of the path.

The second field to be set is `checkpoint_path`. This should be set to the location of the `Tshirt_pipeline.ckpt file which can be downloaded with the other pretrained models associated with this project. Please refer to [Pretrained Models](#pretrained-models) to obtain pretrained models.

## Running Evaluation Script

We provide an evaluation script at `eval_garment_simple.py`. Different configurations of the filter can be chosen by altering variables at the start of the `main()` function. `DYNAMICS` indicates whether rigid transformations or learned transformations are used to predict future states, `MATCHING` controls whether or not observations are used to update predicted states, and `PLOT` controls whether or not plots are generated.

## Simulation

For the original GarmentNets publication, Cheng did the simulation with Blender and has shared the repository with me. However, this code is several years old and relied on a Blender version that is significantly outdated. Due to this, I had to put in significant effort to revive the code. As such, I rewrote most of the code and only ported over the pertinent utility functions.

The [simulation pipeline README](./simulation/README.md) *verbosely* details my thoughts as I worked through this process in addition to how to run the simulation pipeline.

# Original GarmentNets Documentation

As this repository is a fork of the original GarmentNets repository, this README also contains the documentation included with that repository in an attempt to make this work as reproducible as possible. The following is the original documentation for the repository that contains the source code for the paper [GarmentNets:
Category-Level Pose Estimation for Garments via Canonical Space Shape Completion](https://garmentnets.cs.columbia.edu/). This paper has been accepted to ICCV 2021.

![Overview](assets/teaser_web.png)

## Cite this work
```
@inproceedings{chi2021garmentnets,
  title={GarmentNets: Category-Level Pose Estimation for Garments via Canonical Space Shape Completion},
  author={Chi, Cheng and Song, Shuran},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Datasets
1. [GarmentNets Dataset](https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset.zarr.tar) (GarmentNets training and evaluation) [Split in parts](https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset_parts/)

2. [GarmentNets Simulation Dataset](https://garmentnets.cs.columbia.edu/dataset/garmentnets_simulation_dataset.zarr.tar) (raw Blender simluation data to generate the GarmentNets Dataset) [Split in parts](https://garmentnets.cs.columbia.edu/dataset/garmentnets_simulation_dataset_parts/)

3. [CLOTH3D Dataset](https://chalearnlap.cvc.uab.cat/dataset/38/description/) (cloth meshes in a canonical pose)

The GarmentNets Dataset contains point clouds before and after gripping simulation with point-to-point correspondance, as well as the winding number field ($128^3$ volume).

The GarmentNets Simulation Dataset contains the raw vertecies, RGBD images and per-pixel UV from Blender simulation and rendering of CLOTH3D dataset. Each cloth instance in CLOTH3D is simulated 21 times with different random gripping points.

Both datasets are stored using [Zarr](https://zarr.readthedocs.io/en/stable/) format.

🆕 If your internet connection is not stable, please try to download part files for our datasets using command
```
$ wget --recursive --no-parent --no-host-directories --relative --reject="index.html*" https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset_parts/
```
and then combine the part files into a complete tar file for extraction:
```
$ cat garmentnets_dataset.zarr.tar.part* > garmentnets_dataset.zarr.tar
```

## Sample Datasets
Upon popular demand, we added a small subset of our datasets, which contains 25 instances of the `Tshirt` category. We are also working on alternative ways to host the complete datasets for reserchers outside of the United States.
1. [GarmentNets Dataset Sample](https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset_sample.zarr.tar.gz)

2. [GarmentNets Simulation Dataset Sample](https://garmentnets.cs.columbia.edu/dataset/garmentnets_simulation_dataset_sample.zarr.tar.gz)


## Pretrained Models
[GarmentNets Pretrained Models](https://garmentnets.cs.columbia.edu/dataset/garmentnets_checkpoints.tar)

GarmentNets are trained in 2 stages:
1. PointNet++ canoninicalization network
2. Winding number field and warp field prediction network

The checkpoints for 2 stages x 6 categories (12 in total) are all included. For evaluation, the checkpoints in the `garmentnets_checkpoints/pipeline_checkpoints` directory should be used.

## Usage
### Installation
A conda [environment.yml](./environment.yml) for `python=3.9, pytorch=1.9.0, cudatoolkit=11.1` is provided.
```
conda env create --file environment.yml
```

Alternatively, you can directly executive following commands:
```
conda install pytorch torchvision cudatoolkit=11.1 pytorch-geometric pytorch-scatter wandb pytorch-lightning igl hydra-core scipy scikit-image matplotlib zarr numcodecs tqdm dask numba -c pytorch -c nvidia -c rusty1s -c conda-forge

pip install potpourri3d==0.0.4
```

### Evaluation
Assuming the project directory is `~/dev/garmentnets`.
Assuming the [GarmentNets Dataset](https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset.zarr.tar) has been extracted to `<PROJECT_ROOT>/data/garmentnets_dataset.zarr` and [GarmentNets Pretrained Models](https://garmentnets.cs.columbia.edu/dataset/garmentnets_checkpoints.tar) has been extracted to `<PROJECT_ROOT>/data/garmentnets_checkpoints`.

Generate prediction Zarr with
```
(garmentnets)$ python predict.py datamodule.zarr_path=<PROJECT_ROOT>/data/garmentnets_dataset.zarr/Dress main.checkpoint_path=<PROJECT_ROOT>/data/garmentnets_checkpoints/pipeline_checkpoints/Dress_pipeline.ckpt
```
Note that the dataset `zarr_path` and `checkpoitn_path` must belong to the same category (`Dress` in this case).

[Hydra](https://hydra.cc/) should automatically create a run directory such as `<PROJECT_ROOT>/outputs/2021-07-31/01-43-33`. To generate evaluation metrics, execute:
```
(garmentnets)$ python eval.py main.prediction_output_dir=<PROJECT_ROOT>/outputs/2021-07-31/01-43-33
```
The `all_metrics_agg.csv` and `summary.json` should show up in the [Hydra](https://hydra.cc/) generated directory for this run.

### Training
As mentioned above, GarmentNets are trained in 2 stages. Using a single Nvidia RTX 2080Ti, training stage 1 will take roughly a week and training stage 2 can usually be done overnight.

To retrain stage 2 with a pre-trained stage 1 checkpoint:
```
(garmentnets)$ python train_pipeline.py datamodule.zarr_path=<PROJECT_ROOT>/data/garmentnets_dataset.zarr pointnet2_model.checkpoint_path=<PROJECT_ROOT>/data/garmentnets_checkpoints/pointnet2_checkpoints/Dress_pointnet2.ckpt
```

To train stage 1 from scratch:
```
(garmentnets)$ python train_pointnet2.py datamodule.zarr_path=<PROJECT_ROOT>/data/garmentnets_dataset.zarr
```

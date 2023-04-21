# GarmentNets Tracking

This is a fork of GarmentNets for the goal of extending GarmentNets' single prediction to tracking deformables. The code in this repository was written by Dylan Colli, Yating Lin, and Abhinav Kumar for the University of Michigan ROB 599, "DeepRob: Deep Learning for Robot Perception" course.

Table of Contents
- [GarmentNets Tracking](#garmentnets-tracking)
  - [Extension Thoughts](#extension-thoughts)
    - ["Simple" Post-GarmentNets Residual Prediction](#simple-post-garmentnets-residual-prediction)
  - [Notes](#notes)
    - [Installation](#installation)
      - [Python Environment](#python-environment)
      - [Data Downloads](#data-downloads)
    - [Successful Run](#successful-run)
  - [Simulation](#simulation)
- [Original GarmentNets Documentation](#original-garmentnets-documentation)
  - [Cite this work](#cite-this-work)
  - [Datasets](#datasets)
  - [Sample Datasets](#sample-datasets)
  - [Pretrained Models](#pretrained-models)
  - [Usage](#usage)
    - [Installation](#installation-1)
    - [Evaluation](#evaluation)
    - [Training](#training)

## Extension Thoughts

**Disclaimer:** I don't really know what I'm talking about here but Abhinav does so hopefully that's comforting to some extent.

### "Simple" Post-GarmentNets Residual Prediction

We're trying to extend GarmentNets to do tracking of the cloth **after** the cloth has initially been ran through the initial GarmentNets pipeline, i.e. the cloth has been picked up, the gripper rotated such that we received 4 views for the sensor to record point clouds, stitched those 4 point clouds together, and ran the pipeline.
We then wish to perform tracking on the cloth now that we have a good idea of the initial pose.

The task we're imagining to demonstrate this well is lowering the cloth onto a table (or floor, basically a flat surface).
This will introduce deformation but likely not to cause extreme deformation such that it's an impossible task to track.

Abhinav and I have talked fairly extensively on how to extend this to tracking.
We've decided that the first thing that should be tried is to try and feed the original + new information into either an MLP or PointNet++.
The information we'd feed to this portion of the model would be:
- Original data (all concatenated point-wise).
  - Point cloud points transformed to the table frame.
    - We transform to the table frame so that the model is aware of the table and deformations that may cause.
    - Note, this will cause some point `Z` values to be negative which I think is a good thing, essentially indicating that those points in the original cloud will NOT be there due to table-induced deformation.
  - Label indicating that the cloud belongs to the original data.
  - Warp field prediction (dx, dy, dz)
  - **Tentative:** Corresponding predicted NOCS coordinates for each point.
  - **Tentative:** NOCS prediction confidence.
    - This could potentially be quite helpful. Would hopefully encode a point-wise importance for warpfield prediction.
- New data (concatenated point-wise)
  - Point cloud transformed to the table frame.
  - Label indicating that the cloud belong to the new cloud.
  - Zero-initialized warp field prediction
  - **Tentative:** Zero-initialized Corresponding predicted NOCS coordinates for each point.
  - **Tentative:** Zero-initialized NOCS prediction confidence.

**Probable Problem:** I'm concerned about the presence of categorical labels here. I doubt that will play nicely. I wonder if we could use two PointNet++ encoders to encode both the original data and the new data, then use a PointNet++ decoder on the concatenation of those two encoded clouds to get a new warp field.
- We might be able to avoid this problem by using a one hot encoding. So instead of original data indicated by a 1 and new data indicated by a 0, Original would be [1, 0] and New data would be indicated by [0, 1].

For this residual prediction to work, the PointNet++ model will have to cover a decently large area of the input clouds so that the original non-table-deformed cloud areas will overlap with the new table-deformed areas.
- **Potential Shortcut** - we could probably do a proof of concept for this and only lower the garment such that ~25% of the original grasped garment was deformed due to the table. Obviously, the most difficult situation would be when the garment is lowered such that all of it is resting on the table but I don't think we need to make something that is this robust.
  - **Note on Implementation** - PointNet's grouping layer takes a distance as a parameter (usually Euclidean but can be non-Euclidean as well). That means that we can easily determine the number of layers needed for a point in the time t=t table-deformed cloud to overlap with the time t=0 original cloud. Much like a receptive field in CNNs.

**Could potentially do this with 3D Convolution by binning the warp field.** Might be too high of resolution to bin, though.


## Notes

### Installation

#### Python Environment

I had a horrible time trying to get Anaconda to work (I already don't like it and this reaffirmed my opinion), so I decided to just use my existing Python environment. That's basically one step forward and two steps back, but I just needed to get this to work.

I've added a pip requirements.txt named, "garmentnets_pip_requirements.txt" to help with package installs.
This has lots of extraneous packages but should help with figuring out which package versions are necessary to get this running.
You'll probably have to run GarmentNets multiple times to see which packages are required but not installed, then install them.

#### Data Downloads

I did the following to create the necessary directories for the datasets.
```
cd <garmentnets_tracking directory>
mkdir data
```

I did the following to download the datasets:

1. `cd <garmentnets_tracking directory>/data`
2. [GarmentNets Dataset Sample](https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset_sample.zarr.tar.gz)
    ```
    curl https://garmentnets.cs.columbia.edu/dataset/garmentnets_dataset_sample.zarr.tar.gz -o garmentnets_dataset_sample.zarr.tar.gz
    tar -xf garmentnets_dataset_sample.zarr.tar.gz
    ```
3. [GarmentNets Pretrained Models](https://garmentnets.cs.columbia.edu/dataset/garmentnets_checkpoints.tar)
    ```
    curl https://garmentnets.cs.columbia.edu/dataset/garmentnets_checkpoints.tar
    # Looks like Cheng zipped up a lot of unnecessary directories with this so we have to move the
    # directory we need to this new data directory
    tar -xf garmentnets_checkpoints.tar
    mv local/crv/cchi/data/cloth_3d_workspace/garmentnets_checkpoints/ .
    rmdir -r local
    ```

### Successful Run

I was able to do a successful run by doing the following:
1. Setup the run:
    ```
    cd <garmentnets_tracking directory>
    GARMENTNETS_ROOT=$(pwd)
    ```
2. Do the prediction(s):
    ```
    python3 predict.py datamodule.zarr_path=$GARMENTNETS_ROOT/data/garmentnets_dataset_sample.zarr/Tshirt \
        main.checkpoint_path=$GARMENTNETS_ROOT/data/garmentnets_checkpoints/pipeline_checkpoints/Tshirt_pipeline.ckpt
    ```
3. Find the output directory. Hydra creates an output directory based on the time that the run occurred so this will be different for each run.
    ```
    ls outputs  # Then just find the directory most recently created
    ```
4. Do the evaluation:
    ```
    # Substitute the directory with whatever directory that was created for you.
    python3 eval.py main.prediction_output_dir=$GARMENTNETS_ROOT/outputs/2023-04-01/23-09-36
    ```

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

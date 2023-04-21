# Simulation Pipeline <!-- omit from toc -->

Cheng did the simulation for this paper in Blender as the CLOTH3D dataset is setup for Blender simulation already. He shared his repository for simulation with me and I ported over/rewrote many of the files as I sorted through what is necessary for the dynamics simulation.


## Table of Contents <!-- omit from toc -->
- [Installation](#installation)
  - [Installation As Typical GUI](#installation-as-typical-gui)
  - [Blender Python API Installation](#blender-python-api-installation)
- [Usage](#usage)
  - [Cloth Simulation](#cloth-simulation)
    - [Original GarmentNets Simulation](#original-garmentnets-simulation)
    - [Freespace Dynamics Simulation](#freespace-dynamics-simulation)
  - [Rendering](#rendering)
  - [Packinging Simulation Into A Zarr](#packinging-simulation-into-a-zarr)
  - [Tangential Note On Unused Simulation Routines](#tangential-note-on-unused-simulation-routines)
- [Datasets](#datasets)
  - [CLOTH3D](#cloth3d)
    - [Accessing in Code](#accessing-in-code)
  - [Provided Sample GarmentNets Dataset](#provided-sample-garmentnets-dataset)

## Installation

Installation of Blender for the purposes of simulation was tricky in our case. The original code was written for Blender version 2.X (likely 2.8 or 2.9), but I don't know which one exactly. Since Blender comes pre-packaged with its own full Python distribution, one typically scripts from within Blender's built-in editor in the GUI if attempting to use simple scripts. However, we're using the API to do advanced cloth simulation so it was important to be able to rapidly prototype the Blender code using Jupyter Notebooks. To make things more complicated, to make Blender's Python API accessible to your Python environment, you typically have to compile Blender on your machine, pointing Blender to the Python distribution you would like to compile the API for.

Instead of going through all of this, the Blender foundation has released Blender's API, `bpy`, as a pip installable package (thankfully). Yet this isn't totally straightforward as the Blender version is tied to specific Python versions. To fix this, I followed the instructions in [this section](#blender-python-api-installation). Additionally, I wanted to be able to prototype actions in the GUI so I downloaded Blender through the typical fashion, detailed in [this section below](#installation-as-typical-gui).

### Installation As Typical GUI

For normal installation of Blender, I installed Blender by downloading the [version 3.5 package](https://www.blender.org/download/) and extracting to my home directory.

Blender is then launched with `~/blender/blender`.

### Blender Python API Installation

I downloaded Python3.10, created a `pipenv` Python virtual environment (stored as Pipfile in this directory), then started working on the code.

To recreate the `pipenv` environment:
- [Install Python3.10](https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux-system/)
- [Install pipenv](https://pypi.org/project/pipenv/) with `pip install pipenv`
- Recreate the environment with:
  ```
  cd <garmentnets_root_directory>/simulation
  pipenv install
  ```
- Then to activate the `pipenv` environment, use `pipenv shell` in the `garmentnets/simulation` directory.

Among the packages installed in the `pipenv` environment, these are the packages necessary to use Blender as a Python module:
- `bpy`
  - Installs Blender as a Python module. My understanding is that this is a *separate* Blender installation than anything on your system.
- `fake-bpy-module-latest`
  - Installs "fake" Blender Python modules (I'd probably calls these "stubs") so that VS Code intellisense works.

I have Python3.8 installed for my system Python but Blender 3.5, 3.4 uses Python 3.10 and it seems pretty impossible to get it to use anything else. Hence, the installation of Python 3.10 for use with simulation.

Note: The script `print_blender_python_info.sh` is an artifact of me trying to run the simulation code using the typical Blender installation method. This wasn't used in the generation of the simulation dataset but it includes some potentially helpful information for running Blender via command line if it's necessary at another point in time.

## Usage

The simulation pipeline is broken into several stages:
1. Cloth Simulation - further broken down into:
   1. Original GarmentNets simulation of grasped state.
   2. Dynamics simulation.
2. Rendering
3. Packaging Into A Zarr Directory Tree

### Cloth Simulation

This portion of the pipeline *only simulates* the mesh deformation induced by gripper control. This does not render the images that are necessary for obtaining partial point clouds that are used in training/test! That is described [in this section](#rendering).

#### Original GarmentNets Simulation

As GarmentNets was trained on data simulated using garment meshes generated as part of the CLOTH3D dataset, it is necessary to [download the CLOTH3D dataset](#cloth3d).

The simulation utilized in the original GarmentNets publication involved grasping a random vertex of the CLOTH3D garment mesh, then "picking the garment up" to simulate the resting state of the grasped garment.

This is then used as the resting state of the 5 trajectories that are generated for each Tshirt in the dataset.

This portion of the pipeline is ran as part of the following [Freespace Dynamics Simulation](#freespace-dynamics-simulation).

#### Freespace Dynamics Simulation

To generate training data for the simulation, we simulated the resting state for the 25 Tshirts included as part of the provided [sample GarmentNets dataset](#provided-sample-garmentnets-dataset).

To run the full GarmentNets Tracking simulation:

1. Launch the `<repo_root>/simulation/runners/freespace_sim_driver_prep.ipynb`. You'll want to select the `simulation` Python 3.10 kernel that was created as part of the `pipenv` simulation environment installation process detailed above.
2. Execute the full notebook. This provides a more interactive method of preparing the dynamics simulation over a traditional, monolithic Python script.
3. Run the full freespace simulation on the run that was prepared in step #1 with:
  ```bash
  # Activate the pipenv virtual environment
  cd <repo_root>/simulation
  pipenv shell
  cd <repo_root>/simulation/runners
  python freespace_sim_driver.py
  ```

### Rendering

Now that the sequence of mesh deformations induced by gripper control are generated, we can render the images necessary to generate the partial point clouds observed from our "virtual sensor".

The `<repo_root>/simulation/runners/dynamics_renderer.py` script is setup with proper command line argument parsing that allows the specification of what you'd specifically like to render. For example, you can choose to render only: depth/rgb, resting state/trajectories, and specific viewpoints. Execute `python dynamics_renderer.py -h` if you'd like to see the full list of command line arguments.

To run the full rendering pipeline:
```bash
# Activate pipenv virtual environment.
cd <repo_root>/simulation
pipenv shell
cd <repo_root>/simulation/runners
python dynamics_renderer.py
```

A helpful tip is that this can be parallelized. For example, you can launch one process to render the depth and another to render the RGB images. To do this:

```
# Activate pipenv virtual environment as above.
python dynamics_renderer.py --no-render-rgb
# Open another terminal and active pipenv environment.
python dynamics_renderer.py --no-render-depth
```

Given that you have enough GPU compute, this effectively halves the amount of time required to render the dataset.

### Packinging Simulation Into A Zarr

Now that all simulation data is generated and rendered, the data must be packaged into a [Zarr](https://zarr.readthedocs.io/en/stable/) directory tree. Zarr provides a convenient way to store compressed N-dimensional arrays (much like HDF5) for easy access during training and testing. We chose to use Zarr as the original GarmentNets publication utilized the package for the training pipeline, streamlining integration of our tracking algorithmic extension.

To extract the data from simulation and package it into a Zarr:
1. Extract dynamics simulation results:
   1. Launch the extraction Ipython notebook, `<repo_root>/data_packager/extract_dynamics_sim_results.ipynb`
   2. Select the `simulation` `pipenv` kernel.
   3. Execute the full notebook.
2. Package all simulation data into a Zarr:
   1. Activate the **conda environment** included in this repository. Due to package version issues with Python 3.10, this portion of the code does not use the `simulation` `pipenv` environment. As we had version issues with this, this portion of the code was actually developed using Dylan's system Python installation (Python 3.8), not a virtual environment. All packages necessary to run this script *should* be in the `conda` environment but if it is missing a package, such as `pyexr`, the installation of the few missing packages should be straightforward.
   2. Execute the Zarr packager script:
      ```bash
      cd <repo_root>/simulation/data_packager
      python pack_data_to_zarr.py
      ```
      This takes a surprising amount of time due to the sheer amount of file I/O necessary to generate point clouds for all viewpoints, of all timesteps, of all trajectories, of all Tshirts.

Assuming this and all other portions of the simulation code completed successfully, you're ready to start training!

### Tangential Note On Unused Simulation Routines

We initially targeted tracking of cloth state as the garment was lowered onto a table. As this is a *very* difficult problem, we decided to pursue tracking of trajectories executed entirely in freespace. However, the code to do the simulation was still developed for the most part and can be found at: `<repo_root>/simulation/runners/lowering_onto_table_sim_driver.ipynb`.


## Datasets

### CLOTH3D

The CLOTH3D dataset provides the meshes necessary to run simulation which is then rendered to get simulated point clouds that we could expect to receive from a sensor (e.g. RGBD) and finally input into GarmentNets and our extension.

#### Accessing in Code

Cheng wrote utiltity functions for reading the CLOTH3D dataset in `cloth_3d_util`.

- The function for reading the data is in the `cloth_3d_util.accessor::Cloth3DCanonicalAccessor` class.

### Provided Sample GarmentNets Dataset

As part of the publication, the authors provided a sample dataset of 25 Tshirts. You can find [a link to download this in the root README](../README.md#sample-datasets).

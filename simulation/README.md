# Simulation Pipeline <!-- omit from toc -->

Cheng did the simulation for this paper in Blender as the CLOTH3D dataset is setup for Blender simulation already. He shared his repository for simulation with me and I'll be moving the files here as I sort through what is necessary for simulation as there's lots of code (and redundant versions of code) in the repo.


## Table of Contents <!-- omit from toc -->
- [Blender Installation](#blender-installation)
  - [Installation As Typical GUI](#installation-as-typical-gui)
  - [Blender Python API Installation](#blender-python-api-installation)
- [Datasets](#datasets)
  - [CLOTH3D](#cloth3d)
    - [Accessing in Code](#accessing-in-code)

## Blender Installation

Installation of Blender for the purposes of simulation was tricky in our case. The original code was written for Blender version 2.X (likely 2.8 or 2.9), but I don't know which one exactly. Since Blender comes pre-packaged with its own full Python distribution, one typically scripts from within Blender's built-in editor in the GUI if attempting to use simple scripts. However, we're using the API to do advanced cloth simulation so it was important to be able to rapidly prototype the Blender code using Jupyter Notebooks. To make things more complicated, to make Blender's Python API accessible to your Python environment, you typically have to compile Blender on your machine, pointing Blender to the Python distribution you would like to compile the API for.

Instead of going through all of this, the Blender foundation has released Blender's API, `bpy`, as a pip installable package (thankfully). Yet this isn't totally straightforward as the Blender version is tied to specific Python versions. To fix this, I followed the instructions in [this section](#blender-python-api-installation). Additionally, I wanted to be able to prototype actions in the GUI so I downloaded Blender through the typical fashion, detailed in [this section below](#installation-as-typical-gui).

### Installation As Typical GUI

For normal installation of Blender, I installed Blender by downloading the [version 3.5 package](https://www.blender.org/download/) and extracting to my home directory.

Blender is then launched with `~/blender/blender`.

### Blender Python API Installation

I downloaded Python3.10, created a `pipenv` Python virtual environment (stored as Pipfile in this directory), then started working on the code.

To recreate the `pipenv` environment:
- [install Python3.10](https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux-system/)
- [install pipenv](https://pypi.org/project/pipenv/) with `pip install pipenv`
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

I have Python3.8 installed for my system Python but Blender 3.5, 3.4 uses Python 3.10 and it seems pretty impossible to get it to use anything else.

The script `print_blender_python_info.sh` is an artifact of me trying to run the simulation code using the typical Blender installation method. This wasn't used in the generation of the simulation dataset but it includes some potentially helpful information for running Blender via command line if it's necessary at another point in time.

## Datasets

### CLOTH3D

The CLOTH3D dataset provides the meshes necessary to run simulation which is then rendered to get simulated point clouds that we could expect to receive from a sensor (e.g. RGBD) and finally input into GarmentNets and our extension.

#### Accessing in Code

Cheng wrote utiltity functions for reading the CLOTH3D dataset in `cloth_3d_util`.

- The function for reading the data is in the `cloth_3d_util.accessor::Cloth3DCanonicalAccessor` class.


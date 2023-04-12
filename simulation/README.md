# Simulation Pipeline <!-- omit from toc -->

Cheng did the simulation for this paper in Blender as the CLOTH3D dataset is setup for this simulation already.
He shared his repository for simulation with me and I'll be moving here as I sort through what is necessary for simulation as there's lots of code in the repo.


## Table of Contents <!-- omit from toc -->
- [Blender Installation](#blender-installation)
  - [Python Version Issues](#python-version-issues)
- [Accessing The CLOTH3D Dataset](#accessing-the-cloth3d-dataset)

## Blender Installation

I downloaded Python3.10, created a `pipenv` Python virtual environment (stored as Pipfile in this directory), then started working on the code. In terms of actual Blender Python installation, I downloaded the following via pip:
- `bpy`
  - Installs Blender as a Python module. My understanding is that this is a *separate* Blender installation than anything on your system.
- `fake-bpy-module-latest`
  - Installs "fake" Blender Python modules (I'd probably calls these "stubs") so that VS Code intellisense works.

For normal installation, I *separately* installed Blender by downloading the [version 3.5 package](https://www.blender.org/download/) and extracting to my home directory.

### Python Version Issues

I have Python3.8 installed for my system Python but Blender 3.5, 3.4 uses Python 3.10 and it seems pretty impossible to get it to use anything else.
I ultimately installed Python3.10 and used the pip-installable bpy library which installs Blender as a Python library.

The script `print_blender_python_info.sh` is an artifact of me debugging this but it includes some potentially helpful information for running Blender via command line.

## Accessing The CLOTH3D Dataset

Cheng wrote utiltity functions for reading the CLOTH3D dataset in `cloth_3d_util`.

- The function for reading the data is in the `cloth_3d_util.accessor::Cloth3DCanonicalAccessor` class.


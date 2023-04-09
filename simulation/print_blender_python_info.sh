#! /bin/bash

BLENDER_BIN_PATH='/home/dcolli23/blender/blender'

# -b makes blender run in the background (no GUI)
# -o <path> tells blender where to save the renders.
# -t <threads> sets the number of threads for blender.
# -P <filepath> runs the given Python script file.
# --python-use-system-env tells Blender to use system environment variables for Python such as
#                        'PYTHONPATH' and the user site-packages directory.
# --debug-python enables debug messages for Python
$BLENDER_BIN_PATH -b -P ./blender_python_info_printer.py

$BLENDER_BIN_PATH -b --python-use-system-env -P ./blender_python_info_printer.py
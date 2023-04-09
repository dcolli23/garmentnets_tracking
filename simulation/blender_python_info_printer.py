import sys

line_sep = 80 * '-' + '\n'
print(line_sep, "Blender Python Info Printer", line_sep)
print("Python version:", sys.version)
print("Python version info:", sys.version_info)
print("Available modules:", help('modules'))
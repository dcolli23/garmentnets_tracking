# %%
# parse input
import sys
import json

argv = sys.argv
argv = argv[argv.index('--') + 1:]
assert(len(argv) > 0)
json_path = argv[0]
print(json_path)
task_packet = json.load(open(json_path, 'r'))

# %%
def _setup():
    sample_config = {
        'sample_id': '00001',
        'garment_name': 'Skirt',
        'grip_vertex_idx': 0
    }

    task_packet = {
        'sample_configs': [sample_config],
        'output_root': '/home/cchi/data/cloth_3d_grip/cloth_3d_sim',
        'cloth_3d_path': '/home/cchi/data/CLOTH3D/train',
        'import_dirs': ['/home/cchi/miniconda3/envs/blender/lib/python3.7', '/home/cchi/miniconda3/envs/blender/lib/python3.7/lib-dynload', '/home/cchi/miniconda3/envs/blender/lib/python3.7/site-packages'],
        'proj_dir': "~/dev/blender_experiment"
    }

# %%
# Change CWD
import os
import sys
proj_dir = task_packet['proj_dir']
import_dirs = task_packet['import_dirs']
os.chdir(os.path.expanduser(proj_dir))
sys.path.append(os.path.expanduser(proj_dir))
sys.path.extend(import_dirs)

# %%
import pathlib
import pickle
from tqdm import tqdm

from cloth_3d_util.accessor import Cloth3DCanonicalAccessor
from pipeline.smpl_simulation_pipeline import smpl_simulation_pipeline

# %%
# global setup
accessor = Cloth3DCanonicalAccessor(task_packet['cloth_3d_path'])
sample_configs = task_packet['sample_configs']
output_root = pathlib.Path(task_packet['output_root'])
assert(output_root.is_dir())


for sample_config in tqdm(task_packet['sample_configs']):
    # %% setup task
    sample_key = "_".join([str(sample_config[x]) for x in
        ['sample_id', 'garment_name', 'grip_vertex_idx']])
    sample_dir = output_root.joinpath(sample_key)
    sample_dir.mkdir(exist_ok=True)
    result_file = sample_dir.joinpath('simulation_result.pk')
    if result_file.exists():
        print("{} already exists. Skip!".format(result_file))
        continue

    sample_data = accessor.get_sample_data(**sample_config)
    garment_info = accessor.reader.read_info(sample_config['sample_id'])

    args_dict = dict()
    args_dict.update(sample_config)
    args_dict.update(sample_data)
    args_dict['simulation_duration_pair'] = (0, 120)

    # %%
    # run
    result_data = smpl_simulation_pipeline(**args_dict)

    # %%
    print('writing to {}'.format(result_file))
    pickle.dump(result_data, result_file.open('wb'))
    print('Done!')


# %%
print('All Done!')

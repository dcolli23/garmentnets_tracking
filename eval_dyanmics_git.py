

import torch
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
import yaml
import os
import pathlib
import zarr
from dynamics_helper import chamfer_loss, DynamicsMLP, match_points
from torch_geometric.data import Data
from common.cache import file_attr_cache
from datasets.pointnet_dynamics_dataset import _get_groups_df
import numpy as np
from tqdm import tqdm

from common.gripper_util import get_gripper_locs, translate_cloud_to_origin, \
    translate_gripper_to_world_frame, GRIPPER_POS_INIT_ARR

FILE_ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(FILE_ROOT, "out")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rs = np.random.RandomState(seed=None)
input_dim = 128 +3
hidden_dim = 512
output_dim = 3

dynamics_mlp = DynamicsMLP(input_dim, hidden_dim, output_dim).to(device)
dynamics_mlp.load_state_dict(torch.load('models/dynamics_mlp.pt'))

with open('config/dynamics_test.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

checkpoint_path = os.path.expanduser(cfg['main']['checkpoint_path'])
model_cpu = ConvImplicitWNFPipeline.load_from_checkpoint(checkpoint_path)
device = torch.device('cuda:{}'.format(cfg['main']['gpu_id']))
model = model_cpu.to(device)
model.eval()
model.requires_grad_(False)

# Load test data
#-----------
path = pathlib.Path(os.path.expanduser(cfg['datamodule']['zarr_path']))
root = zarr.open(str(path.absolute()), mode='r')
samples_group = root['samples']
groups_df = file_attr_cache(cfg['datamodule']['zarr_path'], 
    cache_dir=cfg['datamodule']['metadata_cache_dir'])(_get_groups_df)(samples_group)
# check if index is sorted
assert(groups_df.index.is_monotonic_increasing)
groups_df['idx'] = np.arange(len(groups_df))

#-----------
with torch.no_grad():
    errors = []
    for shirts in tqdm(range(len(samples_group))):
        for traj_ind in tqdm(range(5)):
            chamfer_losses = [] 

            for t in tqdm(range(74)):
                row = groups_df.iloc[shirts]
                group = samples_group[row.group_key]

                # Get gripper location/delta data.
                # Need to add a zero delta in front so that our gripper location array starts at the same 
                # timestep as the dynamics simulation
                gripper_zero_delta = np.array((0, 0, 0)).reshape(1, 3)
                gripper_deltas = np.concatenate((
                    gripper_zero_delta, 
                    group['dynamics'][traj_ind]['delta_gripper_pos'][:]
                ), axis=0)
                # gripper_deltas = group['dynamics'][traj_ind]['delta_gripper_pos'][:]
                grip_pos_cumulative = np.cumsum(gripper_deltas, axis=0)
                grip_locs = torch.tensor(get_gripper_locs(grip_pos_cumulative), device=device, dtype=torch.float32)
                
                if t == 0:

                    point_cloud_group = group['point_cloud']
                    ## TO DO: downsampling point_cloud_data
                    all_idxs_next = np.arange(len(point_cloud_group['point']))
                    selected_idxs_next = rs.choice(all_idxs_next, size=6000, replace=False)
                    # test = point_cloud_group['point'][:]
                    pos = point_cloud_group['point'][:][selected_idxs_next,:].astype(np.float32)
                    # print('pos z min/max', pos[:, 2].min(), pos[:, 2].max())
                    x = point_cloud_group['rgb'][:][selected_idxs_next].astype(np.float32)
                    
                ######################
                    pos = torch.tensor(pos).to(device)
                    # print('pos z min/max', pos[:, 2].min(), pos[:, 2].max())
                    x = torch.tensor(x).to(device) / 255
                
                    # Translate the point clouds back to the origin to make clouds in distribution that pointnet++ is expecting.
                    # action_np = group['dynamics'][traj_ind]['delta_gripper_pos'][t][:]

                point_cloud_data = Data(pos=pos, x=x, batch=torch.zeros(pos.shape[0],dtype=int).to(device))

                # print('pos z min/max', pos[:, 2].min(), pos[:, 2].max())

                #Handle state invariance for features
                #This setup could have batching problems if we don't downsample the point clouds.
                #However, if we do downsample the point clouds,there could be problems with the proper number of correspondences.
                #Quick fix: Downsample the full point cloud to 6000 points. Downsample the partial view to 1500 points. Match.
                action_np = gripper_deltas[t, :]
                action = torch.tensor(action_np).to(device).float()
                pointnet_features = model.pointnet2_forward(point_cloud_data)
                features = pointnet_features['per_point_features']
                
                state_action = torch.cat((features, action.view(1, -1).repeat(6000,1)), dim=1).float()
                
                # Have to translate the prediction back to the global frame as the `pos` cloud is
                # in the gripper frame so as to remain in distribution for PointNet++
                # pred_gripper_frame = action + pos # rigid body translation
                pred_gripper_frame = dynamics_mlp(state_action) + pos # rigid body translation
                print("Local frame max position:", torch.max(pred_gripper_frame,dim=0)[0],"Local frame min position",torch.min(pred_gripper_frame,dim=0)[0])
                
                pred_world_frame = translate_gripper_to_world_frame(pred_gripper_frame, grip_locs, t).float()
                

                #Downsampling for partial view point cloud.
                partial_view_pos = group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}']['view_0']['point'][:]
                partial_view_rgb = group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}']['view_0']['rgb'][:]
                all_idxs_next = np.arange(len(partial_view_pos))
                selected_idxs_next = rs.choice(all_idxs_next, size=1500, replace=False)
                partial_view_pos = torch.tensor(partial_view_pos[selected_idxs_next,:]).to(device).float()
                partial_view_rgb =  torch.tensor(partial_view_rgb[selected_idxs_next]).to(device).float() / 255

                # If we want to do matching, uncomment.
                # partial_view_pos = translate_cloud_to_origin(partial_view_pos, grip_locs, t)
                # pos, x = match_points(pred_gripper_frame, partial_view_pos, x, partial_view_rgb)

                
                # When we move to using `match_points`, the partial view point cloud positions will 
                # need to be translated to the gripper frame by using `translate_cloud_to_origin`
                pos, x  = pred_gripper_frame, x

                #TO DO: Run point_cloud through GarmentNets to get predictions 
                #-----------

                #-----------
                # True full view pc to compare loss
    


                full_view_gt_pc= np.concatenate([group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}'][f'view_{i}']['point'][:] for i in range(4)], axis=0)
                
                all_idxs_next = np.arange(len(full_view_gt_pc))
                selected_idxs_next = rs.choice(all_idxs_next, size=6000, replace=False)
                full_view_gt_pc = torch.tensor(full_view_gt_pc[selected_idxs_next,:]).to(device).float()
                
                chamfer_losses.append(chamfer_loss(pred_world_frame, full_view_gt_pc).item())
            print(chamfer_losses)
            errors.append(chamfer_losses)
            all_errors_np = np.array(errors)
            np.save(os.path.join(OUTPUT_DIR, "most_recent_eval_errors.npy"), all_errors_np)

    all_errors_np = np.array(errors)
    

    print(all_errors_np.mean(0))
    print(all_errors_np.std(0))


 
# %%
# import
import os
import pathlib
from matplotlib import pyplot as plt

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from dynamics_helper import chamfer_loss, DynamicsMLP, match_points,match_points_chamfer
import pandas as pd
import numpy as np
import scipy.ndimage as ni
from skimage.measure import marching_cubes
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import plotly.graph_objects as go
from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from datasets.pointnet_dynamics_dataset import _get_groups_df
from networks.pointnet2_nocs import PointNet2NOCS
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from components.gridding import VirtualGrid, ArraySlicer
from common.torch_util import to_numpy
from common.cache import file_attr_cache
from common.geometry_util import AABBGripNormalizer
from plot_file.plot import plot_pointcloud,plot_volume,plot_mesh,plot_pointclouds
from torch_geometric.data import Data

from common.gripper_util import get_gripper_locs, translate_cloud_to_origin, \
    translate_gripper_to_world_frame, GRIPPER_POS_INIT_ARR


# %%
# main script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rs = np.random.RandomState(seed=None)
# GRIPPER_POS_INIT_ARR = np.array((0.0, 0.0, 0.4)).reshape(1, 3).float()
def load_garnet_model(cfg):

    checkpoint_path = os.path.expanduser(cfg['main']['checkpoint_path'])
    model_cpu = ConvImplicitWNFPipeline.load_from_checkpoint(checkpoint_path)
    model = model_cpu.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

def load_dynamics_model(cfg):
    action_dim = 3
    feature_dim = 128
    input_dim = action_dim + feature_dim
    hidden_dim = 512
    output_dim = 3
    dynamics_mlp = DynamicsMLP(input_dim, hidden_dim, output_dim).to(device)

    dynamics_mlp_path = os.path.expanduser(cfg['main']['dynamics_mlp_path'])

    dynamics_mlp.load_state_dict(torch.load(dynamics_mlp_path))
    
    return dynamics_mlp

def load_test_data(cfg):

    path = pathlib.Path(os.path.expanduser(cfg['datamodule']['zarr_path']))
    root = zarr.open(str(path.absolute()), mode='r')
    samples_group = root['samples']
    groups_df = file_attr_cache(cfg['datamodule']['zarr_path'], 
        cache_dir=cfg['datamodule']['metadata_cache_dir'])(_get_groups_df)(samples_group)
    # check if index is sorted
    assert(groups_df.index.is_monotonic_increasing)
    groups_df['idx'] = np.arange(len(groups_df))
    return samples_group,groups_df

def garnet_predict(cfg, model, data):
    
    batch = data.to(device=device)
    # print(batch.batch.shape)
    # stage 1/1.5
    pointnet2_result = model.pointnet2_forward(batch)       
    nocs_data = pointnet2_result['nocs_data']

    #spare nocs space point_cloud
 


    unet3d_result = model.unet3d_forward(pointnet2_result)
    
    
    # stage 2 generate volume
    vg = VirtualGrid(grid_shape=(cfg.prediction.volume_size,)*3)
    grid_points = vg.get_grid_points(include_batch=False)
    array_slicer = ArraySlicer(grid_points.shape, (64,64,64))
    result_volume = torch.zeros(grid_points.shape[:-1], dtype=torch.float32, device=device)

    for i in range(len(array_slicer)):
        slices = array_slicer[i]
        query_points = grid_points[slices]
        query_points_gpu = query_points.to(device).view(1,-1,3)
        decoder_result = model.volume_decoder_forward(unet3d_result, query_points_gpu)
        pred_volume_value = decoder_result['pred_volume_value'].view(*query_points.shape[:-1])
        result_volume[slices] = pred_volume_value
    pred_volume = result_volume
    wnf_volume = to_numpy(pred_volume)

    
  

    # stage 2.5 marching cubes
    volume_size = wnf_volume.shape[-1]
    wnf_ggm = ni.gaussian_gradient_magnitude(
        wnf_volume, sigma=cfg.prediction.gradient_sigma, mode="nearest")
    voxel_spacing = 1 / (volume_size - 1)
    mc_verts = np.ones((1,3), dtype=np.float32) * np.nan
    mc_faces = np.zeros((1,3), dtype=np.int64)
    mc_normals =np.ones((1,3), dtype=np.float32) * np.nan
    mc_values = np.ones((1,), dtype=np.float32) * np.nan
    mc_verts_ggm = np.ones((1,), dtype=np.float32) * np.nan
    mc_warp_field = np.ones((1,3), dtype=np.float32) * np.nan
    try:
        mc_verts, mc_faces, mc_normals, mc_values = marching_cubes(
            wnf_volume, 
            level=cfg.prediction.iso_surface_level, 
            spacing=(voxel_spacing,)*3, 
            gradient_direction=cfg.prediction.gradient_direction,
            method='lewiner')
        
        mc_verts_nn_idx = (mc_verts / voxel_spacing).astype(np.uint32)
        mc_verts_ggm = wnf_ggm[
            mc_verts_nn_idx[:,0], mc_verts_nn_idx[:,1], mc_verts_nn_idx[:,2]]
        
        # stage 3
        surface_query_points = torch.from_numpy(mc_verts.astype(np.float32)).view(1,-1,3).to(device)
        surface_decoder_result = model.surface_decoder_forward(
            unet3d_result, surface_query_points)
        mc_warp_field = to_numpy(surface_decoder_result['out_features'].view(-1, 3))
    except ValueError as e:
        pass

    # write data to disk
    mc_data = {
        'verts': mc_verts.astype(np.float32),
        'faces': mc_faces.astype(np.int32),
        'normals': mc_normals.astype(np.float32),
        'volume_value': mc_values.astype(np.float32),
        'volume_gradient_magnitude': mc_verts_ggm.astype(np.float32),
        'warp_field': mc_warp_field.astype(np.float32)
    }

       
   
    return mc_data

def predic_mesh_and_plot(fig,cfg,garnet_model,pc_position_local_frame,rgb,grip_locs,t):

    point_cloud_data = Data(pos=pc_position_local_frame, x=rgb, batch=torch.zeros(pc_position_local_frame.shape[0],dtype=int).to(device))
    mc_data = garnet_predict(cfg,garnet_model,point_cloud_data)  
   
    mc_pos = to_numpy(translate_gripper_to_world_frame(torch.tensor(mc_data['warp_field'],device=device), grip_locs, t).float())
    plot_mesh(mc_pos,mc_data['faces'],fig=fig)     
    return fig
# def downsampling_PC():

@hydra.main(config_path="config", 
    config_name="predict_func_default")
def main(cfg: DictConfig) -> None:

    np.random.seed(0)
    DYNAMICS = 'rigid_body' # rigid_body, mlp
    MATCHING = False
    PLOT = True


    # load module to gpu
    garnet_model = load_garnet_model(cfg)

    dynamics_model = load_dynamics_model(cfg)

    # Load test data
    samples_group,groups_df = load_test_data(cfg)
     
    with torch.no_grad():
        errors = []
        for shirts in tqdm(range(len(samples_group))):
            for traj_ind in tqdm(range(5)):
                chamfer_losses = [] 
                for t in tqdm(range(74)):
                    row = groups_df.iloc[shirts]
                    group = samples_group[row.group_key]

                    gripper_zero_delta = np.array((0, 0, 0)).reshape(1, 3)
                    gripper_deltas = np.concatenate((
                        gripper_zero_delta, 
                        group['dynamics'][traj_ind]['delta_gripper_pos'][:]
                    ), axis=0)
                     
                    grip_pos_cumulative = np.cumsum(gripper_deltas, axis=0)
                    grip_locs = torch.tensor(get_gripper_locs(grip_pos_cumulative), device=device, dtype=torch.float32)
                
                    if t == 0:
                        point_cloud_group = group['point_cloud']
                        all_idxs_next = np.arange(len(point_cloud_group['point']))
                        selected_idxs_next = rs.choice(all_idxs_next, size=6000, replace=False)
                        pos = point_cloud_group['point'][:][selected_idxs_next,:].astype(np.float32) + GRIPPER_POS_INIT_ARR
                        x = point_cloud_group['rgb'][:][selected_idxs_next].astype(np.float32)
                            
                    ######################
                        pred_world_frame = torch.tensor(pos).to(device).float()
                        x = torch.tensor(x).to(device) / 255
                        print("World frame max position:", torch.max(pred_world_frame,dim=0)[0],"World frame min position",torch.min(pred_world_frame,dim=0)[0])
                    
                

                    if t==0 and PLOT:
                        fig1 = go.Figure()
                        print("Cumulative grip movement: ",grip_pos_cumulative[-1,:])
                        pred_local_frame = translate_cloud_to_origin(pred_world_frame,grip_locs,t)
                        predic_mesh_and_plot(fig1,cfg,garnet_model,pred_local_frame,x,grip_locs,t).show()
                        
                        
                        

              

                    action_np = gripper_deltas[t, :]
                    action = torch.tensor(action_np).to(device).float()

                    if DYNAMICS == 'rigid_body':
                         
                        pred_world_frame += action  # rigid body translation

                        
                    elif DYNAMICS == 'mlp':
                        pos_local_frame = translate_cloud_to_origin(pred_world_frame, grip_locs, t)
                        # print("Local frame max position:", torch.max(pos_local_frame,dim=0)[0],"Local frame min position",torch.min(pos_local_frame,dim=0)[0])
                        
                        point_cloud_data = Data(pos=pos_local_frame, x=x, batch=torch.zeros(pos_local_frame.shape[0],dtype=int).to(device))

                        pointnet_features = garnet_model.pointnet2_forward(point_cloud_data)
                        features = pointnet_features['per_point_features']
                        state_action = torch.cat((features, action.view(1, -1).repeat(6000,1)), dim=1).float()
                        pred_world_frame += dynamics_model(state_action)  


                    partial_view_pos = group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}']['view_0']['point'][:]
                    partial_view_rgb = group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}']['view_0']['rgb'][:]
                    all_idxs_next = np.arange(len(partial_view_pos))
                    selected_idxs_next = rs.choice(all_idxs_next, size=1500, replace=False)
                    partial_view_pos = torch.tensor(partial_view_pos[selected_idxs_next,:]).to(device).float()
                    partial_view_rgb =  torch.tensor(partial_view_rgb[selected_idxs_next]).to(device).float() / 255

                    # # If we want to do matching, uncomment.
                    if MATCHING:
                        pred_pos_local_frame = translate_cloud_to_origin(pred_world_frame, grip_locs, t+1)

                        partial_view_pos_local_frame = translate_cloud_to_origin(partial_view_pos, grip_locs, t+1)
                        pred_pos_local_frame, x = match_points_chamfer(pred_pos_local_frame, partial_view_pos_local_frame, x, partial_view_rgb)
                        pred_world_frame = translate_gripper_to_world_frame(pred_pos_local_frame, grip_locs, t+1).float()
                    # else:
                    #     pos = pred_gripper_frame.clone()
                    #     x = x.clone()

                    
                    # Just for evaluation
                    full_view_gt_pc= np.concatenate([group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}'][f'view_{i}']['point'][:] for i in range(4)], axis=0)
                    full_view_gt_rgb= np.concatenate([group['dynamics'][traj_ind]['point_cloud'][f'timestep_{t+1}'][f'view_{i}']['rgb'][:] for i in range(4)], axis=0)
                    all_idxs_next = np.arange(len(full_view_gt_pc))
                    selected_idxs_next = rs.choice(all_idxs_next, size=6000, replace=False)
                    full_view_gt_pc = torch.tensor(full_view_gt_pc[selected_idxs_next,:]).to(device).float()
                    full_view_gt_rgb =  torch.tensor(full_view_gt_rgb[selected_idxs_next]).to(device).float() / 255
                    
                    chamfer_losses.append(chamfer_loss(pred_world_frame, full_view_gt_pc).item())
                    
                    if t == 35 and PLOT:
                        fig2 = go.Figure()
                        pred_local_frame = translate_cloud_to_origin(pred_world_frame, grip_locs, t+1)
                        predic_mesh_and_plot(fig2,cfg,garnet_model,pred_local_frame,x,grip_locs,t+1).show()
                        
                        fig3 = go.Figure()
                        full_view_gt_pc_local = translate_cloud_to_origin(full_view_gt_pc, grip_locs, t+1)
                        predic_mesh_and_plot(fig3,cfg,garnet_model,full_view_gt_pc_local,full_view_gt_rgb,grip_locs,t).show()
                        # plot_pointcloud(full_view_gt_pc,fig=fig)
                        # plot_pointclouds([pos_0,to_numpy(full_view_gt_pc)-np.array([0.,0.,0.4])],fig=fig)
                    if t == 73:
                        print("World frame max position:", torch.max(pred_world_frame,dim=0)[0],"World frame min position",torch.min(pred_world_frame,dim=0)[0])
                        pred_local_frame = translate_cloud_to_origin(pred_world_frame, grip_locs, t+1)
                        
                        print("Local frame max position:", torch.max(pred_local_frame,dim=0)[0],"Local frame min position",torch.min(pred_local_frame,dim=0)[0])





                    
                # print(pos.min(), pos.max())
                print(chamfer_losses)
                errors.append(chamfer_losses)
                all_errors_np = np.array(errors)
              
                np.save(f'../../../out/{DYNAMICS}_match_{MATCHING}.npy', all_errors_np)
        all_errors_np = np.array(errors)
    

        print(all_errors_np.mean(0))
        print(all_errors_np.std(0))


if __name__ == "__main__":
    main()

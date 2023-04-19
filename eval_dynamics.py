import torch
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
import yaml
import os
import pathlib
import zarr
from dynamics_helper import chamfer_loss, DynamicsMLP, match_points
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 128
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

#-----------

chamfer_losses = []
for traj_ind in range(5):
    for t in range(76):
        if t == 0:
            point_cloud_group = root['point_cloud']
            pos = point_cloud_group['point'].to(device)
            x = point_cloud_group['rgb'].to(device)
        point_cloud_data = Data(pos=point_cloud, x=x)
        #Handle state invariance for features
        #This setup could have batching problems if we don't downsample the point clouds.
        #However, if we do downsample the point clouds,there could be problems with the proper number of correspondences.
        #Quick fix: Downsample the full point cloud to 6000 points. Downsample the partial view to 1500 points. Match.
        action = root['dynamics_sequences'][traj_ind]['delta_gripper_pos'][t].to(device)
        pointnet_features = model.get_pointnet_features(point_cloud_data)
        state_action = torch.cat((pointnet_features, action.view(1, -1)), dim=1)
        pred = dynamics_mlp(pointnet_features) + point_cloud
        partial_view = root['dynamics_sequences'][traj_ind]['point_cloud'][f'timestep_{t}']['view_0']['point'].to(device)
        point_cloud = match_points(pred, partial_view)
        # Run point_cloud through GarmentNets to get predictions
        #-----------

        #-----------
        full_view_for_loss = torch.cat([root['dynamics_sequences'][traj_ind]['point_cloud'][f'timestep_{t+1}'][f'view_{i}']['point'].to(device) for i in range(4)], dim=0)
        chamfer_losses.append(chamfer_loss(pred, full_view_for_loss))




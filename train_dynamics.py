from datasets.pointnet_dynamics_dataset import PointNetDynamicsDataModule
import torch
from torch import nn
from dynamics_helper import chamfer_loss, DynamicsMLP
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
import yaml
from tqdm import tqdm
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config/dynamics.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
with open('config/dynamics_val.yaml', 'r') as f:
    cfg_val = yaml.load(f, Loader=yaml.FullLoader)
EPOCHS = 100
action_dim = 3
input_dim = 128 + action_dim
hidden_dim = 512
output_dim = 3
train_loader = None
val_loader = None

dynamics_mlp = DynamicsMLP(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.AdamW(dynamics_mlp.parameters(), lr=1e-3)

checkpoint_path = os.path.expanduser(cfg['main']['checkpoint_path'])
model_cpu = ConvImplicitWNFPipeline.load_from_checkpoint(checkpoint_path)
device = torch.device('cuda:{}'.format(cfg['main']['gpu_id']))
model = model_cpu.to(device)
model.eval()
model.requires_grad_(False)

# Load train data
#-----------
datamodule = PointNetDynamicsDataModule(**cfg['datamodule'])
datamodule.prepare_data()
batch_size = 1
dataloader = getattr(datamodule, '{}_dataloader'.format('train'))()

datamodule_val = PointNetDynamicsDataModule(**cfg_val['datamodule'])
datamodule_val.prepare_data()
batch_size = 1
dataloader_val = getattr(datamodule_val, '{}_dataloader'.format('train'))()

best_loss_val = None
for epoch in range(EPOCHS):
    total_loss = total_iters = 0
    model.train()
    print('train')
    for batch_idx, batch_cpu in enumerate(tqdm(dataloader)):
        batch = batch_cpu.to(device=device)
        # print("World frame max position:", torch.max(batch.pos,dim=0)[0],"World frame min position",torch.min(batch.pos,dim=0)[0])

        pointnet2_result = model.pointnet2_forward(batch)
        features = pointnet2_result['per_point_features']
        action = batch.delta_gripper
         
        feature_action = torch.cat((features, action.view(1, -1).repeat(6000,1)), dim=1).float()
        pred = dynamics_mlp(feature_action) + batch.pos
        loss = chamfer_loss(pred, batch.next_pos)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        total_iters += 1
    with torch.no_grad():
        model.eval()
        total_loss_val = total_iters_val = 0
        print('val')
        for batch_idx, batch_cpu in enumerate(tqdm(dataloader_val)):
            pointnet2_result = model.pointnet2_forward(batch)
            features = pointnet2_result['per_point_features']
            action = batch.delta_gripper
            
            feature_action = torch.cat((features, action.view(1, -1).repeat(6000,1)), dim=1).float()
            pred = dynamics_mlp(feature_action) + batch.pos
            loss = chamfer_loss(pred, batch.next_pos)
            total_loss_val += loss.item()
            total_iters_val += 1
    print(f'Epoch {epoch} train loss: {total_loss/total_iters}, val loss: {total_loss_val/total_iters_val}')
    if best_loss_val is None:
        torch.save(dynamics_mlp.state_dict(), 'models/dynamics_mlp.pt')
        best_loss_val = total_loss_val/total_iters_val
    elif total_loss_val/total_iters_val < best_loss_val:
        torch.save(dynamics_mlp.state_dict(), 'models/dynamics_mlp.pt')
        best_loss_val = total_loss_val/total_iters_val

    
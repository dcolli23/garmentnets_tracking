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
# Rigid baseline.
# [0.00449599 0.00486236 0.00531547 0.00587186 0.00669546 0.00783389
#  0.0092937  0.01093761 0.01270723 0.01467821 0.01678138 0.01900596
#  0.02147918 0.02404569 0.02699935]
# [0.0006304  0.00067864 0.00069684 0.00075588 0.00121553 0.00197399
#  0.00284563 0.00370283 0.00463877 0.00572717 0.00703605 0.00854797
#  0.01033316 0.01238551 0.01483241]
#-----------

# Rigid baseline with matching.
# [0.00449115 0.00477049 0.00505576 0.00541249 0.00598812 0.00681501
#  0.00788586 0.00915361 0.01054634 0.01207104 0.01371735 0.01537587
#  0.01730672 0.01918267 0.02121439]
# [0.00066226 0.0006896  0.00068634 0.00071714 0.00102866 0.00162108
#  0.002355   0.00318439 0.00406354 0.00512737 0.00627455 0.00737747
#  0.00886264 0.01029826 0.0117446 ]

# Running through dynamics model with no matching/update from sensor.
# [0.00495876 0.0061671  0.00730418 0.00820878 0.00907775 0.01001427
#  0.0110458  0.01217295 0.01327047 0.01439993 0.0154584  0.01652609
#  0.0176741  0.01884541 0.02000354]
# [0.00065668 0.00078336 0.00097895 0.00107289 0.00108327 0.00108925
#  0.00112488 0.00115614 0.00126341 0.00140339 0.00150303 0.00166396
#  0.0018115  0.00197314 0.00213696]

# With matching
# [0.00496827 0.00590686 0.00667737 0.00730907 0.00790623 0.00854919
#  0.00926451 0.00998863 0.01072496 0.01142181 0.01215687 0.01282992
#  0.01359412 0.0143551  0.01516661]
# [0.00066636 0.00074267 0.00081998 0.00081737 0.00079092 0.00079774
#  0.00078113 0.00079397 0.00083721 0.0008731  0.00094547 0.00101048
#  0.00112436 0.00124047 0.00140141]

for epoch in range(EPOCHS):
    total_loss = total_iters = 0
    for batch_idx, batch_cpu in enumerate(tqdm(dataloader)):
        batch = batch_cpu.to(device=device)
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
    print(f'Epoch {epoch} loss: {total_loss/total_iters}')
    torch.save(dynamics_mlp.state_dict(), 'models/dynamics_mlp.pt')
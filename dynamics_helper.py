import torch
from torch import nn
import numpy as np
from torch_geometric.nn.pool import knn
import torch.optim as optim

class DynamicsMLP(nn.Sequential):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=3):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

def chamfer_loss(pred_points, gt_points):
    forward = knn(gt_points, pred_points, 1)
    backward = knn(pred_points, gt_points, 1)
    forward_chamfer = torch.linalg.norm(pred_points - gt_points[forward[1]], dim=1).mean()
    backward_chamfer = torch.linalg.norm(gt_points - pred_points[backward[1]], dim=1).mean()
    symmetrical_chamfer = (forward_chamfer + backward_chamfer)/2
    return symmetrical_chamfer

def match_points(full_view, partial_view, full_rgb, partial_rgb):
    """
    Match points in partial view to points in full view. Update the points in the full view with the closest points in the partial view.
    Use Euclidean distance to find closest points Use Pytorch knn
    full_view: 6000x3, partial_view: 1500x3
    idx: 2 X D

    """
    idx = knn(full_view, partial_view, 1)
    # idx = knn(partial_view, full_view, 1)
    full_view[idx[1]] = partial_view
    full_rgb[idx[1]] = partial_rgb
    return full_view, full_rgb



 

def match_points_chamfer(predict_pc, partial_sensor_pc, full_rgb, partial_rgb,lr=0.01,num_iter=50,device='cuda'):
    """
    Match points in partial view to points in full view. Update the points in the full view with the closest points in the partial view.
    Use Euclidean distance to find closest points Use Pytorch knn
    full_view: 6000x3, partial_view: 1500x3
    idx: 2 X D

    """
    idx = knn(predict_pc, partial_sensor_pc, 1)
    predict_partical_view = predict_pc[idx[1],:]
    c_loss_initial = chamfer_loss(predict_partical_view,partial_sensor_pc)
    
    with torch.enable_grad():
        # dx = torch.nn.Parameter(torch.zeros(predict_pc.shape, requires_grad=True,device=device))
        dx = torch.nn.Parameter(torch.zeros((1,3), requires_grad=True,device=device))
        opt = optim.Adam([dx], lr=1e-3)

        for i in range(num_iter):
            opt.zero_grad()
            loss = chamfer_loss(predict_partical_view + dx,partial_sensor_pc)
            if loss<1e-6:
                break
            loss.backward()
            opt.step()   

    full_view = predict_pc+dx
    full_view[idx[1]] = partial_sensor_pc
    full_rgb[idx[1]] = partial_rgb
    c_loss_final = chamfer_loss(predict_partical_view+dx,partial_sensor_pc)
   
    return  full_view,full_rgb
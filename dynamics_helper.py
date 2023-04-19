import torch
from torch import nn
import numpy as np
from torch_geometric.nn.pool import knn

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
    """
    knn = knn(full_view, partial_view, 1)
    full_view[knn[1]] = partial_view
    full_rgb[knn[1]] = partial_rgb
    return full_view, full_rgb

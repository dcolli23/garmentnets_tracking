from typing import Tuple, Optional
import os
import pathlib
import copy

import igl
import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data import Subset
from scipy.spatial.transform import Rotation
import pytorch_lightning as pl
from torch.utils.data import Subset

from common.cache import file_attr_cache
from components.gridding import nocs_grid_sample
from common.geometry_util import (
    barycentric_interpolation, mesh_sample_barycentric, AABBGripNormalizer)
from common.gripper_util import get_gripper_locs, translate_cloud_to_origin

# helper functions
# ================
def _get_groups_df(samples_group):
    rows = dict()
    for key, group in samples_group.items():
        rows[key] = group.attrs.asdict()
    groups_df = pd.DataFrame(data=list(rows.values()), index=rows.keys())
    groups_df.drop_duplicates(inplace=True)
    groups_df['group_key'] = groups_df.index
    return groups_df

# data sets
# =========
class PointNetDynamicsDataset(Dataset):
    def __init__(self, 
            # zarr
            zarr_path: str, 
            metadata_cache_dir: str,
            # sample size
            num_pc_sample: int = 6000,
            num_volume_sample: int = 0,
            num_surface_sample: int = 0,
            num_mc_surface_sample: int = 0,
            # mixed sampling config
            surface_sample_ratio: float = 0,
            surface_sample_std: float = 0.05,
            # surface sample noise
            surface_normal_noise_ratio: float = 0,
            surface_normal_std: float = 0,
            # data augumentaiton
            enable_augumentation: bool = True,
            random_rot_range: Tuple[float, float] = (-90, 90),
            num_views: int = 4,
            pc_noise_std: float = 0,
            # volume config
            volume_size: int = 128,
            volume_group: str = 'nocs_winding_number_field',
            tsdf_clip_value: Optional[float] = None,
            volume_absolute_value: bool = False,
            include_volume: bool = False,
            # random seed
            static_epoch_seed: bool = False,
            # catch all
            **kwargs):
        """
        If static_point_sample is True, the points sampled for each index
        will be identical each time being called.
        """
        super().__init__()
        path = pathlib.Path(os.path.expanduser(zarr_path))
        assert(path.exists())
        print(path)
        print(str(path.absolute()))
        root = zarr.open(str(path.absolute()), mode='r')
        samples_group = root['samples']

        # extract common info from sample group
        _, sample_group = next(samples_group.groups())
        # print(sample_group.tree())

        # load group metadata
        groups_df = file_attr_cache(zarr_path, 
            cache_dir=metadata_cache_dir)(_get_groups_df)(samples_group)
        print(len(groups_df))
        # check if index is sorted
        assert(groups_df.index.is_monotonic_increasing)
        groups_df['idx'] = np.arange(len(groups_df))

        volume_task_space = False
        if volume_group == 'sim_nocs_winding_number_field':
            # don't make mistance twice
            volume_task_space = True
            assert(num_mc_surface_sample == 0)
        
        # global state
        self.samples_group = samples_group
        self.groups_df = groups_df
        # sample size
        self.num_pc_sample = num_pc_sample
        # mixed sampling config
        self.surface_sample_ratio = surface_sample_ratio
        self.surface_sample_std = surface_sample_std
        # surface sample noise
        self.surface_normal_noise_ratio = surface_normal_noise_ratio
        self.surface_normal_std = surface_normal_std
        # data augumentaiton
        self.enable_augumentation = enable_augumentation
        self.random_rot_range = random_rot_range
        self.num_views = num_views
        assert(num_views > 0)
        self.pc_noise_std = pc_noise_std
        # volume config
        self.volume_size = volume_size
        self.volume_group = volume_group
        self.tsdf_clip_value = tsdf_clip_value
        self.volume_absolute_value = volume_absolute_value
        self.include_volume = include_volume
        self.volume_task_space = volume_task_space
        # random seed
        self.static_epoch_seed = static_epoch_seed

        # aabb

    def __len__(self):
        return len(self.groups_df) * 5 * 74

    def data_io(self, idx: int) -> dict:
        dataset_idx = idx // (5 * 74)
        dyn_seq_idx = (idx % (5 * 74)) // 74
        pc_idx = (idx % (5 * 74)) % 74
        row = self.groups_df.iloc[dataset_idx]
        group = self.samples_group[row.group_key]

        # io
        pc_group = group['point_cloud']
        dyn_seq = group['dynamics'][dyn_seq_idx]

        # Need to add a zero delta in front so that our gripper location array starts at the same 
        # timestep as the dynamics simulation
        gripper_zero_delta = np.array((0, 0, 0)).reshape(1, 3)
        gripper_deltas = np.concatenate((
            gripper_zero_delta, 
            dyn_seq['delta_gripper_pos'][:]
        ), axis=0)
        grip_pos_cumulative = np.cumsum(gripper_deltas, axis=0)
        grip_locs = get_gripper_locs(grip_pos_cumulative) 

        if pc_idx != 0:
            pc_spec = dyn_seq['point_cloud'][f'timestep_{pc_idx}']
            full_view_pos = np.concatenate([pc_spec['view_0']['point'], pc_spec['view_1']['point'], pc_spec['view_2']['point'], pc_spec['view_3']['point']], axis=0)
            full_view_rgb = np.concatenate([pc_spec['view_0']['rgb'], pc_spec['view_1']['rgb'], pc_spec['view_2']['rgb'], pc_spec['view_3']['rgb']], axis=0)
        else:
            full_view_pos = pc_group['point'] + np.array((0.0, 0.0, 0.4),dtype=float).reshape(1, 3)
            full_view_rgb = pc_group['rgb']
        pc_spec_1 = dyn_seq['point_cloud'][f'timestep_{pc_idx + 1}']
        next_view_pos = np.concatenate([pc_spec_1['view_0']['point'], pc_spec_1['view_1']['point'], pc_spec_1['view_2']['point'], pc_spec_1['view_3']['point']], axis=0)

        # Translate both the full view at t=t-1 and the partial view at t=t clouds back to origin.
        # Confusing indexing is to preserve dimension.
        full_view_pos = translate_cloud_to_origin(full_view_pos, grip_locs, pc_idx)
        next_view_pos = translate_cloud_to_origin(next_view_pos, grip_locs, pc_idx)
        data = {
            'pc_sim': full_view_pos[:],
            'pc_sim_rgb': full_view_rgb[:],
            'next_pc': next_view_pos[:],
            'delta_gripper': dyn_seq['delta_gripper_pos'][pc_idx][:]
        }
        return data
    
    def get_base_data(self, idx:int, data_in: dict) -> dict:
        """
        Get non-volumetric data as numpy arrays
        """
        num_pc_sample = self.num_pc_sample
        static_epoch_seed = self.static_epoch_seed
        num_views = self.num_views

        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(len(data_in['pc_sim']))

        selected_idxs = rs.choice(all_idxs, size=num_pc_sample, replace=False)

        pc_sim_rgb = data_in['pc_sim_rgb'][selected_idxs].astype(np.float32) / 255
        pc_sim = data_in['pc_sim'][selected_idxs].astype(np.float32)

        all_idxs_next = np.arange(len(data_in['next_pc']))
        selected_idxs_next = rs.choice(all_idxs_next, size=num_pc_sample, replace=False)
        next_x = data_in['next_pc'][selected_idxs_next].astype(np.float32)

        data = {
            'x': pc_sim_rgb,
            'pos': pc_sim,
            'next_pos': next_x,
            'delta_gripper': data_in['delta_gripper'].astype(np.float32),
        }
        return data
        
    def reshape_for_batching(self, data: dict) -> dict:
        out_data = dict()
        for key, value in data.items():
            out_data[key] = value.reshape((1,) + value.shape)
        return out_data
    
    def __getitem__(self, idx: int) -> Data:
        data_in = self.data_io(idx)
        data = self.get_base_data(idx, data_in=data_in)
        data['input_aug_rot_mat'] = np.expand_dims(np.eye(3, dtype=np.float32), axis=0)

        data_torch = dict(
            (x[0], torch.from_numpy(x[1])) for x in data.items())
        pg_data = Data(**data_torch)
        return pg_data

# data modules
# ============
class PointNetDynamicsDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert(len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_args['static_epoch_seed'] = False
        train_dataset = PointNetDynamicsDataset(**train_args)
        print('len dataset', len(train_dataset))
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.static_epoch_seed = True

        groups_df = train_dataset.groups_df
        instances_df = groups_df.groupby('sample_id').agg({'idx': lambda x: sorted(x)})
        # split for train/val/test
        num_instances = len(instances_df)
        normalized_split = np.array(dataset_split)
        normalized_split = normalized_split / np.sum(normalized_split)
        instance_split = (normalized_split * num_instances).astype(np.int64)

        # add leftover instance to training set
        instance_split[0] += num_instances - np.sum(instance_split)

        # generate index for each
        all_idxs = np.arange(num_instances)
        rs = np.random.RandomState(seed=split_seed)
        perm_all_idxs = rs.permutation(all_idxs)

        split_instance_idx_list = list()
        prev_idx = 0
        for x in instance_split:
            next_idx = prev_idx + x
            split_instance_idx_list.append(perm_all_idxs[prev_idx: next_idx])
            prev_idx = next_idx
            break
        print([len(x) for x in split_instance_idx_list], instance_split)
        # assert(np.allclose([len(x) for x in split_instance_idx_list], instance_split))
        print(split_instance_idx_list)
        split_idx_list = list()
        for instance_idxs in split_instance_idx_list:
            idxs = np.sort(np.concatenate(instances_df.iloc[instance_idxs].idx))
            split_idx_list.append(idxs)
            break
        # assert(sum(len(x) for x in split_idx_list) == len(groups_df))
        # generate subsets
        # train_idxs, val_idxs, test_idxs = split_idx_list
        train_idxs = split_idx_list[0]
        train_subset = Subset(train_dataset, train_idxs)

        # val_subset = Subset(val_dataset, val_idxs)
        # test_subset = Subset(val_dataset, test_idxs)

        self.groups_df = groups_df
        self.train_idxs = train_idxs
        # self.val_idxs = val_idxs
        # self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        # self.val_dataset = val_dataset
        self.train_subset = train_subset
        # self.val_subset = val_subset
        # self.test_subset = test_subset
    
    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        dataloader = DataLoader(self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers)
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        val_subset = self.val_subset
        dataloader = DataLoader(val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers)
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        test_subset = self.test_subset
        dataloader = DataLoader(test_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers)
        return dataloader

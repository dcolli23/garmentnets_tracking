import numpy as np

GRIPPER_POS_INIT_ARR = np.array((0.0, 0.0, 0.4)).reshape(1, 3)

def get_gripper_locs(gripper_deltas_cumulative):
    return gripper_deltas_cumulative + GRIPPER_POS_INIT_ARR

def translate_cloud_to_origin(cloud, grip_locs, timestep_idx):
    return cloud - grip_locs[timestep_idx:timestep_idx + 1, :]

def translate_gripper_to_world_frame(cloud, grip_locs, timestep_idx):
    return cloud + grip_locs[timestep_idx:timestep_idx + 1, :]
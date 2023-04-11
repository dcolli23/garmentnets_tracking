from simulation.pipeline.smpl_simulation_pipeline import smpl_simulation_pipeline

LINE_SEP = 80 * '-'

def simulate_garment_hanging_rest_state(sample_config: dict, sample_data: dict,
                                       frames_to_resting_state: int=120):
    """Wrapper for SMPL simulation that runs physics to simulate garment's hanging resting state

    This can then be used with simulation.physics.set_sim_output_as_default_mesh_shape() to set the
    output of this simulation as the new mesh default state. We can then run additional physics
    simulations (like movement with a "gripper") on this new resting state since Blender doesn't
    support sequential physics simulations (to my knowledge).
    """
    smpl_sim_args = dict()
    smpl_sim_args.update(sample_config)
    smpl_sim_args.update(sample_data)
    smpl_sim_args["simulation_duration_pair"] = (0, frames_to_resting_state)

    print("Running SMPL Simulation Pipeline")
    print(LINE_SEP)
    result_data_smpl = smpl_simulation_pipeline(**smpl_sim_args)

    return result_data_smpl
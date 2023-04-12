import os
from pathlib import Path

import bpy

class BlendFileCheckpointer:
    """Saves .blend file checkpoints if given a file path root"""
    garment_sample_dir: str
    __save_new_checkpoints: bool
    __checkpoint_idx_next: int

    def __init__(self, garment_sample_dir: Path, save_new_checkpoints: bool):
        self.garment_sample_dir = garment_sample_dir
        self.__save_new_checkpoints = save_new_checkpoints
        self.__checkpoint_idx_next = 1
        self.__rest_state_file_name = "hanging_rest_state.blend"
        self.__rest_state_file_path = self.garment_sample_dir / self.__rest_state_file_name

    def does_rest_state_checkpoint_exist(self):
        """Public method for checking if the resting state checkpoint exists"""
        return self.__rest_state_file_path.exists()

    def save_hanging_rest_state_if_desired(self):
        """Saves special checkpoint for after simulating the garment's hanging rest state IF DESIRED

        The "IF DESIRED" portion refers to if `save_new_checkpoints=True` was set when constructing
        the checkpointer.
        """
        if self.__save_new_checkpoints:
            self.save_hanging_rest_state()
        else:
            if self.__rest_state_file_path.exists():
                # The rest state has already been checkpointed and we're not saving new checkpoints.
                # This is okay but we want to print so user knows what's happening.
                print("Rest state already checkpointed and not saving new checkpoints.")
                print("If you'd like to save this rest state as a checkpoint, delete the old")
                print("checkpoint and set the checkpointer to save new checkpoint files.")
            else:
                # The rest state hasn't been checkpointed and we're not saving new checkpoints.
                raise RuntimeError(
                    "Not saving new checkpoints and the rest state hasn't been checkpointed!\n"
                    "Can't resimulate if the hanging rest state hasn't been checkpointed."
                )

    def save_hanging_rest_state(self):
        """Saves special checkpoint for after simulating the garment's hanging rest state

        This is particularly helpful in the case of simulating multiple dynamics runs on the same
        garment.
        """
        if not self.__rest_state_file_path.exists():
            # Reset data so we don't have to do this everytime we reload the checkpoint.
            # NOTE: For some reason, physics bakes won't delete if we reset the frame first.
            bpy.ops.ptcache.free_bake_all()
            bpy.context.scene.animation_data_clear()
            bpy.context.scene.frame_set(0)

            bpy.ops.wm.save_as_mainfile(filepath=self.__rest_state_file_path.as_posix())
            print("Saved hanging rest state checkpoint to:", self.__rest_state_file_path)
        else:
            raise RuntimeError("Hanging rest state checkpoint already exists! Delete it to save new checkpoint.")

    def save_checkpoint_if_desired(self):
        """Saves Blender checkpoint as .blend file for e.g. checking status or resimulation"""
        if self.__save_new_checkpoints:
            filepath = self.__form_checkpoint_path(self.__checkpoint_idx_next)
            bpy.ops.wm.save_as_mainfile(filepath=filepath.as_posix())
            print(f"Saved blend file checkpoint to '{filepath}'")

        checkpoint_saved = self.__checkpoint_idx_next
        self.__checkpoint_idx_next += 1
        return checkpoint_saved

    def load_checkpoint(self, checkpoint_idx: int):
        """Loads checkpoint, reseting the Blender environment so that it's ready for resim"""
        checkpoint_filepath = self.__form_checkpoint_path(checkpoint_idx)
        assert (checkpoint_filepath.exists()), (
            f"Given checkpoint doesn't exist! Checkpoint: {checkpoint_filepath}"
        )

        bpy.ops.wm.open_mainfile(filepath=checkpoint_filepath.as_posix())
        print("Successfully reset Blender to checkpoint:", checkpoint_filepath)

    def load_hanging_rest_state(self):
        if not self.__rest_state_file_path.exists():
            raise RuntimeError("Hanging rest state checkpoint doesn't exist! Can't load!")

        bpy.ops.wm.open_mainfile(filepath=self.__rest_state_file_path.as_posix())
        print("Successfully reloaded hanging rest state checkpoint.")

    def __form_checkpoint_path(self, checkpoint_idx: int) -> Path:
        filepath = self.garment_sample_dir / f"checkpoint_{checkpoint_idx}.blend"
        return filepath

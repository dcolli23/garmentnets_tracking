
import bpy

class BlendFileCheckpointer:
    """Saves .blend file checkpoints if given a file path root"""
    filepath_root: str
    _save_checkpoints: bool
    _checkpoint_idx: int

    def __init__(self, checkpoint_filepath_root: str):
        self.filepath_root = checkpoint_filepath_root
        self._save_checkpoints = self.filepath_root is not None
        self._checkpoint_idx = 1

    def save_checkpoint_if_desired(self):
        if self._save_checkpoints:
            filepath = self.filepath_root + f"_checkpoint_{self._checkpoint_idx}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=filepath)
            print(f"Saved blend file checkpoint to '{filepath}'")

        self._checkpoint_idx += 1
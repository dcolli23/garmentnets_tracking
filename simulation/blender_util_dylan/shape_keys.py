import bpy

def set_obj_default_shape_from_shape_key(obj: bpy.types.Object, key_name: str, 
                                         verbose: bool = False):
    # I don't know if selecting the object is necessary but it seems wise to do so.
    bpy.context.view_layer.objects.active = obj

    # TODO: Is the first key always "Key" or can it change?
    # I'll add an assert here so I at least catch when/if this assumption is wrong.
    assert (("Key" in bpy.data.shape_keys.keys())
            and (len(list(bpy.data.shape_keys.keys())) == 1)), (
                "Assumption that 'Key' is the one and only key in bpy.data.shape_keys is wrong!\n"
                "Dig into why this is and how to fix it."
            )

    # Set the interpolation value of the shape key that we wish to set as the mesh's default shape 
    # to 1.0 so that this is the only shape key that influences the shape of the mesh.
    bpy.data.shape_keys["Key"].key_blocks[key_name].value = 1.0

    last_shape_key_to_remove = None
    for i, (k, v) in enumerate(bpy.data.shape_keys["Key"].key_blocks.items()):
        # Need to remove the shape_key that we want the object to deform to last.
        if k == key_name:
            last_shape_key_to_remove = v
            continue

        if verbose:
            print(f"Removing '{k}' shape key")
            keys_before_remove = set(bpy.data.shape_keys["Key"].key_blocks.keys())

        # bpy.context.object.active_shape_key_index = i
        # bpy.ops.object.shape_key_remove(all=False)
        obj.shape_key_remove(key=v)

        if verbose:
            keys_after_remove = set(bpy.data.shape_keys["Key"].key_blocks.keys())
            key_diff = keys_before_remove - keys_after_remove
            assert (len(key_diff) == 1)
            assert (k in key_diff)

    assert (last_shape_key_to_remove is not None), f"Didn't find key, '{key_name}' in shape keys!"

    # Now remove the last shape key, essentially applying it so that that shape is the mesh's
    # resting shape.
    obj.shape_key_remove(key=v)
# Planned Final Zarr Structure

Full structure of the Zarr file used to store data.

```
/
 └── Tshirt
 |   ├── samples
 |   |   ├── 00380_Tshirt_509
     │   │   ├── image
     │   │   │   ├── depth (4, 1024, 1024, 1) float16
     │   │   │   ├── mask (4, 1024, 1024, 1) bool
     │   │   │   ├── rgb (4, 1024, 1024, 3) uint8
     │   │   │   └── uv (4, 1024, 1024, 2) float16
     │   │   ├── misc
     │   │   │   ├── cloth_aabb (2, 3) float32
     │   │   │   ├── cloth_canonical_aabb (2, 3) float32
     │   │   │   ├── cloth_canonical_verts (4762, 3) float32
     │   │   │   ├── cloth_faces (4668, 4) uint16
     │   │   │   ├── cloth_texture (2048, 2048, 3) uint8
     │   │   │   ├── cloth_uv_faces (4668, 4) uint16
     │   │   │   ├── cloth_uv_verts (4936, 2) float32
     │   │   │   ├── cloth_verts (4762, 3) float32
     │   │   │   ├── extrinsic_list (4, 4, 4) float32
     │   │   │   ├── human_canonical_aabb (2, 3) float32
     │   │   │   ├── human_faces (13776, 3) uint16
     │   │   │   ├── human_verts (6890, 3) float32
     │   │   │   └── intrinsic (3, 3) float32
     │   │   ├── point_cloud
     │   │   |   ├── canonical_point (91320, 3) float16
     │   │   |   ├── point (91320, 3) float16
     │   │   |   ├── rgb (91320, 3) uint8
     │   │   |   ├── sizes (4,) int64
     │   │   |   └── uv (91320, 2) float16
     │   │   ├── dynamics_sequences
     │   │   |   ├── 0
     │   │   |   |   ├── point_cloud
     |   |   |   |   |   ├── timestep_0
     │   │   |   |   |   |   ├── view_0
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_1
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_2
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_3
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     |   |   |   |   |   ├── timestep_1
     │   │   |   |   |   |   ├── view_0
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_1
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_2
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8
     │   │   |   |   |   |   ├── view_3
     │   │   |   |   |   |   |   ├── point (num_points, 3) float16
     |   │   │   |   |   |   |   ├── rgb (num_frames, num_points, 3) uint8

                          REPEAT AD NAUSEAM

     |   |   |   |   |
     │   │   |   |   ├── delta_gripper_pos (num_frames, 3) float16
     │   │   |   ├── 1 (same structure as '0' above)
     │   │   |   ├── 2 (same structure as '0' above)
     │   │   |   ├── 3 (same structure as '0' above)
     │   │   |   ├── 4 (same structure as '0' above)
```
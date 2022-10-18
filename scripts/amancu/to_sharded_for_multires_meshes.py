from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


src_path = 'file:///ssdscratch/songbird/j0251/agglo2_unsharded/sv/'
dest_path = 'file:///ssdscratch/songbird/j0251/agglo2_meshes/sv_multilod_1/'
# src_path = 'file:///wholebrain/scratch/amancu/multires_meshes_precomputed/sv/'
# dest_path = 'file:///wholebrain/scratch/amancu/multires_meshes_sharded/sv/'

with LocalTaskQueue(parallel=80) as tq:
    tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
        src_path, 
        dest_path,
        min_shards=400,
        draco_compression_level = 7,
        num_lod = 4,
    )
    tq.insert_all(tasks)

# src_path = 'file:///ssdscratch/songbird/j0251/agglo2_unsharded/mi/'
# dest_path = 'file:///ssdscratch/songbird/j0251/agglo2_meshes/mi_multilod/'
# src_path = 'file:///wholebrain/scratch/amancu/multires_meshes_precomputed/mi/'
# dest_path = 'file:///wholebrain/scratch/amancu/multires_meshes_sharded/mi/'

# with LocalTaskQueue(parallel=50) as tq:
#     tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
#         src_path, 
#         dest_path,
#         min_shards=400,
#         draco_compression_level = 7,
#         num_lod = 2,
#     )
#     tq.insert_all(tasks)

# src_path = 'file:///ssdscratch/songbird/j0251/agglo2_unsharded/vc/'
# dest_path = 'file:///ssdscratch/songbird/j0251/agglo2_meshes/vc_multilod/'
# src_path = 'file:///wholebrain/scratch/amancu/multires_meshes_precomputed/vc/'
# dest_path = 'file:///wholebrain/scratch/amancu/multires_meshes_sharded/vc/'

# with LocalTaskQueue(parallel=50) as tq:
#     tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
#         src_path, 
#         dest_path,
#         min_shards=400,
#         draco_compression_level = 7,
#         num_lod = 2,
#     )
#     tq.insert_all(tasks)

# src_path = 'file:///wholebrain/scratch/amancu/multires_meshes_precomputed/sj/'
# dest_path = 'file:///wholebrain/scratch/amancu/multires_meshes_sharded/sj/'
# src_path = 'file:///ssdscratch/songbird/j0251/agglo2_unsharded/unsharded_sj/'
# dest_path = 'file:///ssdscratch/songbird/j0251/agglo2_meshes/sj_multilod/'

# with LocalTaskQueue(parallel=2) as tq:
#     tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
#         src_path, 
#         dest_path,
#         min_shards=400,
#         draco_compression_level = 7,
#         num_lod = 4,
#     )
#     tq.insert_all(tasks)
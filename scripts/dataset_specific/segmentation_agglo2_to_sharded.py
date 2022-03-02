from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag1/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag1/'

# print(f'Start mag 1')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(64,64,64),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)


# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag2/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag2/'

# print(f'Start mag 2')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(64,64,64),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)

# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag4/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag4/'

# print(f'Start mag 4')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(64,64,64),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)

src_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_precomputed_mag8/'
dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag8/'

print(f'Start mag 8')

with LocalTaskQueue(parallel=20) as tq:

    tasks = tc.create_image_shard_transfer_tasks(
        src_path, dest_path,
        mip=0, chunk_size=(64,64,64),
        encoding=None, bounds=None, fill_missing=True,
        translate=(0, 0, 0), dest_voxel_offset=None,
        agglomerate=False, timestamp=None,
        memory_target=3.5e9,
    )
    tq.insert_all(tasks)

# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag16/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag16/'

# print(f'Start mag 16')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(32,32,32),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)

# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag32/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag32/'

# print(f'Start mag 32')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(16,16,16),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)


# src_path = 'file:///wholebrain/scratch/anaml/j0251/agglo2/agglo2_precomputed_mag64/compressed_segmentation/'
# dest_path = 'file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_sharded_mag64/'

# print(f'Start mag 64')

# with LocalTaskQueue(parallel=20) as tq:

#     tasks = tc.create_image_shard_transfer_tasks(
#         src_path, dest_path,
#         mip=0, chunk_size=(8,8,8),
#         encoding=None, bounds=None, fill_missing=True,
#         translate=(0, 0, 0), dest_voxel_offset=None,
#         agglomerate=False, timestamp=None,
#         memory_target=3.5e9,
#     )
#     tq.insert_all(tasks)

print("Done!")
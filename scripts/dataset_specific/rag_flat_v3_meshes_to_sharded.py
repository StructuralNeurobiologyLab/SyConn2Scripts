import numpy as np
import multiprocessing
from PIL import Image
from cloudvolume import CloudVolume
# from cloudvolume.metadata import PrecomputedMetadata
from cloudvolume.mesh import Mesh
from cloudvolume.datasource.precomputed import create_precomputed
from cloudvolume.datasource.precomputed.mesh import UnshardedLegacyPrecomputedMeshSource, PrecomputedMeshSource
from concurrent.futures import ProcessPoolExecutor
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


src_path = 'file:///dev/shm/tmpam/rag_flat_v3_meshes/'
dest_path = 'file:///dev/shm/tmpam/rag_flat_v3_meshes_sharded/sv/'

with LocalTaskQueue(parallel=100) as tq:
    tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
        src_path, 
        dest_path,
    )
    tq.insert_all(tasks)



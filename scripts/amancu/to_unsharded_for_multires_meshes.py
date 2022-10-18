import os, sys
import numpy as np
import multiprocessing
from knossos_utils import KnossosDataset
from PIL import Image
from cloudvolume import CloudVolume
# from cloudvolume.metadata import PrecomputedMetadata
from cloudvolume.mesh import Mesh
from cloudvolume.datasource.precomputed import create_precomputed
from cloudvolume.datasource.precomputed.mesh import UnshardedLegacyPrecomputedMeshSource, PrecomputedMeshSource
# from cloudvolume.multilod import UnshardedMultiLevelPrecomputedMeshSource
from concurrent.futures import ProcessPoolExecutor
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
import syconn.reps.segmentation_helper as helper

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.cacheservice import CacheService
from cloudvolume.cloudvolume import SharedConfiguration


cloudpath = 'file:///wholebrain/scratch/amancu/multires_meshes_precomputed/'
# cloudpath = 'file:///ssdscratch/songbird/j0251/agglo2_unsharded/'

global_params.wd = '/ssdscratch/pschuber/songbird/j0251/j0251_72_seg_20210127_agglo2/'
ssd = SuperSegmentationDataset()
ssv_ids = [1479909980, 758242433, 1080458707, 1639126257]
n_proc = 15

volume_info = {
        "type": "segmentation",
        "layer_type": "segmentation",
        "mesh": "mesh",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "key": "1_1_1",
                "size": [27119, 27350, 15494],
                "resolution": [10, 10, 25],
                "chunk_sizes": [[512, 512, 512]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8,8,8],
                "factor": (1,1,1),
            }
        ]
    }

# vol_sv = create_precomputed(cloudpath + "sv/", fill_missing=True, info=volume_info, mesh_dir='mesh')
# vol_sv.provenance.description = 'Part j0251 dataset meshes, agglo2 acquisition'
# vol_sv.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
# vol_sv.commit_info()  # generates file://.../info json file
# vol_sv.commit_provenance()  # generates file://.../provenance json file

vol_vc = create_precomputed(cloudpath + "vc/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_vc.provenance.description = 'Part j0251 dataset meshes vcs'
vol_vc.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_vc.commit_info()  # generates file://.../info json file
vol_vc.commit_provenance()  # generates file://.../provenance json file

vol_mi = create_precomputed(cloudpath + "mi/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_mi.provenance.description = 'Part j0251 dataset meshes mitos'
vol_mi.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_mi.commit_info()  # generates file://.../info json file
vol_mi.commit_provenance()  # generates file://.../provenance json file

vol_sj = create_precomputed(cloudpath + "sj/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_sj.provenance.description = 'Part j0251 dataset meshes sjs'
vol_sj.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_sj.commit_info()  # generates file://.../info json file
vol_sj.commit_provenance()  # generates file://.../provenance json file

# split ssv_ids into multiple slices for processing
len_end = len(ssv_ids)
chunksize = len_end // n_proc
proc_slices = []

for i_proc in range(n_proc):
    chunkstart = int(i_proc * chunksize)
    # make sure to include the division remainder for the last process
    chunkend = int((i_proc + 1) * chunksize) if i_proc < n_proc - 1 else len_end
    proc_slices.append(np.s_[chunkstart:chunkend])

print(f"proc_slices: {proc_slices}")

cache=False
config = SharedConfiguration(
    cdn_cache=False,
    compress=True,
    compress_level=None,
    green=False,
    mip=0,
    parallel=1,
    progress=False,
    secrets=None,
    spatial_index_db=None
)
cache = CacheService(
    cloudpath=(cache if type(cache) == str else cloudpath),
    enabled=bool(cache),
    config=config,
    compress=True,
)
# mesh_meta_sv = PrecomputedMetadata(cloudpath+'sv/', config, cache, volume_info)
mesh_meta_vc = PrecomputedMetadata(cloudpath+'vc/', config, cache, volume_info)
mesh_meta_mi = PrecomputedMetadata(cloudpath+'mi/', config, cache, volume_info)
mesh_meta_sj = PrecomputedMetadata(cloudpath+'sj/', config, cache, volume_info)


# mesh_source_sv = PrecomputedMeshSource(mesh_meta_sv, cache, config)
mesh_source_vc = PrecomputedMeshSource(mesh_meta_vc, cache, config)
mesh_source_mi = PrecomputedMeshSource(mesh_meta_mi, cache, config)
mesh_source_sj = PrecomputedMeshSource(mesh_meta_sj, cache, config)

sources = [mesh_source_vc, mesh_source_mi, mesh_source_sj] #[mesh_source_vc, mesh_source_mi, mesh_source_sj] [mesh_source_sv, mesh_source_vc, mesh_source_mi, mesh_source_sj]

# load meshes for the cells and put them into cloudvolume
def process(slice):
    global ssv_ids
    len_slice = len(ssv_ids[slice])
    # print(f'For a processor, length of ids: {len_slice} and slice {slice}')
    for i, ssv_id in enumerate(ssv_ids[slice]):
    # for i, ssv_id in enumerate([2766961]):
        ssv = ssd.get_super_segmentation_object(int(ssv_id))

        if i % 1000 == 0:
            print(f'Computing... {i} of slice of {len_slice}')            

        ssv.load_attr_dict()

        # load each obj_type
        for j, obj_type in enumerate(['vc', 'mi', 'sj']): # enumerate(['vc', 'mi','sj']): # add SVs here if desired
            if obj_type == 'vc':
                mesh = ssv.vc_mesh
            elif obj_type == 'mi':
                mesh = ssv.mi_mesh
            elif obj_type == 'sj':
                mesh = ssv.syn_ssv_mesh
            else:
                mesh = ssv.mesh

            indices = np.array(mesh[0], dtype=np.uint32).reshape(-1, 3)
            vertices = np.array(mesh[1], dtype=np.float32).reshape(-1, 3)
            normals = np.array(mesh[2], dtype=np.float32).reshape(-1, 3)

            if len(vertices) == 0:
                print(f'No vertices for cell {ssv_id}')
                continue

            if len(normals) == 0:
                # print(f'No normals for cell {ssv_id}')
                normals = None

            # num_vert = vertices.shape[0]

            curr_mesh = Mesh(
                vertices=vertices, faces=indices, 
                segid=ssv.id,
                normals=normals, 
            )
            curr_mesh.segid=ssv.id
            # print(f'Mesh {obj_type}: {curr_mesh}')
            
            sources[j].put(curr_mesh)
            # print(f'after put')
            

with ProcessPoolExecutor(max_workers=n_proc) as executor:
    executor.map(process, proc_slices)


# vol_sv.mesh = mesh_source_sv
vol_vc.mesh = mesh_source_vc
vol_mi.mesh = mesh_source_mi
vol_sj.mesh = mesh_source_sj

# vol.mesh.viewer()






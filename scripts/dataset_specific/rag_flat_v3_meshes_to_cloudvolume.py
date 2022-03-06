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

# initialize all relevant data
cloudpath = 'file:///dev/shm/tmpam/rag_flat_v3_meshes/'
global_params.wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'
ssd = SuperSegmentationDataset()
ssv_ids = ssd.ssv_ids
n_proc = 140

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

vol_sv = create_precomputed(cloudpath, fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_sv.provenance.description = 'Whole j0251 dataset meshes'
vol_sv.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_sv.commit_info()  # generates file://.../info json file
vol_sv.commit_provenance()  # generates file://.../provenance json file

vol_vc = create_precomputed(cloudpath + "vc/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_vc.provenance.description = 'Whole j0251 dataset meshes'
vol_vc.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_vc.commit_info()  # generates file://.../info json file
vol_vc.commit_provenance()  # generates file://.../provenance json file

vol_mi = create_precomputed(cloudpath + "mi/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_mi.provenance.description = 'Whole j0251 dataset meshes'
vol_mi.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_mi.commit_info()  # generates file://.../info json file
vol_mi.commit_provenance()  # generates file://.../provenance json file

vol_sj = create_precomputed(cloudpath + "sj/", fill_missing=True, info=volume_info, mesh_dir='mesh')
vol_sj.provenance.description = 'Whole j0251 dataset meshes'
vol_sj.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de', 'lacatusu@neuro.mpg.de']
vol_sj.commit_info()  # generates file://.../info json file
vol_sj.commit_provenance()  # generates file://.../provenance json file

# split ssv_ids into multiple slices for processing
chunksize = len(ssv_ids) // n_proc
# chunksize = (len(ssv_ids)/2) // n_proc
proc_slices = []

for i_proc in range(n_proc):
    chunkstart = int(i_proc * chunksize)
    # make sure to include the division remainder for the last process
    chunkend = int((i_proc + 1) * chunksize) if i_proc < n_proc - 1 else (len(ssv_ids)/2)
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

sources = [mesh_source_sv, mesh_source_vc, mesh_source_mi, mesh_source_sj]

# load meshes for the cells and put them into cloudvolume
def process(slice):
    global ssv_ids
    for i, ssv_id in enumerate(ssv_ids[slice]):
        ssv = ssd.get_super_segmentation_object(int(ssv_id))

        if i % 10 == 0:
            print(f'Computing... {i} of slice')            

        ssv.load_attr_dict()

        # load each obj_type
        for j, obj_type in enumerate(['vc', 'mi','sj']): # add SVs here if desired

            if obj_type == 'vc':
                so_list = ssv.vcs
            elif obj_type == 'mi':
                so_list = ssv.mis
            elif obj_type == 'sj':
                so_list = ssv.sjs

            for so in so_list:    
                mesh = helper.load_mesh(so)

                indices = np.array(mesh[0], dtype=np.uint32).reshape(-1, 3)
                vertices = np.array(mesh[1], dtype=np.float32).reshape(-1, 3)
                # num_vert = vertices.shape[0]

                curr_mesh = Mesh(
                    vertices=vertices, faces=indices, 
                    segid=so.id,
                    normals=None,
                    encoding_type='draco', 
                )
                curr_mesh.segid=so.id
                print(f'Mesh: {curr_mesh}')
                
                sources[j+1].put(curr_mesh)
            

with ProcessPoolExecutor(max_workers=n_proc) as executor:
    executor.map(process, proc_slices)


# vol_sv.mesh = mesh_source_sv
vol_vc.mesh = mesh_source_vc
vol_mi.mesh = mesh_source_mi
vol_sj.mesh = mesh_source_sj

# vol.mesh.viewer()


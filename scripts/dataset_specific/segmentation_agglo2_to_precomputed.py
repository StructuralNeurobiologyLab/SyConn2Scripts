import os, sys
import numpy as np
import multiprocessing
from knossos_utils import KnossosDataset
from PIL import Image
from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch
from concurrent.futures import ProcessPoolExecutor
import time

kd = KnossosDataset("/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf")

volume_info8 = volume_info8 = {
        "type": "segmentation",
        "layer_type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "key": "8_8_8",
                "size": [3389, 3418, 1936],
                "resolution": [80, 80, 200],
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8,8,8],
                "factor": (8,8,8),
            }
        ]
    }

vol8 = CloudVolume('file:///wholebrain/scratch/amancu/j0251/agglo2/agglo2_precomputed_mag8/', info=volume_info8, bounded=False, non_aligned_writes=True)
vol8.provenance.description = 'Whole j0251 segmentation conversion mag 8'
vol8.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
vol8.commit_info()  # generates file://.../info json file
vol8.commit_provenance()  # generates file://.../provenance json file

to_upload = []
# split dataset
num_x, num_y, num_z = kd.boundary[0]//512, kd.boundary[1]//512, kd.boundary[2]//512
print(f"num_x: {num_x}")
print(f'Splitting dataset into number of chunks on each xyz-axis: {num_x}, {num_y}, {num_z}')
for x in range(num_x+1):          # +1 for the boundary chunks
    for y in range(num_y+1):
        for z in range(num_z+1):
            to_upload.append((x,y,z))


def process(z):
    # global kd
    try:
        ####### MAG 8 ####################################################################################################################
        x_start = z[0] * 512
        y_start = z[1] * 512
        z_start = z[2] * 512
        # check for boundary chunk ends for knossos
        x_end = (x_start + 512) if (x_start + 512) < kd.boundary[0] else kd.boundary[0]
        y_end = (y_start + 512) if (y_start + 512) < kd.boundary[1] else kd.boundary[1]
        z_end = (z_start + 512) if (z_start + 512) < kd.boundary[2] else kd.boundary[2]
        print(f"Computing chunk {x_start}:{x_end}, {y_start}:{y_end} {z_start}:{z_end}")
        size = (x_end-x_start, y_end-y_start, z_end-z_start)
        # get bytes array from Knossos and reshape it to the according required size
        bytes_array = kd.load_seg(offset=(x_start, y_start, z_start), size=size, mag=8)     # Knossos returns zyx arrays
        bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz
        # check for boundary chunk ends for volume
        x_start = z[0] * 64
        y_start = z[1] * 64
        z_start = z[2] * 64
        x_end = x_start + 64 if (x_start + 64) < 3389 else 3389
        y_end = y_start + 64 if (y_start + 64) < 3418 else 3418
        z_end = z_start + 64 if (z_start + 64) < 1936 else 1936
        vol8[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

    except IOError as err:
      errno, strerror = err.args
      print ('I/O error({0}): {1}'.format(errno, strerror))
      print (err)
    except:
      print ('Unexpected error:', sys.exc_info())
      raise

with ProcessPoolExecutor(max_workers=10) as executor:
    executor.map(process, to_upload)
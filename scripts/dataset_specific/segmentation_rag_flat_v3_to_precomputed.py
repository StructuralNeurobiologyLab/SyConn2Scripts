import os, sys
import numpy as np
import multiprocessing
from knossos_utils import KnossosDataset
from PIL import Image
from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch
from concurrent.futures import ProcessPoolExecutor
import time

kd = KnossosDataset("/ssdscratch/songbird/j0251/segmentation/j0251_rag_flat_Jan2019_seg/knossos.pyk.conf")

volume_info8 = {
        "type": "segmentation",
        "layer_type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "key": "8_8_8",
                "size": [3389, 3418, 1936],
                "resolution": [80, 80, 200],
                "chunk_sizes": [[128, 128, 128]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": [8,8,8],
                "factor": (8,8,8),
            }
        ]
    }

vol8 = CloudVolume('file:///wholebrain/scratch/amancu/j0251/rag_flat_v3/rag_flat_v3_precomputed_mag8/', info=volume_info8, bounded=False, non_aligned_writes=True)
vol8.provenance.description = 'Whole j0251 segmentation conversion mag 8'
vol8.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
vol8.commit_info()  # generates file://.../info json file
vol8.commit_provenance()  # generates file://.../provenance json file

to_upload = []
# split dataset
num_x, num_y, num_z = kd.boundary[0]//1024, kd.boundary[1]//1024, kd.boundary[2]//1024
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
        x_start = z[0] * 1024
        y_start = z[1] * 1024
        z_start = z[2] * 1024
        # check for boundary chunk ends for knossos
        x_end = (x_start + 1024) if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
        y_end = (y_start + 1024) if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
        z_end = (z_start + 1024) if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]
        print(f"Computing chunk {x_start}:{x_end}, {y_start}:{y_end} {z_start}:{z_end}")
        size = (x_end-x_start, y_end-y_start, z_end-z_start)
        # get bytes array from Knossos and reshape it to the according required size
        bytes_array = kd.load_seg(offset=(x_start, y_start, z_start), size=size, mag=8)     # Knossos returns zyx arrays
        bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz
        # check for boundary chunk ends for volume
        x_start = z[0] * 128
        y_start = z[1] * 128
        z_start = z[2] * 128

        x_end = x_start + 128 if (x_start + 128) < 3389 else 3389
        y_end = y_start + 128 if (y_start + 128) < 3418 else 3418
        z_end = z_start + 128 if (z_start + 128) < 1936 else 1936
        vol8[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

    except IOError as err:
      errno, strerror = err.args
      print ('I/O error({0}): {1}'.format(errno, strerror))
      print (err)
    except:
      print ('Unexpected error:', sys.exc_info())
      raise

with ProcessPoolExecutor(max_workers=18) as executor:
    executor.map(process, to_upload)

vol8[0:1000,0:1000,0:500].viewer(port=8000)
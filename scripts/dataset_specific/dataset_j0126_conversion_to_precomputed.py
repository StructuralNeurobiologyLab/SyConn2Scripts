import os, sys
import numpy as np
from syconn import global_params
from knossos_utils import KnossosDataset

from cloudvolume import CloudVolume
from concurrent.futures import ProcessPoolExecutor

kd = KnossosDataset('/wholebrain/songbird/j0126/areaxfs_v5/knossosdatasets/')

volume_info1 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "1_1_1",
                "size": [10664, 10913,  5700],
                "resolution": [10, 10, 20],
                "chunk_sizes": [[1024, 1024, 1024]],
                "encoding": "raw",
                "factor": (1,1,1),
            }
        ]
    }

volume_info2 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "2_2_2",
                "size": [5332, 5456, 2850],
                "resolution": [20, 20, 40],
                "chunk_sizes": [[512, 512, 512]],
                "encoding": "raw",
                "factor": (2,2,2),
            }
        ]
    }

volume_info4 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "4_4_4",
                "size": [2666, 2728, 1425],
                "resolution": [40, 40, 80],
                "chunk_sizes": [[256, 256, 256]],
                "encoding": "raw",
                "factor": (4,4,4),
            }
        ]
    }

volume_info8 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "8_8_8",
                "size": [1333, 1364,  712],
                "resolution": [80, 80, 160],
                "chunk_sizes": [[128, 128, 128]],
                "encoding": "raw",
                "factor": (8,8,8),
            }
        ]
    }

volume_info16 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "16_16_16",
                "size": [666, 682, 356],
                "resolution": [160, 160, 320],
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "raw",
                "factor": (16,16,16),
            }
        ]
    }

volume_info32 = {
        "type": "image",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "32_32_32",
                "size": [333, 341, 178],
                "resolution": [320, 320, 640],
                "chunk_sizes": [[32, 32, 32]],
                "encoding": "raw",
                "factor": (32,32,32),
            }
        ]
    }

try:
  vol1 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag1/', info=volume_info1, bounded=False, non_aligned_writes=True)
  vol1.provenance.description = 'Whole j0251 dataset conversion mag 1'
  vol1.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol1.commit_info()  # generates file://.../info json file
  vol1.commit_provenance()  # generates file://.../provenance json file

  vol2 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag2/', info=volume_info2, bounded=False, non_aligned_writes=True)
  vol2.provenance.description = 'Whole j0251 dataset conversion mag 2'
  vol2.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol2.commit_info()  # generates file://.../info json file
  vol2.commit_provenance()  # generates file://.../provenance json file

  vol4 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag4/', info=volume_info4, bounded=False, non_aligned_writes=True)
  vol4.provenance.description = 'Whole j0251 dataset conversion mag 4'
  vol4.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol4.commit_info()  # generates file://.../info json file
  vol4.commit_provenance()  # generates file://.../provenance json file

  vol8 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag8/', info=volume_info8, bounded=False, non_aligned_writes=True)
  vol8.provenance.description = 'Whole j0251 dataset conversion mag 8'
  vol8.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol8.commit_info()  # generates file://.../info json file
  vol8.commit_provenance()  # generates file://.../provenance json file

  vol16 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag16/', info=volume_info16, bounded=False, non_aligned_writes=True)
  vol16.provenance.description = 'Whole j0251 dataset conversion mag 16'
  vol16.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol16.commit_info()  # generates file://.../info json file
  vol16.commit_provenance()  # generates file://.../provenance json file

  vol32 = CloudVolume('file:///wholebrain/scratch/amancu/j0126/j0126_precomputed_mag32/', info=volume_info32, bounded=False, non_aligned_writes=True)
  vol32.provenance.description = 'Whole j0251 dataset conversion mag 32'
  vol32.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses
  vol32.commit_info()  # generates file://.../info json file
  vol32.commit_provenance()  # generates file://.../provenance json file

  to_upload = []

  # split dataset
  num_x, num_y, num_z = kd.boundary[0]//1024, kd.boundary[1]//1024, kd.boundary[2]//1024
  print(f'Splitting dataset into number of chunks on each xyz-axis: {num_x}, {num_y}, {num_z}')

  for x in range(num_x+1):          # +1 for the boundary chunks
    for y in range(num_y+1):
      for z in range(num_z+1):
        to_upload.append((x,y,z))

except IOError as err:
  errno, strerror = err.args
  print ('I/O error({0}): {1}'.format(errno, strerror))
  print (err)
except ValueError as ve:
  print ('Could not convert data to an integer.')
  print (ve) 
except:
  print ('Unexpected error:', sys.exc_info()[0])
  raise

def process(z):
    # kd = KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2/")
    global kd
    try:

      ####### MAG 1 ########################################################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = x_start + 1024 if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 1024 if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 1024 if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]

      print(f"Computing chunk {x_start}:{x_end}, {y_start}:{y_end} {z_start}:{z_end}")

      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=1)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                      # puts it in xyz
      
      vol1[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

      ####### MAG 2 ########################################################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = x_start + 1024 if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 1024 if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 1024 if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]
      
      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=2)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                      # puts it in xyz
      
      # check for boundary chunk ends for volume
      x_start = z[0] * 512
      y_start = z[1] * 512
      z_start = z[2] * 512

      x_end = x_start + 512 if (x_start + 512) < 5332 else 5332
      y_end = y_start + 512 if (y_start + 512) < 5456 else 5456
      z_end = z_start + 512 if (z_start + 512) < 2850 else 2850
      
      vol2[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

      ####### MAG 4 #######################################################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = x_start + 1024 if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 1024 if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 1024 if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]

      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=4)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz

      # check for boundary chunk ends for volume
      x_start = z[0] * 256
      y_start = z[1] * 256
      z_start = z[2] * 256

      x_end = x_start + 256 if (x_start + 256) < 2666 else 2666
      y_end = y_start + 256 if (y_start + 256) < 2728 else 2728
      z_end = z_start + 256 if (z_start + 256) < 1425 else 1425
      
      vol4[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array
      
      ####### MAG 8 ####################################################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = (x_start + 1024) if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = (y_start + 1024) if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = (z_start + 1024) if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]

      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=8)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz

      # check for boundary chunk ends for volume
      x_start = z[0] * 128
      y_start = z[1] * 128
      z_start = z[2] * 128

      x_end = x_start + 128 if (x_start + 128) < 1333 else 1333
      y_end = y_start + 128 if (y_start + 128) < 1364 else 1364
      z_end = z_start + 128 if (z_start + 128) < 712 else 712
      
      vol8[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array
      ####### MAG 16 ##################################################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = x_start + 1024 if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 1024 if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 1024 if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]
      
      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=16)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz
      
      # check for boundary chunk ends for volume
      x_start = z[0] * 64
      y_start = z[1] * 64
      z_start = z[2] * 64

      x_end = x_start + 64 if (x_start + 64) < 666 else 666
      y_end = y_start + 64 if (y_start + 64) < 682 else 682
      z_end = z_start + 64 if (z_start + 64) < 356 else 356
      
      vol16[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

      ####### MAG 32 ###########################################################################################

      x_start = z[0] * 1024
      y_start = z[1] * 1024
      z_start = z[2] * 1024

      # check for boundary chunk ends for knossos
      x_end = x_start + 1024 if (x_start + 1024) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 1024 if (y_start + 1024) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 1024 if (z_start + 1024) < kd.boundary[2] else kd.boundary[2]

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=32)     # Knossos returns zyx arrays
      bytes_array = np.swapaxes(bytes_array,0,2)                                          # puts it in xyz
      
      # check for boundary chunk ends for volume
      x_start = z[0] * 32
      y_start = z[1] * 32
      z_start = z[2] * 32

      x_end = x_start + 32 if (x_start + 32) < 333 else 333
      y_end = y_start + 32 if (y_start + 32) < 341 else 341
      z_end = z_start + 32 if (z_start + 32) < 178 else 178
      
      vol32[x_start:x_end, y_start:y_end, z_start:z_end] = bytes_array

    except IOError as err:
      errno, strerror = err.args
      print ('I/O error({0}): {1}'.format(errno, strerror))
      print (err)
    except:
      print ('Unexpected error:', sys.exc_info())
      raise

with ProcessPoolExecutor(max_workers=20) as executor:
    executor.map(process, to_upload)
 
# vol.viewer(port=8080)
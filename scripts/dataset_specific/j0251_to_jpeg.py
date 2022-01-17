from knossos_utils import KnossosDataset
import knossos_cuber.knossos_cuber as kc
import os
import time
import glob
import multiprocessing as mp
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import fadvise
    FADVISE_AVAILABLE = True
except ImportError:
    FADVISE_AVAILABLE = False


path = "/wholebrain/songbird/j0251/j0251_72_clahe2"
dataset = KnossosDataset(path)

mag_files = glob.glob(path+ f'/mag*')
list_of_all_cubes = []

for mag in mag_files:
    all_cubes=[]
    ref_time = time.time()
    for root, _, files in os.walk(mag):
        #if len(files) > 1:
            #print
            # either overlay cubes or different compressions found
        for cur_file in files:
            if '.raw' in cur_file:
                all_cubes.append(os.path.join(root, cur_file))
    list_of_all_cubes.extend(all_cubes)
    print("Cube listing took: {0} s".format(time.time()-ref_time))
    # print(f'{list_of_all_cubes}')

compress_job_infos = []
for cube_path in list_of_all_cubes:
    this_job_info = kc.CompressionJobInfo()

    this_job_info.cube_edge_len = 128
    this_job_info.compressor = 'jpeg', 
    this_job_info.quality_or_ratio = 80
    this_job_info.pre_gauss = 0.5
    this_job_info.src_cube_path = cube_path

    compress_job_infos.append(this_job_info)


log_queue = mp.Queue()

worker_pool = mp.Pool(20,
                    initializer=kc.compress_cube_init,
                    initargs=[log_queue])
# distribute cubes to worker pool

async_result = worker_pool.map_async(kc.compress_cube,
                                        compress_job_infos,
                                        chunksize=10)

worker_pool.close()

while not async_result.ready():
    log_output = log_queue.get()
    print(log_output)

worker_pool.join()

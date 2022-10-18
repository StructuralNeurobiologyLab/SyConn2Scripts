from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
import syconn.reps.segmentation_helper as helper
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

global_params.wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/'
ssd = SuperSegmentationDataset()
ssv_ids = ssd.ssv_ids
n_proc = 150

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

def process(slice, lis):
    global ssv_ids
    len_slice = len(ssv_ids[slice])
    for i, ssv_id in enumerate(ssv_ids[slice]):
        ssv = ssd.get_super_segmentation_object(int(ssv_id))

        if i % 100 == 0:
            print(f'Computing... {i} of slice of {len_slice}')            

        ssv.load_attr_dict()
        mesh = ssv.mesh
        vert_num = np.array(mesh[1], dtype=np.float32).reshape(-1, 3).shape[0]

        if vert_num < 1e+3:
            lis[0] += 1
        elif vert_num >= 1e+3 and vert_num < 1e+4:
            lis[1] += 1
        elif vert_num >= 1e+4 and vert_num < 1e+5:
            lis[2] += 1
        elif vert_num >= 1e+5 and vert_num < 1e+6:
            lis[3] += 1
        elif vert_num >= 1e+6 and vert_num < 1e+7:
            lis[4] += 1
        elif vert_num >= 1e+7 and vert_num < 1e+8:
            lis[5] += 1
        else:                   # >100M
            lis[6] += 1

    return lis

with Manager() as mng:
    lis = mng.list([0,0,0,0,0,0,0])
    proc_params = [(x,lis) for x in proc_slices]
    running_tasks = [mp.Process(target=process, args=param) for param in proc_params]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

    print(f'data: {[x for x in lis]}')

# results data: [4933630, 1734817, 129923, 13113, 684, 11, 0]
import numpy as np
import multiprocessing
import time
from PIL import Image
from cloudvolume import CloudVolume
# from cloudvolume.multilod import UnshardedMultiLevelPrecomputedMeshSource
from concurrent.futures import ProcessPoolExecutor
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from tqdm import tqdm
from multiprocessing import Pool, Value

class PoolProgress:
  def __init__(self,pool,update_interval=3):
    self.pool            = pool
    self.update_interval = update_interval
  def track(self, job):
    task = self.pool._cache[job._job]
    while task._number_left>0:
      print("Tasks remaining = {0}".format(task._number_left*task._chunksize))
      time.sleep(self.update_interval)

def init(args):
    ''' store the counter for later use '''
    global counter, cv
    counter = args

def analyze_data(id):
    global counter
    cloudpath = 'file:///ssdscratch/songbird/j0251/agglo2_meshes/sv_multilod_1/'
    cv = CloudVolume(cloudpath, parallel=True, progress=False)
    try:
        mesh = cv.mesh.get(id)
    except:
        print(f'Exception for cell {id}')
        with counter.get_lock():
            counter.value += 1
        return id
    return 0

if __name__ == "__main__":
    global_params.wd = '/ssdscratch/pschuber/songbird/j0251/j0251_72_seg_20210127_agglo2/'
    ssd = SuperSegmentationDataset()
    ssv_ids = list(ssd.ssv_ids)
    give = ssv_ids
    # print(f'{give}')

    total_ssvs = len(ssv_ids)
    counter = Value('i', 0)

    p = Pool(initializer = init, initargs = (counter, ))
    pp  = PoolProgress(p)
    i = p.map_async(analyze_data, give, chunksize=1)
    pp.track(i)
    i.wait()
    ids = np.array(i.get(), dtype=np.uint64)
    print(f'ids: {ids[np.nonzero(ids)[0]]}')

    with open('/wholebrain/u/amancu/faulty_SVs_1.npy', 'wb') as f:
        np.save(f, ids[np.nonzero(ids)[0]])

    print(f'cnt: {counter.value}')
    faulty_ssvs = counter.value

    print(f'{faulty_ssvs} faulty from {total_ssvs} \nAmounts to {faulty_ssvs * 100 / total_ssvs}% of total meshes.')
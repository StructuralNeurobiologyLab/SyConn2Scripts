import pickle
from syconn.extraction.find_object_properties_C import map_subcell_C
from concurrent.futures import ProcessPoolExecutor
import time
from knossos_utils import knossosdataset as kd
import tempfile
import glob
import numpy as np
from collections import defaultdict

tmpdir = tempfile.TemporaryDirectory()
tmpdir_path = tmpdir.name

path_knossos_conf_seg1 = '/mnt/wholebrain/songbird/j0251/segmentation/j0251_rag_flat_Jan2019_seg/knossos.pyk.conf'
path_knossos_conf_seg2 = '/mnt/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf'
out_path = '/mnt/ssdscratch/songbird/j0251/j0251_rag_flat_Jan2019_seg_TO_j0251_72_seg_20210127_agglo2_SSVID_MAP.pickle'


# create chunk jobs for old seg and new seg, with per-job IDs;
# each job writes out the map for its ID into a single folder
kd1 = kd.KnossosDataset()
kd2 = kd.KnossosDataset()

kd1.initialize_from_conf(path_knossos_conf_seg1)
kd2.initialize_from_conf(path_knossos_conf_seg2)
# split dataset

start = time.time()

chunk_size = 512
num_x, num_y, num_z = kd1.boundary[0] // chunk_size, kd1.boundary[1] // chunk_size, kd1.boundary[2] // chunk_size
print(f'Splitting dataset into number of chunks on each xyz-axis: {num_x}, {num_y}, {num_z}')
job_chunk_params = []

job_id = 0
for x in range(num_x):
    for y in range(num_y):
        for z in range(num_z):
            job_chunk_params.append((job_id, x * chunk_size, y * chunk_size, z * chunk_size))
            job_id += 1

print(f'Created in total {job_id} jobs.')


def process(param):

    kd1 = kd.KnossosDataset()
    kd2 = kd.KnossosDataset()

    kd1.initialize_from_conf(path_knossos_conf_seg1)
    kd2.initialize_from_conf(path_knossos_conf_seg2)

    kd1.show_progress = False
    kd2.show_progress = False

    try:
        this_j_time = time.time()
        job_id = param[0]
        x_start = param[1]
        y_start = param[2]
        z_start = param[3]

        x_end = x_start + 512 if (x_start + 512) < kd1.boundary[0] else kd1.boundary[0]
        y_end = y_start + 512 if (y_start + 512) < kd1.boundary[1] else kd1.boundary[1]
        z_end = z_start + 512 if (z_start + 512) < kd1.boundary[2] else kd1.boundary[2]

        size = (x_end - x_start, y_end - y_start, z_end - z_start)
        #print(f'(x_start, y_start, z_start) {(x_start, y_start, z_start)}')
        # get bytes array from Knossos and reshape it to the according required size
        seg_1 = kd1.load_seg(offset=(x_start, y_start, z_start), size=size, mag=1)
        seg_2 = kd2.load_seg(offset=(x_start, y_start, z_start), size=size, mag=1)

        this_chunk_map = map_subcell_C(seg_2, seg_1[None,])

        # pickle to temp out folder tmpdir_path
        with open(tmpdir_path + f'/{job_id}_dict.pickle', 'wb') as pkl_file:
            pickle.dump(this_chunk_map, pkl_file)

        print(f'job {job_id} took: {time.time() - this_j_time} s.')

    except Exception as e:
        print(e)
        print(f'job {job_id} failed')


with ProcessPoolExecutor(max_workers=100) as executor:
    executor.map(process, job_chunk_params)

print(f'Done with all chunk jobs, took {(time.time()-start)/3600.} h. Combining dicts now into global dict.')

start = time.time()
all_chunk_dict_paths = glob.glob(tmpdir_path + '/*.pickle')

# load all dicts into mem
global_map_SSV_IDs_cnt = defaultdict(int)
for chunk_dict_path in all_chunk_dict_paths:
    with open(chunk_dict_path, 'rb') as pkl_file:
        chunk_dict = pickle.load(pkl_file)[0]

        # convert dict of dict into flat dict struct
        for k_seg_ID_1 in chunk_dict.keys():
            for k_seg_ID_2 in chunk_dict[k_seg_ID_1].keys():
                global_map_SSV_IDs_cnt[(k_seg_ID_1,k_seg_ID_2)] += chunk_dict[k_seg_ID_1][k_seg_ID_2]

# keep only seg1->seg2 ID with largest count in seg2
global_map_SSV_to_SSV = dict()
global_map_SSV_to_SSV_max_cnt = defaultdict(int)
for k_k in global_map_SSV_IDs_cnt.keys():
    if global_map_SSV_IDs_cnt[k_k] >= global_map_SSV_to_SSV_max_cnt[k_k[0]]:
        global_map_SSV_to_SSV[k_k[0]] = k_k[1]
        global_map_SSV_to_SSV_max_cnt[k_k[0]] = global_map_SSV_IDs_cnt[k_k]

with open(out_path, 'wb') as pkl_file:
    pickle.dump(global_map_SSV_to_SSV, pkl_file)


tmpdir.cleanup()
print(f'Done with global dict creation and tmp dir cleanup, took {(time.time()-start)/3600.} h')
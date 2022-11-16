#uses parts from syconn.exec.syns.run_cs_ssv_generation
#frisat part of function, that generates cs_ssv already run
#generates cs_ssv segmentation dataset from directory that includes cs_ssv_attr_dicts
import numpy as np
from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis
from syconn.handler.basics import load_pkl2obj

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
f_name = f'{global_params.wd}/cs_ssv_0'

log = initialize_logging('221115 generating cs_ssv sd', log_dir=f_name + '/logs/')
log.info("Generated from attr_dicts from 221025 cs_ssv generation")
log.info('Only collecting files again, not running _data_analysis_thread')

sd_cs_ssv = SegmentationDataset("cs_ssv", working_dir=global_params.config.working_dir)

dataset_analysis(sd_cs_ssv, compute_meshprops=False, recompute=False)
log.info(f'SegmentationDataset of type "cs_ssv" was generated with {len(sd_cs_ssv.ids)} '
             f'objects.')

log.info('Save excluded cs pairs as .npy')
storage_dir = f_name + '/so_storage_10000/'

excluded_ssv_id_pairs = []
for j in range(100):
    for i in range(100):
        if i < 10:
            i = '0' + str(i)
        if j < 10:
            j = '0' + str(j)
        excl_pairs = load_pkl2obj(f'{storage_dir}{i}/{j}/excluded_ssv_id_pairs.pkl')
        excluded_ssv_id_pairs.append(excl_pairs)

excluded_pairs = np.concatenate(excluded_ssv_id_pairs)
np.save(file=f'{f_name}/excluded_ssv_id_pairs.pkl', arr=excluded_pairs)

log.info(('Excluded ssv id pairs saved'))
del sd_cs_ssv
#uses parts from syconn.exec.syns.run_cs_ssv_generation
#frisat part of function, that generates cs_ssv already run
#generates cs_ssv segmentation dataset from directory that includes cs_ssv_attr_dicts

from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
f_name = f'{global_params.wd}/cs_ssv_0'

log = initialize_logging('221111 generating cs_ssv sd', log_dir=f_name + '/logs/')
log.info("Generated from attr_dicts from 221025 cs_ssv generation")

sd_cs_ssv = SegmentationDataset("cs_ssv", working_dir=global_params.config.working_dir)

dataset_analysis(sd_cs_ssv, compute_meshprops=False, recompute=False, add_npy_param=['excluded_ssv_id_pairs'])
log.info(f'SegmentationDataset of type "cs_ssv" was generated with {len(sd_cs_ssv.ids)} '
             f'objects.')
del sd_cs_ssv
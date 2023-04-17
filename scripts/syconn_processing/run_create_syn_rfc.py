#small script to run create_syn_rfc with updated ground truth

from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.extraction.cs_processing_steps import create_syn_rfc
from syconn.reps.segmentation import SegmentationDataset

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

#added 8 synapses from GPe/i terminals as they had low synapse probability in agglo2 (#300 - 307)
path2gt = f'{global_params.wd}/syn_rfc/230417_julian_triblecheck_ar.xls'
rfc_out = f'{global_params.wd}/syn_rfc/'

sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=global_params.wd)

create_syn_rfc(sd_syn_ssv=sd_syn_ssv, path2file=path2gt, rfc_path_out=rfc_out)
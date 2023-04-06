#run celltype inference

from syconn import global_params
from syconn.exec.exec_inference import run_celltype_prediction

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"


run_celltype_prediction(exclude_nodes=['cajalg002', 'cajalg003', 'cajalg004', 'cajalg005', 'cajalg006', 'cajalg007', 'cajalg008', 'cajalg009',
                                              'cajalg010', 'cajalg011', 'cajalg012', 'cajalg013', 'cajalg014', 'cajalg015'])
#call exec_syn run_syn_generation
#generates syn_ssv dataset but here with cs_dataset already existing


from syconn import global_params
from syconn.exec.exec_syns import  run_syn_generation

#taken from start_j0251_organelles with labels from current wd
cellorganelle_transf_funcs = dict(sj=lambda x: (x == 1).astype('u1'),
                                      vc=lambda x: (x == 2).astype('u1'),
                                      mi=lambda x: (x == 3).astype('u1'))

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

run_syn_generation(transf_func_sj_seg=cellorganelle_transf_funcs['sj'], exclude_nodes=['cajalg002', 'cajalg003', 'cajalg004', 'cajalg005', 'cajalg006', 'cajalg007', 'cajalg008', 'cajalg009',
                                              'cajalg010', 'cajalg011', 'cajalg012', 'cajalg013', 'cajalg014', 'cajalg015'])
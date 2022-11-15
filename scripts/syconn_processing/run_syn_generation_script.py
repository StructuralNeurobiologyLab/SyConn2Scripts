#call exec_syn run_syn_generation
#generates syn_ssv dataset but here with cs_dataset already existing


from syconn import global_params
from syconn.exec.exec_syns import  run_syn_generation

#set trans_func_sj

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

run_syn_generation(transf_func_sj_seg=)
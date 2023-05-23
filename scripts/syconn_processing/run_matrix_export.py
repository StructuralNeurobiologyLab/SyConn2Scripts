#run matrix export which includes mapping attributes from ssv to syn objects
#stores them in cache arrays

from syconn.exec.exec_syns import run_matrix_export
from syconn import global_params
from syconn.handler.config import initialize_logging

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

f_name = f'{global_params.wd}/syn_ssv_0'

log = initialize_logging('230504 matrix export', log_dir=f_name + '/logs/')
log.info(f'start matrix export, no ssd.save_dataset_deep')
log.info(f'Matrix export done with synapses from this wd, updated axoness_avg10000 prediction and same spiness prediction as agglo2')

run_matrix_export(exclude_nodes=['cajalg002', 'cajalg003', 'cajalg004', 'cajalg005', 'cajalg006', 'cajalg007',
                                  'cajalg008', 'cajalg009',
                                  'cajalg010'])

log.info('matrix export finished generation finished')

#script to rewrite spiness vertex labels in all cells in v5

def rewrite_spiness_vertex_label_cell(cellid):
    '''
    Overwrite vertex labels for spiness from old to new working directory
    Args:
        cellid: id of the cell

    Returns:

    '''
    old_wd = 'cajal/nvmescratch/projects/from_ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2'
    new_wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    cell_old = SuperSegmentationObject(cellid, working_dir=old_wd)
    cell_new = SuperSegmentationObject(cellid, working_dir=new_wd)
    old_ld = cell_old.label_dict('vertex')
    old_ld_spiness = old_ld['spiness']
    new_ld = cell_new.label_dict('vertex')
    new_ld['spiness'] = old_ld_spiness
    new_ld.push()
    #test now if overwriting worked
    cell_new_test = SuperSegmentationObject(cellid, working_dir=new_wd)
    test_spiness_labels = cell_new_test.label_dict('vertex')['spiness']
    assert(np.all(test_spiness_labels == old_ld_spiness))

if __name__ == '__main__':
    from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
    import numpy as np
    from syconn import global_params
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.handler.config import initialize_logging

    global_params.wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    log = initialize_logging('overwrite vertex spiness labels', log_dir=global_params.wd + '/logs/')
    log.info('Overwrite current vertex spiness labels with ones from agglo2')
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    cellids = ssd.ssv_ids
    log.info('Start overwriting all vertex spiness labels now')
    start_multiprocess_imap(rewrite_spiness_vertex_label_cell, cellids)
    log.info('All vertex spiness labels are overwritten')






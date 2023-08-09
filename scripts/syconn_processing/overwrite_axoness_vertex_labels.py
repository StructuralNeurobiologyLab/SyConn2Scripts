#script to rewrite axoness vertex labels in all cells in v5

def rewrite_axoness_vertex_label_cell(cellid):
    '''
    Overwrite vertex labels for axoness from old to new working directory
    Args:
        cellid: id of the cell

    Returns:

    '''
    old_wd = 'cajal/nvmescratch/projects/from_ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2'
    new_wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    cell_old = SuperSegmentationObject(cellid, working_dir=old_wd)
    cell_new = SuperSegmentationObject(cellid, working_dir=new_wd)
    old_ld = cell_old.label_dict('vertex')
    old_ld_axoness = old_ld['axoness']
    new_ld = cell_new.label_dict('vertex')
    new_ld['axoness'] = old_ld_axoness
    new_ld.push()
    #test now if overwriting worked
    cell_new_test = SuperSegmentationObject(cellid, working_dir=new_wd)
    test_axoness_labels = cell_new_test.label_dict('vertex')['axoness']
    assert(np.all(test_axoness_labels == old_ld_axoness))

if __name__ == '__main__':
    from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
    import numpy as np
    from syconn import global_params
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.handler.config import initialize_logging

    global_params.wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    log = initialize_logging('overwrite vertex axoness labels', log_dir=global_params.wd + '/logs/')
    log.info('Overwrite current vertex axoness labels with ones from agglo2')
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    cellids = ssd.ssv_ids
    log.info('Start overwriting all vertex axoness labels now')
    start_multiprocess_imap(rewrite_axoness_vertex_label_cell, cellids)
    log.info('All vertex axoness labels are overwritten')

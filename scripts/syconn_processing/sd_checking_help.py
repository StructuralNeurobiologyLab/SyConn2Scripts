#check sd generation dataset

from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
import numpy as np
from scipy.spatial import KDTree

def check_min_max_mapping_ratios(cellid, cell_object_ids, obj_type, wd):
    '''
    Checks min and max of mapping ratio of one cell for specific obj_type
    Args:
        cellid: id of cell
        cell_object_ids: obj ids of cells
        obj_type: 'mi', 'vc', 'sj'
        wd: working directory

    Returns: min, max values of mapping ratios

    '''
    sd = SegmentationDataset(obj_type, working_dir=wd)
    obj_ids = sd.ids
    #mapping ids and mapping ratios is an array full of lists which can contain more than one cell
    obj_mapping_ids = sd.load_numpy_data('mapping_ids')
    obj_mapping_ratios = sd.load_numpy_data('mapping_ratios')
    cell_inds = np.in1d(obj_ids, cell_object_ids)
    cell_obj_mapping = obj_mapping_ids[cell_inds]
    cell_obj_ratios = obj_mapping_ratios[cell_inds]

    #get ratios only of this cell
    cell_obj_mapping_con = np.concatenate(cell_obj_mapping)
    cell_obj_ratios_con = np.concatenate(cell_obj_ratios)
    cell_con_inds = np.in1d(cell_obj_mapping_con, cellid)
    cell_obj_ratios_con_id = cell_obj_ratios_con[cell_con_inds]

    #check limits of mapping to see if in agreement with boundaries set:
    min_mapping = np.min(cell_obj_ratios_con_id)
    max_mapping = np.max(cell_obj_ratios_con_id)
    return min_mapping, max_mapping

global_params.wd = "cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

cellid = 10965555
#for cehcking mitos get all mitos from cell
cell = SuperSegmentationObject(cellid)
cell_mi_ids = cell.mi_ids

min_mapping_mi, max_mapping_mi = check_min_max_mapping_ratios(cellid = cellid, cell_object_ids=cell_mi_ids, obj_type='mi', wd = global_params.wd)
#check if coordinate close to mito in mito, use KDtree with mito_coords
sd = SegmentationDataset('mi', working_dir=global_params.wd)
mi_ids = sd.ids
#mapping ids and mapping ratios is an array full of lists which can contain more than one cell
mi_mapping_ids = sd.load_numpy_data('mapping_ids')
mi_mapping_ratios = sd.load_numpy_data('mapping_ratios')
mi_rep_coords = sd.load_numpy_data('rep_coords')
cell_ind = np.in1d(mi_ids, cell_mi_ids)
cell_mi_coords = mi_rep_coords[cell_ind]
reord_mi_ids = mi_ids[cell_ind]
mi_tree = KDTree(mi_rep_coords * cell.scaling)
#mi rep coords usually (lower) left corner, check afterwards visually if right mito found
test_coord = [22528, 21375,  8781]
test_mi_ind = mi_tree.query(test_coord * cell.scaling)[1]
test_coord_mapped = mi_rep_coords[test_mi_ind]
test_mapping = mi_mapping_ids[test_mi_ind]
test_ratio = mi_mapping_ratios[test_mi_ind]
test_id = mi_ids[test_mi_ind]
cell_mi_test_ind = np.where(reord_mi_ids == test_id)
cell_mi_test_coord = cell_mi_coords[cell_mi_test_ind]
from syconn.reps.super_segmentation import *
from syconn.reps.segmentation_helper import calc_center_of_mass
from syconn.backend.storage import VoxelStorageLazyLoading, AttributeDict
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.proc.sd_proc import dataset_analysis


def _recalc_worker(base_dir):
    voxel_dc = VoxelStorageLazyLoading(base_dir + "/voxel.npz", overwrite=False)
    attr_dc = AttributeDict(base_dir + "/attr_dict.pkl", read_only=False)
    if not os.path.isfile(base_dir + "/attr_dict.pkl"):
        print(f'Could not find attr_dict at {base_dir}')
        return
    for so_id in attr_dc.keys():
        voxel_list = voxel_dc[so_id]
        rep = calc_center_of_mass(voxel_list * sd_syn.scaling) // sd_syn.scaling
        attr_dc[so_id]['rep_coord'] = rep
    attr_dc.push()


if __name__ == '__main__':
    global_params.wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/'
    sd_syn = SegmentationDataset('syn_ssv')
    # paths = list(sd_syn.iter_so_dir_paths())
    # start_multiprocess_imap(_recalc_worker, paths, nb_cpus=20)
    dataset_analysis(sd_syn, recompute=False, compute_meshprops=False)

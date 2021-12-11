import glob
import os

import numpy as np
from knossos_utils.skeleton_utils import load_skeleton


def _load_kzip_skelnode_labels(kzip_path):
    a_obj = load_skeleton(kzip_path)

    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    elif 'skeleton' in a_obj:
        a_obj = a_obj["skeleton"]
    elif 1 in a_obj:  # use first annotation object.. OBDA
        a_obj = a_obj[1]
    else:
        raise ValueError(f'Could not find annotation skeleton in "{kzip_path}".')
    a_nodes = list(a_obj.getNodes())
    a_node_labels = [n.getComment() for n in a_nodes if len(n.getComment()) > 0]
    return a_node_labels


def print_node_support(dest_dir):
    fnames = glob.glob(dest_dir + '/*.k.zip')
    labels = []
    for fname in fnames:
        labels.extend(_load_kzip_skelnode_labels(fname))
    print(f'Node label support of {len(fnames)} files at "{dest_dir}": '
          f'{[os.path.split(fname)[1] for fname in fnames]}')
    print(np.unique(labels, return_counts=True))


if __name__ == '__main__':
    # # multi-view bouton GT
    # print('Bouton GT (multi-view)')
    # dest_dir = f'/wholebrain/songbird/j0126/GT/axgt_semseg/kzips/NEW_including_boutons/batch1_results/'
    # print_node_support(dest_dir)
    # dest_dir = f'/wholebrain/songbird/j0126/GT/axgt_semseg/kzips/NEW_including_boutons/batch2_results_v2/'
    # print_node_support(dest_dir)
    #
    # # spine GT
    # print('spine GT (point clouds)')
    # dest_dir = f'/wholebrain/songbird/j0126/GT/spgt_semseg/kzips/'
    # print_node_support(dest_dir)
    #
    # # synapse GT
    # print('synapse GT (multi-view and point clouds)')
    # dest_dir = f'/wholebrain/scratch/pschuber/cmn_paper/data/semantic_segmentation/eval/syn_gt/'
    # print_node_support(dest_dir)
    #
    # # synapse GT
    # print('spine GT (multi-view paper)')
    # dest_dir = f'/wholebrain/scratch/pschuber/cmn_paper/data/semantic_segmentation/gt_skels/'
    # print_node_support(dest_dir)

    # axgt semseg GT (test multi-view, j0126)
    print('axgt semseg GT (test multi-view, j0126)')
    dest_dir = f'/wholebrain/songbird/j0126/GT/axgt_semseg/testdata/'
    print_node_support(dest_dir)

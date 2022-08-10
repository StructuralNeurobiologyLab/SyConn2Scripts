import glob
import os

import numpy as np
from syconn.handler.basics import load_pkl2obj
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
    fnames_parsed = []
    for fname in fnames:
        try:
            labels.extend(_load_kzip_skelnode_labels(fname))
            fnames_parsed.append(fname)
        except Exception as e:
            if 'k.zip file does not contain annotation.xml' in str(e):
                print(f'\n\nWARNING: kzip {fname} without skeleton xml found - this will be skipped: {str(e)}')
    print(f'Node label support of {len(fnames_parsed)} files at "{dest_dir}": '
          f'{[os.path.split(fname)[1] for fname in fnames_parsed]}')
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

    # # axgt semseg GT (test multi-view, j0126)
    # print('axgt semseg GT (test multi-view, j0126)')
    # dest_dir = f'/wholebrain/songbird/j0126/GT/axgt_semseg/testdata/'
    # print_node_support(dest_dir)

    # # spgt vertex eval GT (test multi-view, j0126, cmn paper)
    # print('spgt vertex eval GT (test multi-view, j0126, cmn paper)')
    # dest_dir = f'/wholebrain/scratch/pschuber/cmn_paper/data/semantic_segmentation/eval/dense_gt'
    # print_node_support(dest_dir)

    # compartment GT (j0251)
    print('compartment GT (j0251, train)')
    dest_dir = "/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_12_final/train/"
    print_node_support(dest_dir)

    print('compartment GT (j0251, test)')
    dest_dir = "/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_12_final/test/"
    print_node_support(dest_dir)
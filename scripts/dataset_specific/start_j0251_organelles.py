#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" SyConn - Synaptic connectivity inference toolkit

Create segmentation datasets for organelles.

This script first performs the connected components on the 
integer-encoded segmentation data (knossos format) of organelles
['sj', 'vc', 'mi] and then creates `SegmentationDataset` objects
for each of these organelles.

Note: The organelle segmentations are encoded as follows:
sj: 1, vc: 2, mi: 3 
"""

__author__    = "Hashir Ahmad"
__authors__   = "Hashir Ahmad", "Joergen Kornfeld"
__copyright__ = "Copyright (c) 2016 - now, \
    Max Planck Institute for Biological Intelligence, in foundation, \
        Munich, Germany"
__date__      = "2022-11-15 12:03:14"
__email__     = "hashir.ahmad@bi.mpg.de"

import os
import time
import re
import glob
import argparse
import numpy as np
import networkx as nx
import pandas
import tqdm

from syconn.handler.config import generate_default_conf, initialize_logging
from syconn import global_params
from syconn.proc.stats import FileTimer
from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference, exec_skeleton


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', dest='wd', type=str, default='/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/',
                        help='Working directory of SyConn')
    parser.add_argument('--cube_offset', nargs='+', type=int, default=[0, 0, 0],
                        help='Offset of the cube in x, y, z direction')
    parser.add_argument('--cube_size', nargs='+', type=int,
                        default=[256, 256, 256], help='Size of the cube in x, y, z direction')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing data')
    parser.add_argument('--create_config', action='store_true',
                        help='Create syconn config file in the working directory')

    args = parser.parse_args()

    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    working_dir = args.wd

    experiment_name = 'j0251'
    scale = np.array([10, 10, 25])

    # TODO: generalize this, e.g. by using a config file
    shape_j0251 = np.array([27119, 27350, 15494])
    cube_size = np.array(args.cube_size).astype(np.int32)
    cube_offset = np.array(args.cube_offset).astype(np.int32)
    cube_of_interest_bb = np.array(
        [cube_offset, cube_offset + shape_j0251], dtype=np.int32)

    key_val_pairs_conf = [
        # minimum bounding box diagonal of cell (fragments) in nm
        ('min_cc_size_ssv', 2000),
        ('glia', {'prior_astrocyte_removal': False}),
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', 'SLURM'),  # 'SLURM'
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('cell_contacts',
         {'generate_cs_ssv': False,  # cs_ssv: contact site objects between cells
          'min_path_length_partners': None,
          }),
        ('nnodes_total', 17),
        ('use_point_models', True),
        ('cube_of_interest_bb', cube_of_interest_bb.tolist()),
        ('cell_contacts',
         {'generate_cs_ssv': False,  # cs_ssv: contact site objects between cells
          'min_path_length_partners': None,
          }),
        ('meshes', {'use_new_meshing': True}),
        ('views', {'use_onthefly_views': True,
                   'use_new_renderings_locs': True,
                   'view_properties': {'nb_views': 3}
                   }),
        ('slurm', {'exclude_nodes': ['wb08', 'wb09']}),
        ('cell_objects',
         {'sym_label': 1, 'asym_label': 2,
          'min_obj_vx': {'sv': 1},
          # first remove small fragments, close existing holes, then erode to trigger watershed segmentation
          'extract_morph_op': {'mi': ['binary_opening', 'binary_closing', 'binary_erosion', 'binary_erosion',
                                      'binary_erosion', 'binary_erosion'],
                               'sj': ['binary_opening', 'binary_closing'],
                               'vc': ['binary_opening', 'binary_closing', 'binary_erosion', 'binary_erosion']}
          }
         )
    ]
    chunk_size = None
    n_folders_fs = 10000
    n_folders_fs_sc = 10000

    # ----------------- DATA DIRECTORY ---------------------
    # TODO: generalize this, e.g. by using a config file
    raw_kd_path = '/wholebrain/songbird/j0251/j0251_72_clahe2/'
    root_dir = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    seg_kd_path = '/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/'
    kd_asym_path = seg_kd_path.split('segmentation')[
        :-1][0] + 'segmentation/j0251_asym_sym/'
    kd_sym_path = kd_asym_path
    syntype_avail = (kd_asym_path is not None) and (kd_sym_path is not None)
    mi_kd_path = root_dir + '/knossosdatasets/mivcsj_seg_final/'
    vc_kd_path = mi_kd_path
    sj_kd_path = mi_kd_path

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `batchjob_object_segmentation.py` (here: numpy)
    cellorganelle_transf_funcs = dict(sj=lambda x: (x == 1).astype('u1'),
                                      vc=lambda x: (x == 2).astype('u1'),
                                      mi=lambda x: (x == 3).astype('u1'))

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    ftimer = FileTimer(working_dir + '/.timing.pkl')
    ftimer.start('Preparation')

    # Preparing config
    # currently this is were SyConn looks for the neuron rag
    if global_params.wd is not None:
        log.critical('Example run started. Original working directory defined'
                     ' in `global_params.py` '
                     'is overwritten and set to "{}".'.format(working_dir))

    if args.create_config:
        # generates default config from syconn/handler/config.yml
        generate_default_conf(working_dir, scale, syntype_avail=syntype_avail, kd_seg=seg_kd_path, kd_mi=mi_kd_path,
                              kd_vc=vc_kd_path, kd_sj=sj_kd_path, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                              key_value_pairs=key_val_pairs_conf, force_overwrite=True)

    else:
        if not os.path.exists(os.path.join(working_dir, 'config.yml')):
            raise ValueError(
                'No config file found. Please run with --create_config.')

    global_params.wd = working_dir
    os.makedirs(global_params.config.temp_path, exist_ok=True)
    start = time.time()

    # check model existence
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype_e3',
                      'mpath_axonsem', 'mpath_glia_e3', 'mpath_myelin',
                      'mpath_tnet']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, working_dir))
    ftimer.stop()

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Starting SyConn pipeline for data cube (shape: {}).'.format(
        ftimer.dataset_shape))
    log.critical('Working directory is set to "{}".'.format(working_dir))

    log.info('Step 2/9 - Creating SegmentationDatasets (incl. SV meshes)')
    ftimer.start('SD generation')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
                                    n_folders_fs=n_folders_fs, cube_of_interest_bb=cube_of_interest_bb,
                                    load_cellorganelles_from_kd_overlaycubes=True,
                                    transf_func_kd_overlay=cellorganelle_transf_funcs,
                                    max_n_jobs=global_params.config.ncore_total * 4, overwrite=args.overwrite)
    ftimer.stop()

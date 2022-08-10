# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import os
import subprocess
import glob
import shutil
import sys
import time
import argparse
import networkx as nx
from knossos_utils import knossosdataset

from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn.handler.compression import load_from_h5py
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_render, exec_inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str, default='',
                        help='Working directory of SyConn')
    parser.add_argument('--example_cube', type=str, default='2',
                        help='Used toy data.')
    args = parser.parse_args()
    example_cube_id = args.example_cube

    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    if args.working_dir == "":  # by default use cube dependent working dir
        args.working_dir = "/wholebrain/songbird/j0251/j0251_syconn_test_run_2" \
                           "/cube{}_j0251/".format(example_cube_id)
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------

    kd_asym_path = None
    kd_sym_path = None
    syntype_avail = (kd_asym_path is not None) and (kd_sym_path is not None)

    # More parameters
    prior_glia_removal = True
    use_new_meshing = True
    scale = np.array([10, 10, 25])
    chunk_size = (512, 512, 512)
    # "example_cube" enables single node processing, set number of resources accordingly
    if 'example_cube' in args.working_dir:
        key_val_pairs_conf = [('ncores_per_node', 20), ('ngpus_per_node', 1),
                              ('nnodes_total', 1)]
    else:
        key_val_pairs_conf = [('ncores_per_node', 20), ('ngpus_per_node', 2),
                              ('nnodes_total', 17)]
    n_folders_fs = 1000
    n_folders_fs_sc = 1000
    experiment_name = 'j0251_example'

    # TODO: adapt for whole-dataset run
    # ----------------- DATA DIRECTORY ---------------------
    if example_cube_id == "2":
        orig_data_dir = '/wholebrain/songbird/j0251/j0251_syconn_test_run_2/'
        raw_kd_path = orig_data_dir + 'seg/'
        mi_kd_path = orig_data_dir + 'sj_vc_mi/'
        vc_kd_path = orig_data_dir + 'sj_vc_mi/'
        sj_kd_path = orig_data_dir + 'sj_vc_mi/'
    elif example_cube_id == "1":
        orig_data_dir = '/wholebrain/songbird/j0251/j0251_syconn_test_run/j0251_test_phil/'
        raw_kd_path = orig_data_dir + 'latest_seg/'
        mi_kd_path = orig_data_dir + 'latest_sj_vc_mito/'
        vc_kd_path = orig_data_dir + 'latest_sj_vc_mito/'
        sj_kd_path = orig_data_dir + 'latest_sj_vc_mito/'
    else:
        raise ValueError("Invalid cube ID (valid: 1, 2).")
    # ----------------- DATA DIRECTORY ---------------------

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `batchjob_object_segmentation.py` (here: numpy)
    cellorganelle_transf_funcs = dict(mi=lambda x: ((x == 1) * 255).astype(np.uint64),
                                      vc=lambda x: ((x == 3) * 255).astype(np.uint64),
                                      sj=lambda x: ((x == 2) * 255).astype(np.uint64))

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    example_wd = os.path.expanduser(args.working_dir)
    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']
    log.info('Step 0/8 - Preparation')

    curr_dir = os.getcwd() + '/'
    kzip_p = curr_dir + '/example_cube{}.k.zip'.format(example_cube_id)
    if os.path.isdir(curr_dir + '/models/') and not os.path.isdir(
            example_wd + '/models/'):
        shutil.copytree(curr_dir + '/models', example_wd + '/models/')
    if not os.path.isfile(kzip_p):
        raise FileNotFoundError('Example data could not be found at "{}"'
                                '.'.format(kzip_p))
    bb = parse_movement_area_from_zip(kzip_p).astype(np.int)
    bd = bb[1] - bb[0] + 1

    # Preparing config
    # currently this is were SyConn looks for the neuron rag
    if global_params.wd is not None:
        log.critical('Example run started. Original working directory defined'
                     ' in `global_params.py` '
                     'is overwritten and set to "{}".'.format(example_wd))

    generate_default_conf(
        example_wd, scale, syntype_avail=syntype_avail,
        kd_seg=raw_kd_path, kd_mi=mi_kd_path,
        kd_vc=vc_kd_path, kd_sj=sj_kd_path, prior_glia_removal=prior_glia_removal,
        kd_sym=kd_sym_path, kd_asym=kd_asym_path, use_new_meshing=use_new_meshing,
        use_new_renderings_locs=True, key_value_pairs=key_val_pairs_conf)

    global_params.wd = example_wd
    os.makedirs(global_params.config.temp_path, exist_ok=True)
    start = time.time()

    # Checking models
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype',
                      'mpath_axoness', 'mpath_glia']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, example_wd))

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Finished example cube initialization (shape: {}). Starting'
             ' SyConn pipeline.'.format(bd))
    log.info('Example data will be processed in "{}".'.format(example_wd))
    time_stamps.append(time.time())
    step_idents.append('Preparation')

    log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes)')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
                                    n_folders_fs=n_folders_fs, cube_of_interest_bb=bb,
                                    load_cellorganelles_from_kd_overlaycubes=True,
                                    transf_func_kd_overlay=cellorganelle_transf_funcs)

    from syconn.reps.segmentation import SegmentationDataset
    sd = SegmentationDataset(obj_type="sv", working_dir=global_params.config.working_dir)
    rag_sub_g = nx.Graph()
    # add SV IDs to graph via self-edges
    mesh_bb = sd.load_cached_data('mesh_bb')  # N, 2, 3
    mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)
    filtered_ids = sd.ids[mesh_bb > global_params.config['glia']['min_cc_size_ssv']]
    rag_sub_g.add_edges_from([[el, el] for el in filtered_ids])
    log.info('{} SVs were added to the RAG after application of the size '
             'filter.'.format(len(filtered_ids)))
    nx.write_edgelist(rag_sub_g, global_params.config.init_rag_path)

    exec_init.run_create_rag()
    time_stamps.append(time.time())
    step_idents.append('SD generation')

    if global_params.config.prior_glia_removal:
        log.info('Step 1.5/8 - Glia separation')
        exec_render.run_glia_rendering()
        exec_inference.run_glia_prediction()
        exec_inference.run_glia_splitting()
        time_stamps.append(time.time())
        step_idents.append('Glia separation')

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    exec_init.run_create_neuron_ssd()
    time_stamps.append(time.time())
    step_idents.append('SSD generation')

    log.info('Step 3/8 - Neuron rendering')
    exec_render.run_neuron_rendering()
    time_stamps.append(time.time())
    step_idents.append('Neuron rendering')

    log.info('Step 4/8 - Synapse detection')
    exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc,
                                 cube_of_interest_bb=bb)
    time_stamps.append(time.time())
    step_idents.append('Synapse detection')

    log.info('Step 5/8 - Axon prediction')
    # # OLD
    # exec_inference.run_axoness_prediction()
    # exec_inference.run_axoness_mapping()
    exec_inference.run_semsegaxoness_prediction()
    exec_inference.run_semsegaxoness_mapping()
    time_stamps.append(time.time())
    step_idents.append('Axon prediction')

    log.info('Step 6/8 - Spine prediction')
    exec_inference.run_spiness_prediction()
    time_stamps.append(time.time())
    step_idents.append('Spine prediction')

    log.info('Step 7/9 - Morphology extraction')
    exec_inference.run_morphology_embedding()
    time_stamps.append(time.time())
    step_idents.append('Morphology extraction')

    log.info('Step 8/9 - Celltype analysis')
    exec_inference.run_celltype_prediction()
    time_stamps.append(time.time())
    step_idents.append('Celltype analysis')

    log.info('Step 9/9 - Matrix export')
    exec_syns.run_matrix_export()
    time_stamps.append(time.time())
    step_idents.append('Matrix export')

    time_stamps = np.array(time_stamps)
    dts = time_stamps[1:] - time_stamps[:-1]
    dt_tot = time_stamps[-1] - time_stamps[0]
    dt_tot_str = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dt_tot))
    time_summary_str = "\nEM data analysis of experiment '{}' finished after" \
                       " {}.\n".format(experiment_name, dt_tot_str)
    n_steps = len(step_idents[1:]) - 1
    for i in range(len(step_idents[1:])):
        step_dt = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dts[i]))
        step_dt_perc = int(dts[i] / dt_tot * 100)
        step_str = "[{}/{}] {}\t\t\t{}\t\t\t{}%\n".format(
            i, n_steps, step_idents[i+1], step_dt, step_dt_perc)
        time_summary_str += step_str
    log.info(time_summary_str)
    log.info('Setting up flask server for inspection. Annotated cell reconst'
             'ructions and wiring can be analyzed via the KNOSSOS-SyConn plugin'
             ' at `SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
    fname_server = os.path.dirname(os.path.abspath(__file__)) + \
                   '/../kplugin/server.py'
    os.system('python {} --working_dir={} --port=10002'.format(
        fname_server, example_wd))

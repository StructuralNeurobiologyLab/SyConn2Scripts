# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import networkx as nx
import numpy as np
from knossos_utils import knossosdataset
from syconn import global_params
from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.compression import save_to_h5py


if __name__ == '__main__':
    curr_dir = os.getcwd() + '/'

    raw_kd_path = curr_dir + 'seg/'
    mi_kd_path = curr_dir + 'sj_vc_mi/'
    vc_kd_path = curr_dir + 'sj_vc_mi/'
    sj_kd_path = curr_dir + 'sj_vc_mi/'
    # sym_kd_path = global_params.config.kd_sym_path
    # asym_kd_path = global_params.config.kd_asym_path

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(raw_kd_path)

    kd_co = knossosdataset.KnossosDataset()
    kd_co.initialize_from_knossos_path(mi_kd_path)

    # kd_mi = knossosdataset.KnossosDataset()
    # kd_mi.initialize_from_knossos_path(mi_kd_path)

    # kd_vc = knossosdataset.KnossosDataset()
    # kd_vc.initialize_from_knossos_path(vc_kd_path)
    #
    # kd_sj = knossosdataset.KnossosDataset()
    # kd_sj.initialize_from_knossos_path(sj_kd_path)

    # kd_sym = knossosdataset.KnossosDataset()
    # kd_sym.initialize_from_knossos_path(sym_kd_path)
    #
    # kd_asym = knossosdataset.KnossosDataset()
    # kd_asym.initialize_from_knossos_path(asym_kd_path)

    # get data
    for example_cube_id in range(2, 3):
        kzip_p = '{}/example_cube{}.k.zip'.format(curr_dir, example_cube_id)
        data_dir = "{}/data{}/".format(curr_dir, example_cube_id)
        os.makedirs(data_dir, exist_ok=True)
        bb = parse_movement_area_from_zip(kzip_p)
        print('Preparing cube "{}" of size {}.'.format(kzip_p, bb))
        raw = kd.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg = kd.from_overlaycubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg_co = kd_co.from_overlaycubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        # seg_mi = kd_mi.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        # seg_vc = kd_vc.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        # seg_sj = kd_sj.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        seg_mi = (seg_co == 1).astype(np.uint8) * 255
        seg_vc = (seg_co == 3).astype(np.uint8) * 255
        seg_sj = (seg_co == 2).astype(np.uint8) * 255
        sym = np.zeros_like(raw)  # kd_sym.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)
        asym = np.zeros_like(raw)  # kd_asym.from_raw_cubes_to_matrix(bb[1] - bb[0], bb[0], mag=1)

        # save data
        save_to_h5py([raw], data_dir + 'raw.h5', hdf5_names=['raw'])
        save_to_h5py([seg], data_dir + 'seg.h5', hdf5_names=['seg'])
        save_to_h5py([seg_mi], data_dir + 'mi.h5', hdf5_names=['mi'])
        save_to_h5py([seg_vc], data_dir + 'vc.h5', hdf5_names=['vc'])
        save_to_h5py([seg_sj], data_dir + 'sj.h5', hdf5_names=['sj'])
        save_to_h5py([sym], data_dir + 'sym.h5', hdf5_names=['sym'])
        save_to_h5py([asym], data_dir + 'asym.h5', hdf5_names=['asym'])

        # store subgraph of SV-agglomeration
        # g_p = "{}/glia/neuron_rag.bz2".format(global_params.config.working_dir)
        # rag_g = nx.read_edgelist(g_p, nodetype=np.uint)
        # sv_ids = np.unique(seg)
        # rag_sub_g = rag_g.subgraph(sv_ids)
        # os.makedirs(data_dir, exist_ok=True)
        # print('Writing subgraph within {} and {} SVs.'.format(
        #     bb, len(sv_ids)))

        rag_sub_g = nx.Graph()
        # add flattened SV agglomeration via self-edges
        rag_sub_g.add_edges_from([[el, el] for el in np.unique(seg)])
        rag_sub_g.remove_node(0)

        nx.write_edgelist(rag_sub_g, data_dir + "/neuron_rag.bz2")




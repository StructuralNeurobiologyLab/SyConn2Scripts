#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" SyConn - Synaptic connectivity inference toolkit

Convert integer-encoded segmentation data to knossos format. 

This script is intended to be used with the tensorstore backend.

Note: The cube shape is fixed to 256^3 and has to be explicitly set in
the knossos dataset initialization.
"""

__author__    = "Hashir Ahmad"
__authors__   = "Hashir Ahmad", "Joergen Kornfeld"
__copyright__ = "Copyright (c) 2016 - now, \
    Max Planck Institute for Biological Intelligence, in foundation, \
        Munich, Germany"
__date__      = "2022-11-15 12:16:52"
__email__     = "hashir.ahmad@bi.mpg.de"

import os
import asyncio
import errno
import multiprocessing as mp
import time
import h5py
import numpy as np
import knossos_utils as ku
import tensorstore as ts
import logging

from queue import Queue
from threading import Thread
from cmath import exp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from knossos_utils import knossosdataset as kds


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def write_in_kd(c):
    """
    Iterate over the knossos dataset and write the segmentation from the given
    source into it.

    :param c: configuration dict
    :return:
    """

    stack = ku.knossosdataset.KnossosDataset()
    stack.initialize_from_knossos_path(c['kd_path'])
    # set cube shape explicitly, knossos supports 128^3 only
    stack._cube_shape = c['cubesize'].astype(np.int32)

    # clean all overlay cubes first
    #print('Starting to delete all existing overlay cubes')
    # stack.delete_all_overlaycubes(nb_processes=10)
    #print('Done deleting all existing overlay cubes.')

    dataset_size = stack._boundary

    totalstart = time.time()
    for x in range(0, dataset_size[0], c['cubesize'][0]):
        startx = time.time()
        print("x", x)
        for y in range(0, dataset_size[1], c['cubesize'][1]):
            starty = time.time()
            print("     y", y)
            for z in range(0, dataset_size[2], c['cubesize'][2]):
                startz = time.time()
                print("          z", z)

                offset = np.array([x, y, z])

                if c['source'] == 'hdf5':
                    sc = read_from_h5file(c['h5file'],
                                          offset,
                                          cubesize=c['cubesize'],
                                          h5name=c['h5name'])

                elif c['source'] == 'brainmaps':
                    sc = read_from_brainmaps(c, offset)

                stack.from_matrix_to_cubes(offset,
                                           mags=c['mags'],
                                           data=sc,
                                           fast_downsampling=True,
                                           verbose=False,
                                           nb_threads=c['n_output_threads'],
                                           overwrite=False)

                print("            --- took", time.time()-startz)
            print("       --- took", time.time()-starty)
        print("  --- took", time.time()-startx)
    print("Total time: ", time.time()-totalstart)

    return


def write_in_kd_parallel():
    """
    Written with producer / consumer queues, based on threads.

    :param config:
    :return:
    """
    start = time.time()

    # generate all offsets for parallel download and
    # push all offsets on to downloader queue
    offsets = []
    if not config['subvol_size']:
        config['subvol_size'] = stack._boundary
        config['subvol_offset'] = (0, 0, 0)

    for x in range(config['subvol_offset'][0], config['subvol_offset'][0]+config['subvol_size'][0], config['cubesize'][0]):
        for y in range(config['subvol_offset'][1], config['subvol_offset'][1]+config['subvol_size'][1], config['cubesize'][1]):
            for z in range(config['subvol_offset'][2], config['subvol_offset'][2]+config['subvol_size'][2], config['cubesize'][2]):
                offset = np.array([x, y, z])
                offsets.append(offset)

    logging.info(
        'Done with generating download offsets, total {0}'.format(len(offsets)))

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(download_write_worker, offsets)

    logging.info('Done with downloading and writing. Took {0}h'.format(
        (time.time()-start)/3600.))

    return


def download_write_worker(offset):
    start_worker = time.time()

    test_path = config['kd_path'] + '/mag1/' + 'x{0:04d}/y{1:04d}/z{2:04d}/j0251_realigned_mag1_x{0:04d}_y{1:04d}_z{2:04d}.seg.sz.zip'.format(
        offset[0]//256, offset[1]//256, offset[2]//256)
    if os.path.exists(test_path):
        logging.debug('worker: Nothing todo for {0} in {1}\n'.format(offset,
                                                                     time.time() - start_worker))
        return

    # mito segmentation
    mi = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': {
            'driver': 'file',
            'path': '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/j0251_72_mito.20220915.var0.full.17006474',
        }
    }, read=True).result()

    # vc, sj segmentation
    vcsj = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': {
            'driver': 'file',
            'path': '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/j0251_72_mito_sj_vc.20220811.var0.full.76152679/'
        }
    }, read=True).result()

    # remove channel dimension
    mi_3d = mi[ts.d['channel'][0]]
    vcsj_3d = vcsj[ts.d['channel'][0]]

    logging.debug('worker: Initialized tensorstores in {0}\n'.format(
        time.time() - start_worker))

    start_reading = time.time()
    try:
        mi_cube = np.array(
            mi_3d[offset[0]:offset[0]+256, offset[1]:offset[1]+256, offset[2]:offset[2]+256])
        vcsj_cube = np.array(
            vcsj_3d[offset[0]:offset[0]+256, offset[1]:offset[1]+256, offset[2]:offset[2]+256])
        logging.debug('worker: Read {0} from tensorstore in {1}\n'.format(
            offset, time.time() - start_reading))

    except Exception as e:
        logging.error(
            f'{e} worker: IndexError for {offset} in {time.time() - start_worker}. Skipping...\n')
        return

    seg_cube = np.zeros_like(mi_cube)
    seg_cube[mi_cube == 1] = 3  # mi
    seg_cube[vcsj_cube == 2] = 1  # sj
    seg_cube[vcsj_cube == 3] = 2  # vc

    seg_cube = np.transpose(seg_cube, (2, 1, 0))

    if config['kd_output_as_raw']:
        data_type = np.uint8
        fast_downsampling = False
    else:
        data_type = np.uint64
        fast_downsampling = True

    assert (stack.cube_shape == config['cubesize']).all(
    ), 'Stack cube shape does not match config cubesize'

    try:
        stack.save_seg(data=seg_cube,
                       data_mag=1,
                       offset=offset,
                       mags=config['mags'],
                       upsample=False,
                       downsample=False,
                       fast_resampling=False
                       )
    except Exception as e:
        logging.error(
            'worker: Error writing in first attempt {e}. Trying again...')

        time.sleep(1)
        stack.save_seg(data=seg_cube,
                       data_mag=1,
                       offset=offset,
                       mags=config['mags'],
                       upsample=False,
                       downsample=False,
                       fast_resampling=False
                       )

    logging.info('worker: Done downloading and writing seg cube {0} in {1}\n'.format(offset,
                                                                                     time.time()-start_worker))
    return


def ts_to_knossos():

    global config
    config = {
        # general
        # TODO: read from the tensorstore
        'subvol_offset': [0, 0, 0],  # 7300, 9000, 7700
        'subvol_size': [27119, 27350, 15494],  # [2600, 2600, 1100],
        'source': 'tensorstore',
        'kd_path': '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/knossosdatasets/sjvcmi_seg',  # target path
        'mags': [1, 2, 4, 8, 16],
        'cubesize': np.array([256, 256, 256]),
        'n_output_threads': 1,
        'n_download_threads': 10,
        'kd_output_as_raw': False,
    }

    if config['n_download_threads'] == 1:
        write_in_kd(config)
    else:
        global bm_download_queue, bm_download_done, stack
        stack = ku.knossosdataset.KnossosDataset()
        # stack.initialize_from_knossos_path(config['kd_path'])
        stack.initialize_from_conf(config['kd_path'] + '/knossos.conf')
        stack._cube_shape = config['cubesize'].astype(np.int32)

        # clean all overlay cubes first
        # if config['kd_output_as_raw'] == False:
        #    print('Starting to delete all existing overlay cubes')
        #    stack.delete_all_overlaycubes(nb_processes=10)
        #   print('Done deleting all existing overlay cubes.')

        write_in_kd_parallel()
    return


if __name__ == '__main__':
    ts_to_knossos()

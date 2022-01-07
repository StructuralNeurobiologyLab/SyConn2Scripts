import time
import pickle as pkl
import os
import torch
import math
import re
from typing import List, Tuple
import glob
from collections import defaultdict

import numpy as np
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import objects, clouds
from sklearn.preprocessing import label_binarize
from torch import nn
from syconn.handler.prediction_pts import pts_feat_dict, pts_feat_ds_dict
from syconn.reps.rep_helper import colorcode_vertices
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.models.lcp_adapt import ConvAdaptSeg
import open3d as o3d
from syconn.handler.basics import write_obj2pkl
from morphx.classes.pointcloud import PointCloud
from syconn import global_params
from syconn.reps.super_segmentation_dataset import SuperSegmentationObject
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from sklearn.metrics import classification_report


def batch_builder(samples: List[Tuple[PointCloud, np.ndarray]], batch_size: int, input_channels: int):
    point_num = len(samples[0][0].vertices)
    batch_num = math.ceil(len(samples) / batch_size)
    batches = []
    ix = -1
    for batch_ix in range(batch_num):
        pts = torch.zeros((batch_size, point_num, 3))
        features = torch.ones((batch_size, point_num, input_channels))
        mapping_idcs = torch.ones((batch_size, point_num))
        for sample_ix in range(batch_size):
            ix += 1
            if ix == len(samples):
                ix = 0
            pts[sample_ix] = torch.from_numpy(samples[ix][0].vertices).float()
            features[sample_ix] = torch.from_numpy(samples[ix][0].features).float()
            mapping_idcs[sample_ix] = torch.from_numpy(samples[ix][1])
        batches.append((pts, features, mapping_idcs))
    return batches


def evaluate_preds(preds_idcs: np.ndarray, preds: np.ndarray, pred_labels: np.ndarray):
    """ ith entry in ``preds_idcs`` contains vertex index of prediction saved at ith entry of preds.
        Predictions for each vertex index are gathered and then evaluated by a majority vote.
        The result gets saved at the respective index in the pred_labels array. """
    pred_dict = defaultdict(list)
    u_preds_idcs = np.unique(preds_idcs)
    for i in range(len(preds_idcs)):
        pred_dict[preds_idcs[i]].append(preds[i])
    for u_ix in u_preds_idcs:
        counts = np.bincount(pred_dict[u_ix])
        pred_labels[u_ix] = np.argmax(counts)


def preds2mesh(pkl_files, out_dir, model):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """
    node_map_properties = {'k': 20, 'ds_vertices': 1}

    log = initialize_logging('model_dasbt', f'{out_dir}', overwrite=True)
    log.info(f'Predicting N={len(pkl_files)} files: {pkl_files}".\nnode_map_properties: {node_map_properties},'
             f'redundancy: {redundancy}, model path: {mpath}, vertex kNN: {smoothing_k}')
    np.random.seed(0)
    device = torch.device('cuda')
    # Model selection
    lcp_flag = True
    context_size = 15000
    nb_points = 15000
    batch_size = 4
    valid_transform = clouds.Compose([clouds.Center(), ])
    total_inference_time = 0
    total_postproc_time = 0
    total_proc_time = 0
    total_path_length = 0
    total_vx_count = 0


    voxel_dc = dict(pts_feat_ds_dict['compartment'])
    feats = dict(pts_feat_dict)
    del feats['syn_ssv_asym']
    del feats['syn_ssv_sym']
    del feats['sv_myelin']
    inp_channels = len(feats)
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])
    # final results
    verts_gt, verts_pred, nodes_gt, nodes_pred = [], [], [], []
    for pkl_file in pkl_files:
        sso_id = int(re.findall(r'sso_(\d+).pkl', pkl_file)[0])
        sso = SuperSegmentationObject(sso_id)
        assert sso.attr_dict_exists
        # get ground truth labels
        # vertex and node labels contain nodes/vertices labels that indicate if not close to a GT node.
        hc = load_pkl2obj(pkl_file)
        hc_unlabeled_flag = load_pkl2obj(pkl_file.replace('.pkl', '_eval.pkl'))
        assert np.all(hc['nodes'] == hc_unlabeled_flag['nodes'])
        node_labels_gt = hc_unlabeled_flag['node_labels'].squeeze()
        # ultrastructure mesh vertices are stored with label != -1, vertices that are close to unlabeled nodes have
        # label -2 in hc_correct_vertex_labels
        vertex_labels_gt = hc['labels'].squeeze()
        vertices_coord_gt = hc['vertices']
        assert np.all(vertices_coord_gt == hc_unlabeled_flag['vertices'])
        vertex_labels_gt_unlabeled = hc_unlabeled_flag['labels'].squeeze()
        # filter ultrastructure
        vertex_labels_gt_unlabeled = vertex_labels_gt_unlabeled[vertex_labels_gt != -1]
        vertices_coord_gt = vertices_coord_gt[vertex_labels_gt != -1]
        assert np.all(vertices_coord_gt == sso.mesh[1].reshape((-1, 3)))
        vertex_labels_gt = vertex_labels_gt[vertex_labels_gt != -1]
        vertex_labels_gt_unlabeled = vertex_labels_gt_unlabeled[vertex_labels_gt != -1]
        # flag vertices that were not labeled by a skeleton node close to a GT node
        # they will be removed before final performance eval
        vertex_labels_gt[vertex_labels_gt_unlabeled == -2] = -1

        # prepare input data
        vert_dc = {}
        voxel_idcs = {}
        offset = 0
        obj_bounds = {}
        start_proc = time.time()
        for ix, k in enumerate(parts):
            # build cell representation by adding cell surface and possible organelles
            pcd = o3d.geometry.PointCloud()
            verts = sso.load_mesh(k)[1].reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd, idcs = pcd.voxel_down_sample_and_trace(parts[k][0], pcd.get_min_bound(), pcd.get_max_bound())
            voxel_idcs[k] = np.max(idcs, axis=1)
            vert_dc[k] = np.asarray(pcd.points)
            obj_bounds[k] = [offset, offset + len(pcd.points)]
            offset += len(pcd.points)

        if type(parts['sv'][1]) == int:
            sample_feats = np.concatenate([[parts[k][1]] * len(vert_dc[k]) for k in parts]).reshape(-1, 1)
            sample_feats = label_binarize(sample_feats, classes=np.arange(len(parts)))
        else:
            feats_dc = {}
            for key in parts:
                sample_feats = np.ones((len(vert_dc[key]), len(parts[key][1])))
                sample_feats[:] = parts[key][1]
                feats_dc[key] = sample_feats
            sample_feats = np.concatenate([feats_dc[k] for k in parts])
        sample_pts = np.concatenate([vert_dc[k] for k in parts])
        if len(feats) == 1:
            sample_feats = np.ones((len(sample_pts), 1), dtype=np.float32)
        start = time.time()
        hc_sample = HybridCloud(hc_unlabeled_flag['nodes'], hc_unlabeled_flag['edges'], vertices=sample_pts,
                                labels=np.ones((len(sample_pts), 1)), features=sample_feats, obj_bounds=obj_bounds)
        node_arrs, source_nodes = splitting.split_single(hc_sample, context_size, context_size / redundancy)
        samples = []
        for ix, node_arr in enumerate(node_arrs):
            # vertices which correspond to nodes in node_arr
            sample, idcs_sub = objects.extract_cloud_subset(hc_sample, node_arr)
            # random subsampling of the corresponding vertices
            sample, idcs_sample = clouds.sample_cloud(sample, nb_points, padding=None)
            # indices with respect to the total HybridCloud
            idcs_global = idcs_sub[idcs_sample.astype(int)]
            bounds = hc_sample.obj_bounds['sv']
            sv_mask = np.logical_and(idcs_global < bounds[1], idcs_global >= bounds[0])
            idcs_global[np.logical_not(sv_mask)] = -1
            if len(sample.vertices) == 0:
                raise ValueError
            valid_transform(sample)
            samples.append((sample, idcs_global))
        total_proc_time += time.time() - start_proc

        model.to(device)
        model.eval()
        preds = []
        idcs_preds = []
        start = time.time()
        with torch.no_grad():
            batches = batch_builder(samples, batch_size, inp_channels)
            for batch in batches:
                pts = batch[0].to(device, non_blocking=True)
                features = batch[1].to(device, non_blocking=True)
                # lcp and convpoint use different axis order
                if lcp_flag:
                    pts = pts.transpose(1, 2)
                    features = features.transpose(1, 2)
                outputs = model(features, pts)
                if lcp_flag:
                    outputs = outputs.transpose(1, 2)
                outputs = outputs.cpu().detach().numpy()
                for ix in range(batch_size):
                    preds.append(np.argmax(outputs[ix], axis=1))
                    idcs_preds.append(batch[2][ix])
        total_inference_time += time.time() - start
        start_proc = time.time()
        preds = np.concatenate(preds)
        idcs_preds = np.concatenate(idcs_preds)
        # filter possible organelles and borders
        preds = preds[idcs_preds != -1]
        idcs_preds = idcs_preds[idcs_preds != -1].astype(int) - hc_sample.obj_bounds['sv'][0]
        # get length of the original cell mesh vertices
        pred_labels = np.ones(len(voxel_idcs['sv'])) * -1
        evaluate_preds(idcs_preds, preds, pred_labels)
        sso_preds = np.ones(np.sum(hc['features'] == 0)) * -1
        sso_preds[voxel_idcs['sv']] = pred_labels
        total_proc_time += time.time() - start_proc

        # map predictions to unpredicted vertices
        if not np.all(sso_preds != -1):
            # this is the method used during prediction pipeline in syconn
            sso_preds = colorcode_vertices(
                vertices_coord_gt, vertices_coord_gt[sso_preds != -1],
                sso_preds[sso_preds != -1],
                k=smoothing_k, return_color=False, nb_cpus=10)
        print(f"Finished predictions after {(time.time() - start):.2f} seconds.")

        start_proc = time.time()
        # propagate vertex labels to node labels
        ld = sso.label_dict('vertex')
        ld['dasbt'] = sso_preds
        node_preds = sso.semseg_for_coords(hc_unlabeled_flag['nodes'] / sso.scaling, 'dasbt', **node_map_properties)
        assert len(vertices_coord_gt) == len(sso_preds)
        total_postproc_time += time.time() - start_proc
        assert len(sso_preds) == len(vertex_labels_gt)

        log.info(f'Evaluation of the following file: {pkl_file}')
        vert_rep = classification_report(vertex_labels_gt, sso_preds, labels=np.arange(5),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4,
                                         zero_division=0)
        log.info(f'----------------------------------------\n'
                 f'Vertex performance:\n{vert_rep}')
        node_rep = classification_report(node_labels_gt, node_preds, labels=np.arange(5),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4,
                                         zero_division=0)
        log.info(f'----------------------------------------\n'
                 f'Node performance:\n{node_rep}')

        # collect results
        verts_gt.append(vertex_labels_gt)
        verts_pred.append(sso_preds)
        nodes_gt.append(node_labels_gt)
        nodes_pred.append(node_preds)

        # neuron GT properties
        total_path_length += sso.total_edge_length() / 1e3  # nm to um
        total_vx_count += sso.size

        # debug output for prediction
        assert len(sso_preds) == len(sso.mesh[1].reshape((-1, 3)))
        # requires that sso.mesh[1] is aligned with hc.vertices! this is tested at the beginning
        sso.semseg2mesh(semseg_key='dasbt', dest_path=f'{out_dir}/{sso_id}_dasbt.k.zip')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(f'{out_dir}/{sso_id}_predictions.pkl', 'wb') as f:
            pkl.dump(dict(vert_pred=sso_preds, vert_gt=vertex_labels_gt,
                          node_pred=node_preds, node_gt=node_labels_gt, vertices=vertices_coord_gt), f)
    verts_gt = np.concatenate(verts_gt)
    verts_pred = np.concatenate(verts_pred)
    nodes_gt = np.concatenate(nodes_gt)
    nodes_pred = np.concatenate(nodes_pred)

    # remove predictions that were not close to an annotated node
    verts_pred = verts_pred[verts_gt != -1]
    verts_gt = verts_gt[verts_gt != -1]
    nodes_pred = nodes_pred[nodes_gt != -1]
    nodes_gt = nodes_gt[nodes_gt != -1]

    log.info(f'Evaluation of all {len(pkl_files)} following files: {pkl_files}')
    vert_rep = classification_report(verts_gt, verts_pred, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4,
                                     zero_division=0)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance:\n{vert_rep}')

    node_rep = classification_report(nodes_gt, nodes_pred, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4,
                                     zero_division=0)
    log.info(f'----------------------------------------\n'
             f'Total node performance:\n{node_rep}')

    # only one bouton label - map terminal to bouton
    verts_gt = np.array(verts_gt)
    verts_gt[verts_gt == 4] = 3
    verts_pred = np.array(verts_pred)
    verts_pred[verts_pred == 4] = 3
    log.info(f'Evaluation of all {len(pkl_files)} following files: {pkl_files}')
    vert_rep = classification_report(verts_gt, verts_pred, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4,
                                     zero_division=0)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance (one bouton label):\n{vert_rep}')
    nodes_gt = np.array(nodes_gt)
    nodes_gt[nodes_gt == 4] = 3
    nodes_pred = np.array(nodes_pred)
    nodes_pred[nodes_pred == 4] = 3
    node_rep = classification_report(nodes_gt, nodes_pred, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4,
                                     zero_division=0)
    log.info(f'----------------------------------------\n'
             f'Total node performance (one bouton label):\n{node_rep}')

    log.info(f'Total processing time in s: {total_proc_time}')
    log.info(f'Total inference time in s: {total_inference_time}')
    log.info(f'Total post-processing time in s: {total_postproc_time}')

    log.info(f'Total path length [um]: {total_path_length}')
    log.info(f'Total voxel count [voxel size: {global_params.config["scaling"]}]: {total_vx_count}')


def _load_model(mpath):
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=True)
    act = nn.ReLU
    m = ConvAdaptSeg(4, 5, get_conv(conv), get_search(search), kernel_num=64,
                     architecture=None, activation=act, norm='gn').to('cuda')
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    # m = torch.load(mpath)
    return m


if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    global_params.wd = wd
    data_path = "/wholebrain/songbird/j0126/GT/axgt_semseg/testdata/hc_out_2021_12_axgtsemseg_SUPPORT/"
    out_dir = f'/wholebrain/scratch/pschuber/experiments/axgtsemseg_testj0126/lcp/k20_nodemapk20/'
    os.makedirs(out_dir, exist_ok=True)
    file_paths = [fname for fname in glob.glob(data_path + '/*.pkl') if 'eval' not in os.path.split(fname)[1]]
    redundancy = 2
    smoothing_k = 20
    mpath = '/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_dasbt_j0126/semseg_pts_nb15000_ctx15000_dasbt_nclass5_lcp_GN_noKernelSep_AdamW_dice_eval0/state_dict.pth'
    model = _load_model(mpath)
    preds2mesh(file_paths, out_dir, model)

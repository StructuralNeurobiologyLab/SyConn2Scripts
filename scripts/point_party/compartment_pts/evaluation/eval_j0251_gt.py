import os
import numpy as np
import open3d as o3d
import torch
import math
import time
import re
import pickle as pkl
from typing import List, Tuple
import glob

from collections import defaultdict
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import objects, clouds
from morphx.classes.pointcloud import PointCloud
from sklearn.preprocessing import label_binarize
from elektronn3.models.lcp_adapt import ConvAdaptSeg

from torch import nn
from syconn import global_params
from syconn.handler.prediction_pts import pts_feat_dict, pts_feat_ds_dict
from syconn.reps.rep_helper import colorcode_vertices
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset, SuperSegmentationObject
from syconn.handler.prediction_pts import load_hc_pkl
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from sklearn.metrics import classification_report

from lightconvpoint.utils.network import get_search, get_conv


def convert_cmpt_preds_(ld):
    ads = ld['ads']
    abt = ld['abt']
    dnh = ld['dnh']
    a_mask = (ads == 1).reshape(-1)
    d_mask = (ads == 0).reshape(-1)
    abt[abt == 1] = 3
    abt[abt == 2] = 4
    abt[abt == 0] = 1
    dnh[dnh == 1] = 5
    dnh[dnh == 2] = 6
    ads[a_mask] = abt[a_mask]
    ads[d_mask] = dnh[d_mask]
    return ads


def convert_7class_to_das(arr):
    """
    Not in-place!

    Final labels: (0, dendrite), (1, axon), (2, soma)
    """
    out = np.array(arr)
    out[out == 3] = 1
    out[out == 4] = 1
    out[out == 5] = 0
    out[out == 6] = 0
    return out


def convert_7class_to_dasbh(arr):
    """
    Not in-place! Merge terminal and en-passant boutons, merge neck to dendrite.

    Final labels: (0, dendrite), (1, axon), (2, soma), (3, bouton), (4, head).
    """
    out = np.array(arr)
    out[out == 4] = 3  # terminal to bouton
    out[out == 5] = 0  # neck to dendrite
    out[out == 6] = 4
    return out


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


def predict_sso_thread_3models_hierarchy(pkl_files: List[str], models: list,
                                         pred_keys: list, redundancy: int, out_p: str, use_subcell, log):
    np.random.seed(0)
    device = torch.device('cuda')
    # Model selection
    lcp_flag = True
    context_size = 15000
    nb_points = 15000
    batch_size = 4
    valid_transform = clouds.Compose([clouds.Center(),])
    total_inference_time = 0
    total_postproc_time = 0
    total_proc_time = 0
    total_path_length = 0
    total_vx_count = 0

    node_map_properties = {'k': 50, 'ds_vertices': 1}

    voxel_dc = dict(pts_feat_ds_dict['compartment'])
    feats = dict(pts_feat_dict)
    del feats['syn_ssv_asym']
    del feats['syn_ssv_sym']
    del feats['sv_myelin']
    if not use_subcell:
        del feats['mi']
        del feats['vc']
        del feats['syn_ssv']
    inp_channels = len(feats)
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])
    # final results
    verts_gt, verts_pred, nodes_gt, nodes_pred = [], [], [], []
    for pkl_file in pkl_files:
        vertex_res_dc = {}

        sso_id = int(re.findall(r'sso_(\d+).pkl', pkl_file)[0])
        sso = SuperSegmentationObject(sso_id)
        if not os.path.isfile(f'{out_p}/{sso_id}_predictions.pkl'):
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

            for model, pred_key in zip(models, pred_keys):
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
                # vertex predictions
                vertex_res_dc[pred_key] = sso_preds
            print(f"Finished predictions after {(time.time() - start):.2f} seconds.")

            start_proc = time.time()
            vertices_pred_converted = convert_cmpt_preds_(vertex_res_dc)
            assert -1 not in vertices_pred_converted
            # propagate vertex labels to node labels
            ld = sso.label_dict('vertex')
            ld['final_3models'] = vertices_pred_converted
            node_preds_converted = sso.semseg_for_coords(hc_unlabeled_flag['nodes'] / sso.scaling, 'final_3models',
                                                         **node_map_properties)
            assert -1 not in node_preds_converted
            assert len(vertices_coord_gt) == len(vertices_pred_converted)
            total_postproc_time += time.time() - start_proc
            assert len(vertices_pred_converted) == len(vertex_labels_gt)

            # debug output for prediction
            assert len(vertices_pred_converted) == len(sso.mesh[1].reshape((-1, 3)))
            # requires that sso.mesh[1] is aligned with hc.vertices! this is tested at the beginning
            sso.semseg2mesh(semseg_key='final_3models', dest_path=f'{out_p}/{sso_id}_final_3models.k.zip')
            if not os.path.exists(out_p):
                os.makedirs(out_p)

            with open(f'{out_p}/{sso_id}_predictions.pkl', 'wb') as f:
                pkl.dump(dict(vert_pred=vertices_pred_converted, vert_gt=vertex_labels_gt,
                              node_pred=node_preds_converted, node_gt=node_labels_gt, vertices=vertices_coord_gt), f)
        else:
            res_dc = load_pkl2obj(f'{out_p}/{sso_id}_predictions.pkl')
            vertices_pred_converted = res_dc['vert_pred']
            vertex_labels_gt = res_dc['vert_gt']
            node_preds_converted = res_dc['node_pred']
            node_labels_gt = res_dc['node_gt']
            ld = sso.label_dict('vertex')
            ld['final_3models_gt'] = vertex_labels_gt
            sso.semseg2mesh(semseg_key='final_3models_gt', dest_path=f'{out_p}/{sso_id}_final_3models_gt.k.zip')

        sso_report = ''
        mask = (vertices_pred_converted != -1) & (vertex_labels_gt != -1)
        vertex_labels_gt_tmp = vertex_labels_gt[mask]
        vertices_pred_converted_tmp = vertices_pred_converted[mask]
        vert_rep = classification_report(vertex_labels_gt_tmp, vertices_pred_converted_tmp, labels=np.arange(7),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} vertex performance:\n{vert_rep}'
        mask = (node_preds_converted != -1) & (node_labels_gt != -1)
        node_labels_gt_tmp = node_labels_gt[mask]
        node_preds_converted_tmp = node_preds_converted[mask]
        node_rep = classification_report(node_labels_gt_tmp, node_preds_converted_tmp, labels=np.arange(7),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} node performance:\n{node_rep}'
        # dasbh report
        vert_rep = classification_report(convert_7class_to_dasbh(vertex_labels_gt_tmp),
                                         convert_7class_to_dasbh(vertices_pred_converted_tmp), labels=np.arange(5),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'head'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} vertex performance:\n{vert_rep}'
        node_rep = classification_report(convert_7class_to_dasbh(node_labels_gt_tmp),
                                         convert_7class_to_dasbh(node_preds_converted_tmp), labels=np.arange(5),
                                         target_names=['dendrite', 'axon', 'soma', 'bouton', 'head'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} node performance:\n{node_rep}'
        # das report
        vert_rep = classification_report(convert_7class_to_das(vertex_labels_gt_tmp),
                                         convert_7class_to_das(vertices_pred_converted_tmp), labels=np.arange(3),
                                         target_names=['dendrite', 'axon', 'soma'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} vertex performance:\n{vert_rep}'
        node_rep = classification_report(convert_7class_to_das(node_labels_gt_tmp),
                                         convert_7class_to_das(node_preds_converted_tmp), labels=np.arange(3),
                                         target_names=['dendrite', 'axon', 'soma'],
                                         zero_division=0, digits=4)
        sso_report += f'----------------------------------------\n{sso_id} node performance:\n{node_rep}'
        with open(f'{out_p}/{sso_id}_report.txt', 'w') as f:
            f.write(sso_report)
        # collect results
        verts_gt.append(vertex_labels_gt)
        verts_pred.append(vertices_pred_converted)
        nodes_gt.append(node_labels_gt)
        nodes_pred.append(node_preds_converted)

        # neuron GT properties
        total_path_length += sso.total_edge_length() / 1e3  # nm to um
        total_vx_count += sso.size

    verts_gt = np.concatenate(verts_gt)
    verts_pred = np.concatenate(verts_pred)
    nodes_gt = np.concatenate(nodes_gt)
    nodes_pred = np.concatenate(nodes_pred)

    # remove predictions that were not close to an annotated node
    verts_pred = verts_pred[verts_gt != -1]
    verts_gt = verts_gt[verts_gt != -1]
    nodes_pred = nodes_pred[nodes_gt != -1]
    nodes_gt = nodes_gt[nodes_gt != -1]

    vert_rep = classification_report(verts_gt, verts_pred, labels=np.arange(7),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                                     zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance:\n{vert_rep}')

    node_rep = classification_report(nodes_gt, nodes_pred, labels=np.arange(7),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                                     zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total node performance:\n{node_rep}')
    # dasbh performance
    vert_rep = classification_report(convert_7class_to_dasbh(verts_gt),
                                     convert_7class_to_dasbh(verts_pred), labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'head'], zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance (dendrite axon soma bouton head):\n{vert_rep}')

    node_rep = classification_report(convert_7class_to_dasbh(nodes_gt),
                                     convert_7class_to_dasbh(nodes_pred), labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'head'], zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total node performance (dendrite axon soma bouton head):\n{node_rep}')
    # das performance
    vert_rep = classification_report(convert_7class_to_das(verts_gt),
                                     convert_7class_to_das(verts_pred), labels=np.arange(3),
                                     target_names=['dendrite', 'axon', 'soma'], zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance (dendrite axon soma):\n{vert_rep}')

    node_rep = classification_report(convert_7class_to_das(nodes_gt),
                                     convert_7class_to_das(nodes_pred), labels=np.arange(3),
                                     target_names=['dendrite', 'axon', 'soma'], zero_division=0, digits=4)
    log.info(f'----------------------------------------\n'
             f'Total node performance (dendrite axon soma):\n{node_rep}')

    log.info(f'Total processing time in s: {total_proc_time}')
    log.info(f'Total inference time in s: {total_inference_time}')
    log.info(f'Total post-processing time in s: {total_postproc_time}')

    log.info(f'Total path length [um]: {total_path_length}')
    log.info(f'Total voxel count [voxel size: {global_params.config["scaling"]}]: {total_vx_count}')


def _load_model(mpath, ch_in):
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=True)
    act = nn.ReLU
    m = ConvAdaptSeg(ch_in, 3, get_conv(conv), get_search(search), kernel_num=64,
                     architecture=None, activation=act, norm='gn').to('cuda')
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    # m = torch.load(mpath)
    return m


if __name__ == '__main__':
    global_params.wd = f'/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3/'
    data_dir = '/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_12_final/test/hc_out_2021_12_fine_SUPPORT_extended_soma/'
    out_dir = f'/wholebrain/scratch/pschuber/experiments' \
              f'/compartment_pts_evalscompartment_3models_j0251_cmpt_j0251_eval/'

    appendix = '_cellshapeOnly'  # '_cellshapeOnly'
    ch_in = 1 if appendix == '_cellshapeOnly' else 4
    mdir_base_ = f'/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_3models{appendix}/'
    mdirs = [mdir_base_ + f'semseg_pts_nb15000_ctx15000_ads_nclass3_lcp_GN_noKernelSep_AdamW_dice{appendix}_eval0/state_dict.pth',
             mdir_base_ + f'semseg_pts_nb15000_ctx15000_dnh_nclass3_lcp_GN_noKernelSep_AdamW_dice{appendix}_eval0/state_dict.pth',
             mdir_base_ + f'semseg_pts_nb15000_ctx15000_abt_nclass3_lcp_GN_noKernelSep_AdamW_dice{appendix}_eval0/state_dict.pth']

    # mdir_base_ = f'/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/models/compartment_pts/'
    # mdirs = [mdir_base_ + 'semseg_pts_nb15000_ctx15000_typesads_nclass3_ptconv_GN_strongerWeighted_noKernelSep_eval0_scale5000_bs4_cnum3_fdim4/state_dict.pth',
    #          mdir_base_ + 'semseg_pts_nb15000_ctx15000_typesdnh_nclass3_ptconv_GN_strongerWeighted_noKernelSep_eval0_scale5000_bs4_cnum3_fdim4/state_dict.pth',
    #          mdir_base_ + 'semseg_pts_nb15000_ctx15000_typesabt_nclass3_ptconv_GN_strongerWeighted_noKernelSep_eval0_scale5000_bs4_cnum3_fdim4/state_dict.pth']

    if appendix is '':
        out_dir += 'baseline/'
    else:
        out_dir += appendix.replace('_', '')

    pred_types = ['ads', 'dnh', 'abt']
    models = [_load_model(mpath, ch_in) for mpath in mdirs]
    red = 2
    smoothing_k = 20
    log = initialize_logging('model_hierarchy_j0251_eval', f'{out_dir}', overwrite=True)

    pkl_files = [fname for fname in glob.glob(data_dir + '/*.pkl') if 'eval' not in os.path.split(fname)[1]]
    log.info(f'Predicting N={len(pkl_files)} files: {pkl_files}".\n'
             f'prediction key: {pred_types}, redundancy: {red}, model paths: {mdirs}, vertex kNN: {smoothing_k}')
    predict_sso_thread_3models_hierarchy(pkl_files, models, pred_types, red, out_dir, use_subcell=appendix != '_cellshapeOnly',
                                         log=log)

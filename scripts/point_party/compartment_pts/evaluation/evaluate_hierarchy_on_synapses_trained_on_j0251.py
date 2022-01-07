import os
import numpy as np
import open3d as o3d
import torch
import math
import time
import pickle as pkl
from typing import List, Tuple

import sklearn.metrics as sm
from scipy.spatial import cKDTree
from utils import merge
from syconn.reps.rep_helper import colorcode_vertices
from collections import defaultdict
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import objects, clouds
from morphx.classes.pointcloud import PointCloud
from sklearn.preprocessing import label_binarize
from elektronn3.models.lcp_adapt import ConvAdaptSeg

from torch import nn
from syconn import global_params
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.config import initialize_logging

from lightconvpoint.utils.network import get_search, get_conv


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


def predict_sso_thread_3models_hierarchy(sso_ids: List[int], wd: str, models: list,
                                         pred_keys: list, redundancy: int, v3: bool = True, out_p: str = None):
    from syconn.handler.prediction_pts import pts_feat_dict, pts_feat_ds_dict
    ssd = SuperSegmentationDataset(working_dir=wd)
    device = torch.device('cuda')
    # Model selection
    lcp_flag = True
    inp_channels = 4
    context_size = 15000
    nb_points = 15000
    batch_size = 4
    scale_norm = 5000
    valid_transform = clouds.Compose([clouds.Center(),
                                      # clouds.Normalization(scale_norm)
                                      ])
    voxel_dc = dict(pts_feat_ds_dict['compartment'])
    feats = dict(pts_feat_dict)
    if 'hc' in feats:
        feats['sv'] = feats['hc']
        feats.pop('hc')
    if v3 and 'syn_ssv' in feats:
        feats['sj'] = feats['syn_ssv']
        feats.pop('syn_ssv')
    if v3 and 'syn_ssv' in voxel_dc:
        voxel_dc['sj'] = voxel_dc['syn_ssv']
        voxel_dc.pop('syn_ssv')
    del feats['syn_ssv_asym']
    del feats['syn_ssv_sym']
    del feats['sv_myelin']
    total_time = 0
    for model, pred_key in zip(models, pred_keys):
        model.to(device)
        model.eval()

        parts = {}
        for key in feats:
            parts[key] = (voxel_dc[key], feats[key])
        for sso_id in sso_ids:
            start = time.time()
            sso = ssd.get_super_segmentation_object(sso_id)
            vert_dc = {}
            voxel_idcs = {}
            offset = 0
            obj_bounds = {}
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
            if not sso.load_skeleton():
                raise ValueError(f"Couldn't find skeleton of {sso}")
            nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
            hc = HybridCloud(nodes, edges, vertices=sample_pts, labels=np.ones((len(sample_pts), 1)),
                             features=sample_feats, obj_bounds=obj_bounds)
            node_arrs, source_nodes = splitting.split_single(hc, context_size, context_size / redundancy)

            samples = []
            for ix, node_arr in enumerate(node_arrs):
                # vertices which correspond to nodes in node_arr
                sample, idcs_sub = objects.extract_cloud_subset(hc, node_arr)
                # random subsampling of the corresponding vertices
                sample, idcs_sample = clouds.sample_cloud(sample, nb_points, padding=None)
                # indices with respect to the total HybridCloud
                idcs_global = idcs_sub[idcs_sample.astype(int)]
                bounds = hc.obj_bounds['sv']
                sv_mask = np.logical_and(idcs_global < bounds[1], idcs_global >= bounds[0])
                idcs_global[np.logical_not(sv_mask)] = -1
                if len(sample.vertices) == 0:
                    continue
                valid_transform(sample)
                samples.append((sample, idcs_global))

            preds = []
            idcs_preds = []
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

            preds = np.concatenate(preds)
            idcs_preds = np.concatenate(idcs_preds)
            # filter possible organelles and borders
            preds = preds[idcs_preds != -1]
            idcs_preds = idcs_preds[idcs_preds != -1].astype(int) - hc.obj_bounds['sv'][0]
            pred_labels = np.ones(len(voxel_idcs['sv'])) * -1
            print(f"Finished predictions after {(time.time() - start):.2f} seconds.")
            total_time += time.time() - start
            print("Evaluating predictions...")
            start = time.time()
            evaluate_preds(idcs_preds, preds, pred_labels)
            print(f"Finished majority vote in {(time.time() - start):.2f} seconds.")
            sso_vertices = sso.mesh[1].reshape((-1, 3))
            sso_preds = np.ones(len(sso_vertices)) * -1
            sso_preds[voxel_idcs['sv']] = pred_labels
            # map predictions to unpredicted vertices
            if not np.all(sso_preds != -1):
                # this is the method used during prediction pipeline in syconn
                sso_preds = colorcode_vertices(
                    sso_vertices, sso_vertices[sso_preds != -1],
                    sso_preds[sso_preds != -1],
                    k=smoothing_k, return_color=False, nb_cpus=10)
            ld = sso.label_dict('vertex')
            ld[pred_key] = sso_preds
            if out_p is None:
                ld.push()
            else:
                sso.semseg2mesh(pred_key, dest_path=f'{out_p}/{sso_id}_{pred_key}.k.zip')
                if not os.path.exists(out_p):
                    os.makedirs(out_p)
                with open(f'{out_p}/{sso_id}_{pred_key}.pkl', 'wb') as f:
                    pkl.dump(sso_preds, f)
    return total_time


def _load_model(mpath, ch_in):
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=True)
    act = nn.ReLU
    m = ConvAdaptSeg(ch_in, 3, get_conv(conv), get_search(search), kernel_num=64,
                     architecture=None, activation=act, norm='gn').to('cuda')
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    # m = torch.load(mpath)
    return m


def predict_3model_hierarchy_j0251():
    pred_types = ['ads', 'dnh', 'abt']

    mdir_base_ = f'/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_3models/'
    mdirs = [mdir_base_ + f'semseg_pts_nb15000_ctx15000_ads_nclass3_lcp_GN_noKernelSep_AdamW_dice_eval0/state_dict.pth',
             mdir_base_ + f'semseg_pts_nb15000_ctx15000_dnh_nclass3_lcp_GN_noKernelSep_AdamW_dice_eval0/state_dict.pth',
             mdir_base_ + f'semseg_pts_nb15000_ctx15000_abt_nclass3_lcp_GN_noKernelSep_AdamW_dice_eval0/state_dict.pth']

    models = {typ: _load_model(mpath, 4) for mpath, typ in zip(mdirs, pred_types)}

    base_dir = f'/wholebrain/scratch/pschuber/experiments/compartment_pts_evals/compartment_3models_j0251_syneval_cmn_paper/'
    red = 2
    pred_keys = [f'{pt}_j0251_corrected' for pt in pred_types]
    log = initialize_logging('model_hierarchy_j0251_eval_corrected', f'{base_dir}/eval_corrected/', overwrite=False)
    log.info(f'Using models: {mdirs}')
    log.info(f'Predicting ssvs {ssv_ids} from working directory "{wd}".\nkNN for synapse label mapping: {nn_syn_mapping}'
             f'prediction key: {pred_keys}, redundancy: {red}, model path: {base_dir}, vertex kNN: {smoothing_k}')

    # set wd according to GT files
    global_params.wd = wd
    duration = predict_sso_thread_3models_hierarchy(
        ssv_ids, wd, models=list(models.values()), pred_keys=pred_keys, redundancy=red, out_p=base_dir)
    ssd = SuperSegmentationDataset(working_dir=wd)
    vx_cnt = np.sum([ssv.size for ssv in ssd.get_super_segmentation_object(ssv_ids)])
    total_inference_speed = vx_cnt / duration * 3600 * np.prod(ssd.scaling) / 1e9  # in um^3 / H
    log.info(f'Processing speed for "{pred_keys}": {total_inference_speed:.2f} µm^3/h')
    log.info(f'Processing speed for "{pred_keys}": {(vx_cnt / 1e9 / duration * 3600):.2f} GVx/h')


def write_confusion_matrix(cm: np.array, names: list) -> str:
    txt = f"{'':<15}"
    for name in names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(names):
        txt += f"{name:<15}"
        for num in cm[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


# synapse GT labels: 0: dendrite, 1: axon, 2: head, 3: soma
mapping_gt = {0: 0, 1: 3, 2: 2, 3: 3}  # dendrite, neck (excluded), head, other
exclude = [3]  # gt labels excluded during eval
# dendrite->dendrite, axon->other, soma->other, bouton->other, terminal->other, neck_>neck, head->head
mapping_preds = {0: 0, 1: 3, 2: 3, 3: 3, 4: 3, 5: 1, 6: 2}


if __name__ == "__main__":
    smoothing_k = 20
    nn_syn_mapping = 20

    # for prediction
    ssv_ids = [141995, 11833344, 28410880, 28479489]
    wd = "/wholebrain/scratch/areaxfs3/"
    predict_3model_hierarchy_j0251()

    # eval
    with open(os.path.expanduser('/wholebrain/scratch/jklimesch/gt/syn_gt/converted_v3.pkl'), 'rb') as f:
        data = pkl.load(f)
    ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
    save_path = os.path.expanduser(f'/wholebrain/scratch/pschuber/experiments/compartment_pts_evals/'
                                   f'compartment_3models_j0251_syneval_cmn_paper/')
    save_path_examples = save_path + '/eval_corrected/examples/'
    if not os.path.exists(save_path_examples):
        os.makedirs(save_path_examples)
    total_gt = np.empty((0, 1))
    total_preds = np.empty((0, 1))
    error_count = 0
    error_coords = []
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = ssd.get_super_segmentation_object(sso_id)

            # 0: axon, 1: bouton, 2: terminal
            with open(f'{save_path}/{sso_id}_abt_j0251_corrected.pkl', 'rb') as f:
                abt = pkl.load(f)
            # 0: dendrite, 1: neck, 2: head
            with open(f'{save_path}/{sso_id}_dnh_j0251_corrected.pkl', 'rb') as f:
                dnh = pkl.load(f)
            # 0: dendrite, 1: axon, 2: soma,
            with open(f'{save_path}/{sso_id}_ads_j0251_corrected.pkl', 'rb') as f:
                ads = pkl.load(f)

            pc = merge(sso, ads, {1: (abt, [(1, 3), (2, 4), (0, 1)]), 0: (dnh, [(1, 5), (2, 6)])})
            pc.save2pkl(save_path + 'eval_corrected/' + str(sso_id) + '_corrected.pkl')
            # query synapse coordinates in KDTree of vertices
            tree = cKDTree(pc.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist, ind = tree.query(coords, k=nn_syn_mapping)
            gt = data[str(sso_id)+'_l']
            mask = np.ones((len(coords), 1), dtype=bool)
            for ix in range(len(gt)):
                gt[ix] = mapping_gt[gt[ix]]
                if gt[ix] in exclude:
                    mask[ix] = False
                preds = pc.labels[ind[ix]].reshape(-1).astype(int)
                mv = np.argmax(np.bincount(preds))
                mv = mapping_preds[mv]
                result[ix] = mv
                if result[ix] != gt[ix] and mask[ix]:
                    error_count += 1
                    error_coords.append(coords[ix])
                    idcs_ball = tree.query_ball_point(coords[ix], 5000)
                    verts = np.concatenate((pc.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl')
            mask = mask.reshape(-1)
            total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))[mask]))
            total_preds = np.concatenate((total_preds, result.reshape((-1, 1))[mask]))

    targets = ['shaft', 'neck', 'head', 'other']
    exclude = [targets[i] for i in exclude]
    report = f'Excluded synapse labels: {exclude}\n\n'
    report += sm.classification_report(total_gt, total_preds, labels=np.arange(len(targets)), target_names=targets,
                                       digits=4, zero_division=0)
    cm = sm.confusion_matrix(total_gt, total_preds, labels=np.arange(len(targets)))
    report += '\n\n'
    report += write_confusion_matrix(cm, targets)
    report += f'\n\nNumber of errors: {error_count}'
    report += f'\n\nError locations: {error_coords / ssd.scaling}'
    with open(save_path + 'eval_corrected/report_corrected.txt', 'w') as f:
        f.write(report)

import os
import glob
import re

import numpy as np
from sklearn.metrics import classification_report
from syconn.handler.prediction import get_semseg_axon_model
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import semseg_of_sso_nocache
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
from syconn.handler.config import initialize_logging


def pred_mp(pkl_paths, out_path, version: str = None, overwrite=True):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    log = initialize_logging('eval', out_path)
    params = [(p, out_path, version, overwrite, log) for p in pkl_paths]

    res = start_multiprocess_imap(preds2mesh, params, nb_cpus=10, debug=True)
    verts_gt, verts_pred, nodes_gt, nodes_pred = [], [], [], []
    for (vert_gt, vert_pred), (node_gt, node_pred) in res:
        verts_gt.append(vert_gt)
        verts_pred.append(vert_pred)
        nodes_gt.append(node_gt)
        nodes_pred.append(node_pred)
    verts_gt = np.concatenate(verts_gt)
    verts_pred = np.concatenate(verts_pred)
    nodes_gt = np.concatenate(nodes_gt)
    nodes_pred = np.concatenate(nodes_pred)
    log.info(f'Evaluation of all {len(pkl_paths)} following files: {pkl_paths}')
    vert_rep = classification_report(verts_gt, verts_pred, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance:\n{vert_rep}')

    node_rep = classification_report(nodes_gt, nodes_pred, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Total node performance:\n{node_rep}')

    # only one bouton label - map terminal to bouton
    verts_gt = np.array(verts_gt)
    verts_gt[verts_gt == 4] = 3
    verts_pred = np.array(verts_pred)
    verts_pred[verts_pred == 4] = 3
    log.info(f'Evaluation of all {len(pkl_paths)} following files: {pkl_paths}')
    vert_rep = classification_report(verts_gt, verts_pred, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Total vertex performance (one bouton label):\n{vert_rep}')
    nodes_gt = np.array(nodes_gt)
    nodes_gt[nodes_gt == 4] = 3
    nodes_pred = np.array(nodes_pred)
    nodes_pred[nodes_pred == 4] = 3
    node_rep = classification_report(nodes_gt, nodes_pred, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Total node performance (one bouton label):\n{node_rep}')


def preds2mesh(args):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """
    fname, out_path, version, overwrite, log = args
    sso_id = int(re.findall(r"sso_(\d+).pkl", os.path.split(fname)[1])[0])
    path2pkl = f'{out_path}/pred_{sso_id}.pkl'
    kzip_out = f'{out_path}/{os.path.split(fname)[1][:-6]}_colored.k.zip'

    if os.path.exists(path2pkl) and overwrite:
        os.remove(path2pkl)
    if os.path.exists(kzip_out) and overwrite:
        os.remove(kzip_out)

    # load hybrid cloud GT file
    hc = load_pkl2obj(fname)
    # vertex and node labels contain nodes/vertices labels that indicate if not close to a GT node.
    hc_correct_vertex_labels = load_pkl2obj(fname.replace('.pkl', '_eval.pkl'))
    nodes_gt = hc['nodes']
    assert np.all(nodes_gt == hc_correct_vertex_labels['nodes'])
    node_labels_gt = hc_correct_vertex_labels['node_labels']

    edges_gt = hc['edges']
    # ultrastructure mesh vertices are stored with label != -1
    vertex_labels_gt = hc['labels'].squeeze()
    vertices_gt = hc['vertices']
    assert np.all(vertices_gt == hc_correct_vertex_labels['vertices'])
    vertex_labels_unlabeled_flag = hc_correct_vertex_labels['labels'].squeeze()
    vertices_gt = vertices_gt[vertex_labels_gt != -1]
    vertex_labels_unlabeled_flag = vertex_labels_unlabeled_flag[vertex_labels_gt != -1]
    vertex_labels_gt = vertex_labels_gt[vertex_labels_gt != -1]
    # flag vertices that were not labeled by a skeleton node close to a GT node
    vertex_labels_gt[vertex_labels_unlabeled_flag == -2] = -1

    # use stored skeleton in the GT pkl file, not the one from the working directory
    sso = SuperSegmentationObject(sso_id, view_caching=True, working_dir=wd)
    sso.skeleton = dict()
    sso.skeleton['nodes'] = nodes_gt / sso.scaling
    sso.skeleton['edges'] = np.array(edges_gt)
    sso.skeleton['diameters'] = np.ones((len(nodes_gt), 1))

    assert np.all(sso.mesh[1].reshape((-1, 3)) == vertices_gt)
    # cache meshes
    _ = sso.mi_mesh
    _ = sso.sj_mesh
    _ = sso.vc_mesh
    # predict
    # get up-to-date multi-view parameters and model from SyConnData
    global_params.wd = "/wholebrain/scratch/pschuber/SyConnData/"
    m = get_semseg_axon_model()
    view_props = dict(global_params.config['compartments']['view_properties_semsegax'])
    view_props["verbose"] = True
    view_props["k"] = 20  # use k=20 nearest neighbor for labeling of unpredicted vertices
    view_props['semseg_key'] = 'cmn_j0126_semsegaxon'
    sso._config = global_params.config
    sso._version = 'tmp'
    sso._working_dir = None
    semseg_of_sso_nocache(sso, dest_path=kzip_out, model=m,
                          add_cellobjects=('mi', 'vc', 'sj'), **view_props)
    # map to skeleton
    node_preds = sso.semseg_for_coords(
        sso.skeleton['nodes'], view_props['semseg_key'],
        **global_params.config['compartments']['map_properties_semsegax'])
    # map gt
    sso.skeleton['node_labels_gt'] = node_labels_gt

    # map pred
    sso.skeleton[view_props['semseg_key']] = node_preds
    sso.save_skeleton_to_kzip(dest_path=kzip_out, additional_keys=[view_props['semseg_key'], 'node_labels_gt'])

    vertices_pred = sso.label_dict('vertex')[view_props['semseg_key']]
    assert np.max(vertices_pred) < 5  # force not background labels in there
    write_obj2pkl(path2pkl, dict(vertices_pred=vertices_pred, vertices_gt=vertex_labels_gt,
                                 nodes_gt=node_labels_gt, nodes_pred=sso.skeleton[view_props['semseg_key']],
                                 nodes=nodes_gt, vertices=vertices_gt))

    vertices_pred = vertices_pred[vertex_labels_gt != -1]
    vertex_labels_gt = vertex_labels_gt[vertex_labels_gt != -1]
    node_preds = node_preds[node_labels_gt != -1]
    node_labels_gt = node_labels_gt[node_labels_gt != -1]
    log.info(f'Evaluation of the following file: {path2pkl}')
    vert_rep = classification_report(vertex_labels_gt, vertices_pred, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Vertex performance:\n{vert_rep}')
    node_rep = classification_report(node_labels_gt, node_preds, labels=np.arange(5),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton', 'terminal'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Node performance:\n{node_rep}')

    # only one bouton label - map terminal to bouton
    vertex_labels_gt_tmp = np.array(vertex_labels_gt)
    vertex_labels_gt_tmp[vertex_labels_gt_tmp == 4] = 3
    vertices_pred_tmp = np.array(vertices_pred)
    vertices_pred_tmp[vertices_pred_tmp == 4] = 3
    log.info(f'Evaluation of the following file: {path2pkl}')
    vert_rep = classification_report(vertex_labels_gt_tmp, vertices_pred_tmp, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Vertex performance (one bouton label):\n{vert_rep}')
    node_labels_gt_tmp = np.array(node_labels_gt)
    node_labels_gt_tmp[node_labels_gt_tmp == 4] = 3
    node_preds_tmp = np.array(node_preds)
    node_preds_tmp[node_preds_tmp == 4] = 3
    node_rep = classification_report(node_labels_gt_tmp, node_preds_tmp, labels=np.arange(4),
                                     target_names=['dendrite', 'axon', 'soma', 'bouton'], digits=4)
    log.info(f'----------------------------------------\n'
             f'Node performance (one bouton label):\n{node_rep}')

    return (vertex_labels_gt, vertices_pred), (node_labels_gt, node_preds)


if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    global_params.wd = wd
    data_path = "/wholebrain/songbird/j0126/GT/axgt_semseg/testdata/hc_out_2021_12_axgtsemseg_SUPPORT/"
    out_dir = f'/wholebrain/scratch/pschuber/experiments/axgtsemseg_testj0126/multiviews/k20/'
    os.makedirs(out_dir, exist_ok=True)
    file_paths = [fname for fname in glob.glob(data_path + '/*.pkl') if 'eval' not in os.path.split(fname)[1]]
    pred_mp(file_paths, out_dir)

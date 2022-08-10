from syconn.backend.storage import MeshStorage, AttributeDict
from syconn.reps.segmentation import SegmentationDataset
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.proc.sd_proc import dataset_analysis
from syconn.proc.meshes import mesh_area_calc
from syconn import global_params
import syconn.reps.segmentation_helper as seghelp

import numpy as np
from open3d import geometry
from open3d import utility


def remove_unreferenced_vertices(mesh):
    ind, vert = mesh[0], mesh[1]
    o3m = geometry.TriangleMesh(utility.Vector3dVector(vert.reshape((-1, 3))),
                                utility.Vector3iVector(ind.reshape((-1, 3))))
    o3m.remove_unreferenced_vertices()
    new_mesh = [np.asarray(o3m.triangles).flatten(), np.asarray(o3m.vertices).flatten(), mesh[2]]
    return new_mesh


def fix_meshes_from_storage(p):
    sd_cs_ssv = SegmentationDataset("cs_ssv")
    scaling = sd_cs_ssv.scaling
    ms = MeshStorage(p + "/mesh.pkl", read_only=False, compress=True)
    ad = AttributeDict(p + "/attr_dict.pkl", read_only=False)
    for k, v in ms.items():
        cs_ssv = sd_cs_ssv.get_segmentation_object(k)
        v_ = remove_unreferenced_vertices(v)
        cs_ssv._mesh = v_
        # update mesh storage
        ms[k] = cs_ssv.mesh
        # update attribute dict
        csssv_attr_dc = ad[k]
        assert 'mesh_bb' not in cs_ssv.attr_dict
        csssv_attr_dc["mesh_bb"] = cs_ssv.mesh_bb
        assert np.all(cs_ssv.mesh_bb <= cs_ssv.mesh[1].reshape((-1, 3)).max(axis=0))
        csssv_attr_dc["mesh_area"] = mesh_area_calc(cs_ssv.mesh)
        csssv_attr_dc["bounding_box"] = (cs_ssv.mesh_bb // scaling).astype(np.int64)
        csssv_attr_dc["rep_coord"] = (seghelp.calc_center_of_mass(cs_ssv.mesh[1].reshape((-1, 3))) // scaling).astype(np.int32)
    ms.push()
    ad.push()


if __name__ == '__main__':
    global_params.wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/'
    sd = SegmentationDataset('cs_ssv')
    paths = list(sd.iter_so_dir_paths())
    start_multiprocess_imap(fix_meshes_from_storage, paths, nb_cpus=20)
    dataset_analysis(sd, recompute=False, compute_meshprops=False)

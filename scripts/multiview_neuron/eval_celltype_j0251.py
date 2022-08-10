# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler.config import initialize_logging
from syconn.handler.prediction import int2str_converter
from syconn.cnn.TrainData import CelltypeViewsJ0251
from syconn import global_params
import numpy as np
import pandas


# validation splits for the CMN cross-validation
valid_splits = {
0: [195496045, 353029255, 128359794, 182568050, 333699391, 216385949,
    65167118,  21201152, 188009630, 200934266,  53808413, 137037221,
    162255421,  89012795,  71703489,  64884243, 244843536, 105969666,
    212262922, 220786848,  89653207, 154238733, 266958913, 304219675,
    78678335,  39430704,   7257604, 266780973, 299575108,    485995,
    101880432],
1: [153211586, 211079237, 151056009, 351770158, 170629911,   4100696,
       223709061, 230315070, 295747979,   5206530, 139806658, 225082243,
        50477870, 209301390,   6356286, 323974411, 249649411,  88521115,
        87950214,  94617439, 124632411, 343195428,  86004395, 115035541,
       310850503, 256856923,  54035479, 166728448,  64395866, 404723878],
2: [211210265, 335314471,   7170983,  44597807, 211464484, 219716873,
       353692047,  48873241, 128844721,   2388278, 102213218, 137598443,
       248594709, 182449429,  10074977, 331439059, 223248782, 358197079,
        53855057, 131072000, 320160438, 360413953,  74820864,   8842899,
       301752359, 296569131, 178438276,  48907087],
3: [257124680, 263317142, 118828862,  53867549, 391828023, 274976822,
       106098747,   5470525, 235027827, 153849003,   1358090, 197790848,
       136882460, 124953466, 248521516, 155980408, 221909311, 349155476,
       151900699, 244669272, 377325329, 187603632,  16924679, 337314871,
       171661324, 262770969],
4: [220659000, 357464326, 271609944,  53951870, 119603503, 381090627,
       394077279,  99592657, 351931151, 186817352, 295847994,  53854647,
       244296480, 103271250, 160309325, 294636810,  69702934,  68833392,
       320819731,   2535430, 323658561,  33807576, 150132244,  27488282,
        19850276],
5: [ 33556608,  51308374, 172566981, 117867794, 368249469, 160189464,
         5247559, 127500596, 183180806, 103209556, 190335973,  33700316,
        11038493, 347356603,  88094108,   4831854, 197741092, 233539078,
       233057815, 344788250,  93525763, 227158348,  92198963,   7544096],
6: [178569847,  37716791,  86513929, 292469190, 233665025,  59676677,
       224920850, 237743106, 258638339, 141460099, 163370866, 164438389,
        39807134, 282401115, 124230071, 346661682, 101681418, 282357255,
        63254546, 190909798,  75275172, 304504333,  25229890, 281778450],
7: [299264637, 162709548, 223958729, 283550113, 105661565, 364576798,
        53707635, 320901095,   2430978, 137495480, 348408912, 160184340,
       110560628, 183735996, 287875428, 178355874,  45793866,  18977964,
       142466714, 125966381,  20045513,  67927538],
8: [ 63748776,  74121558, 178263721, 113784861,   3340838, 107668226,
        15964798, 186273110,  91479436,  53288024, 139531906, 235242244,
       247743765, 264188434, 185371474, 303999828,  76040510, 248680489,
       206235733, 227847277, 103737965, 186506218],
9: [242704599,  59848034, 267674424, 212660043, 177797756, 335950433,
       245428934, 234357273,   6000389, 221031531, 360313118, 101934642,
        43976081,   1970756, 295682906, 311128103, 190854751, 184421996,
         7183873,  14304948, 199281958]}


if __name__ == "__main__":
    WD = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    global_params.wd = WD
    global_params.config['batch_proc_system'] = None
    nclasses = 11
    int2str_label = {ii: int2str_converter(ii, 'ctgt_j0251_v2') for ii in range(nclasses)}
    str2int_label = {int2str_converter(ii, 'ctgt_j0251_v2'): ii for ii in range(nclasses)}
    csv_p = "/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v4.csv"
    base_dir = '/wholebrain/scratch/pschuber/syconn_v2_paper/figures/celltypes/e3_trainings_cmn_celltypes_j0251/'

    # prepare GT
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint64)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
    classes, c_cnts = np.unique(ssv_labels, return_counts=True)
    if np.max(classes) > nclasses:
        raise ValueError('Class mis-match!')
    log_main = initialize_logging('eval_results', base_dir)
    log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
    log_main.info('Successfully parsed "{}" with the following cell type class '
                  'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                                 c_cnts))
    log_main.info('Total #cells: {}'.format(np.sum(c_cnts)))
    ssd_kwargs = dict(working_dir=WD)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ssv_label_dc = {ssvid: str2int_label[el] for ssvid, el in zip(ssv_ids, str_labels)}
    # --------------------------------------------------------------------------
    # TEST PREDICTIONS OF TRAIN AND VALIDATION DATA
    from syconn.handler.prediction import get_celltype_model_e3
    from syconn.proc.stats import cluster_summary, projection_tSNE, model_performance
    from elektronn3.models.base import InferenceModel
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    np.set_printoptions(precision=4)
    # --------------------------------------------------------------------------
    # analysis of VALIDATION set
    for run_ix in range(3):
        # Perform train data set eval as counter check
        gt_l = []
        certainty = []
        pred_l = []
        pred_proba = []
        loaded_ssv_ids = []
        for cv in range(10):
            ccd = CelltypeViewsJ0251(None, None, cv_val=cv)
            split_dc = ccd.splitting_dict
            # ssv_ids = split_dc['valid']
            ssv_ids = valid_splits[cv]

            loaded_ssv_ids.extend(ssv_ids)
            pred_key_appendix = f'celltype_CV' \
                                f'{cv}/celltype_cmn_j0251v2_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold_CV' \
                                f'{cv}_eval{run_ix}'
            print('Loading cv-{}-data of model {}'.format(cv, pred_key_appendix))
            m_path = base_dir + pred_key_appendix
            pred_key_appendix += '_cmn_new'
            m = InferenceModel(m_path, bs=80)
            for ssv_id in ssv_ids:
                ssv = ssd.get_super_segmentation_object(ssv_id)
                ssv.load_attr_dict()
                # predict
                ssv.nb_cpus = 20
                ssv._view_caching = True
                ssv.predict_celltype_multiview(model=m, pred_key_appendix=pred_key_appendix, onthefly_views=True,
                                               view_props={'use_syntype': True, 'nb_views_model': 20, 'nb_views': 4},
                                               overwrite=True,
                                               save_to_attr_dict=False, verbose=True,
                                               model_props={'n_classes': nclasses, 'da_equals_tan': False})
                ssv.save_attr_dict()
                # GT
                curr_l = ssv_label_dc[ssv.id]
                gt_l.append(curr_l)

                pred_l.append(ssv.attr_dict["celltype_cnn_e3" + pred_key_appendix])
                preds_small = ssv.attr_dict["celltype_cnn_e3{}_probas".format(pred_key_appendix)]
                major_dec = np.zeros(preds_small.shape[1])
                preds_small = np.argmax(preds_small, axis=1)
                for ii in range(len(major_dec)):
                    major_dec[ii] = np.sum(preds_small == ii)
                major_dec /= np.sum(major_dec)
                pred_proba.append(major_dec)
                if pred_l[-1] != gt_l[-1]:
                    print(f'{pred_l[-1]}\t{gt_l[-1]}\t{ssv.id}\t{major_dec}')
                certainty.append(ssv.certainty_celltype("celltype_cnn_e3{}".format(pred_key_appendix)))

        assert len(set(loaded_ssv_ids)) == len(ssv_label_dc)
        # # WRITE OUT COMBINED RESULTS
        pred_proba = np.array(pred_proba)
        certainty = np.array(certainty)
        gt_l = np.array(gt_l)

        target_names = [int2str_label[kk] for kk in range(nclasses)]

        # standard
        classes, c_cnts = np.unique(pred_l, return_counts=True)
        log_main.info('Successful prediction [standard] with the following cell type class '
                      'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        perc_50 = np.percentile(certainty, 50)
        model_performance(pred_proba[certainty > perc_50], gt_l[certainty > perc_50],
                          f'{base_dir}/eval{run_ix}_results_rerun/upperhalf/', n_labels=nclasses, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba[certainty <= perc_50], gt_l[certainty <= perc_50],
                          f'{base_dir}/eval{run_ix}_results_rerun/lowerhalf/', n_labels=nclasses, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba, gt_l, f'{base_dir}/eval{run_ix}_results_rerun/', n_labels=nclasses,
                          target_names=target_names)

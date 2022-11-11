import collections
import glob
import pandas
import seaborn as sns
from syconn.handler import basics, config
from syconn.cnn.TrainData import CellCloudDataJ0251
from syconn.handler.prediction import certainty_estimate, str2int_converter, int2str_converter
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset

from syconn.handler.prediction_pts import predict_pts_plain, pts_loader_scalar_infer,\
    pts_loader_scalar, pts_pred_scalar_nopostproc, get_celltype_model_pts, get_pt_kwargs
import os
import pandas as pd

palette_ident = 'colorblind'


# original validation splits, except for the myelin ablation
"""
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
"""

#valid splits with v6 gt:

valid_splits = {0: [298802518, 1084372433,   82967407,  451680496,   58306741,   13733529,
                    20650554,    5803292,   41365295, 1430238776,   78211969, 1858603352,
                    6849290,  992702734, 1879622313,  432988156,  662759023, 1478187822,
                    1306551312,   70246817,  173325651, 1885902289,  342564861, 2186089387,
                    2396455428,  730205247, 1143820677,  61440710,   32494160,  237059539,
                    2281602503,  596830343, 2416261519],
                1: [1179488043,   37215277,  883348491,   52330826,   17415252,  686948959,
                    1469986452,   26958335,  767221938,  134996903,  829534245,   91746880,
                    412743536, 1041487485,   52433603, 1299023959,   77401600, 1539287734,
                    600576536,   50614045,  639982340,  232031927,   51021023, 1287960029,
                    2396455413,  775753802,  471383931,   74329347,   38429927,   54287778,
                    1389349351, 1235460297, 2414844486],
                2: [884326921,   42381287,   11406245,   75546553, 1872809754, 844007128,
                    1260059508,   42280521, 1126849047,   42816115,   43997381, 1506662741,
                    32356701,  707429821, 1111633762,   24194587,  266718802, 2211254176,
                    1647257354,   28209513,   63431281,  481134355,   35531229,  154130123,
                    2392879043,  468147607,  736243930,   48621163, 1110992528,  327784276,
                    657528417, 1891602548, 2017622103],
                3: [21324985,1234246033,642210125,22572226,946881951,580061165,
                    948907048,19366811,639915529,196738187,61710549,82937782,
                    1433039555,19366766,37993733,1554336852,1130665247,37586442,
                    1543539847,2123026143,4218392,239792676,154194912,2393956627,
                    2398277884,2184701599,844683784,61609306,315877001,456403699,
                    1190822162,881965586,1178542776],
                4: [45991236, 1548500233, 1457569565,   73452048,   23552465,  167456363,
                    46932365,  755170042 ,1400211748, 485690128,   10157981,  824236472,
                    91876357,  214386845,  396750419, 1130226186, 2458163814,  615350170,
                    38297192,  327956029,   39847003,   62152255, 2211357026, 2291926857,
                    2390854023,  730271543, 1014766362,  358218760,  441019367, 1739264227,
                    2442271664, 1482336022],
                5: [1548026851,  452894617,  747983832,   79998766,   33643243,   50511113,
                    61912928,    7626258,   18962132,  558972341, 2520142908,   51288310,
                    993477875, 1557170527, 1460436075,  651454283,   37923853,   18222490,
                    74565413,  377048103, 1183399702,  479415063,   20718752,    4457066,
                    2110074037,  159828922,  156018998,   97103815,   72912236,  570478414,
                    673082704, 1613380765],
                6: [25544419,    5668338, 1227495256, 1736632955,  291719074, 1479772383,
                    1095370130, 1288095636,   15388259, 1267478474,    4488613,  390072183,
                    53144390,  260005841, 2023421752, 1181678413,   79021569, 1716356091,
                    242660426, 1482944565,  328866567,  775215267,    4554974,  159831450,
                    2392709168,  616904585, 1635783011, 1407940343,  232943423,  630062872,
                    1075501300, 1706977484],
                7: [1025055349,   62489778,   76693645,   25372570,  247590206,   15419206,
                    950625605,  338244962, 1389045173, 1811539294, 1109272796, 1357834496,
                    303188818,  687119417,   15420661,   49666859, 1298013841, 1459053371,
                    60801698,   44941740,    4555676, 1185827504, 2392711337, 2398614285,
                    466492792,   15622533,  353835929,  828623925,   10965555, 2387815155,
                    784592203, 1277670955],
                8: [436157555,  468515837,    4423089,   67046128,   52229570, 2289634586,
                    59688447,   13394835, 1369949067,  175077676,  971477608,  875118591,
                    2193474680, 2234435544,  247147704,   38297008, 1823786223,  730206659,
                    48654125,  795252861,   39111453,  414633662,   26353030,  354137851,
                    1391910853,  446586134,  736208384,  986559723,  977686149, 1168217589,
                    681180008, 2172455591],
                9: [1613044317,   63532486,    4352945,   54493399, 1786233482,  984230591,
                    26790127, 1274157732,  825519277,  138203264,   19771708,   45452841,
                    968947981,   52131787,   50648088, 1328107641, 1594693963, 2106254401,
                    883552299,   33605129, 1101410378, 1474000401,  236420399, 1391946810,
                    2088339237, 2391097043,   50542644,  627430241, 1943628702, 2459412470,
                    1943931011, 2394704241]}

def predict_celltype_gt(ssd_kwargs, **kwargs):
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.

    Args:
        ssd_kwargs:

    Returns:

    """
    out_dc = predict_pts_plain(ssd_kwargs, get_celltype_model_pts, pts_loader_scalar_infer, pts_pred_scalar_nopostproc,
                               **kwargs)
    for ssv_id in out_dc:
        logit = np.concatenate(out_dc[ssv_id])
        cls = np.argmax(logit, axis=1).squeeze()
        if np.ndim(cls) == 0:
            cls = cls[None]
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        out_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
    return out_dc


def create_catplot(dest_p, qs, ls=6, r=(0, 1.0), add_boxplot=False, legend=False, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    qs :
    r :
    add_boxplot:
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    c = '0.25'
    size = 10
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    if add_boxplot:
        ax = sns.boxplot(data=qs, palette="Greys", showfliers=False, **kwargs)
    ax = sns.swarmplot(data=qs, clip_on=False, color=c, size=size, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    qs.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def create_lineplot(dest_p, df, ls=6, r=(0, 1.0), legend=True, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    r :
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    size = 10
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    ax = sns.lineplot(data=df, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    df.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def create_pointplot(dest_p, df, ls=6, r=(0, 1.0), legend=True, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    r :
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    size = 8
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    ax = sns.pointplot(data=df, size=size, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    df.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def plot_performance_summary(bd, include_special_inputs=False):
    res_dc_pths = glob.glob(bd + '*/redun*_prediction_results.pkl', recursive=True)
    fscores = []
    labels = []
    redundancies = []
    ctx = []
    npts = []
    for fp in res_dc_pths:
        dc = basics.load_pkl2obj(fp)
        if not include_special_inputs and (not dc['use_syntype'] or dc['cellshape_only'] or not dc['use_myelin']):
            continue
        res = list(dc[f'fscore_macro'])
        fscores.extend(res)
        labels.extend([dc['model_tag']] * len(res))
        redundancies.extend([dc['redundancy']] * len(res))
        npts.extend([dc['npts']] * len(res))
        ctx.extend([dc['ctx']] * len(res))
    index = pandas.MultiIndex.from_arrays([labels, redundancies, npts, ctx], names=('labels', 'redundancy', 'npts', 'ctx'))
    df = pandas.DataFrame(fscores, index=index, columns=['fscore'])
    df = df.sort_values(by=['npts', 'ctx', 'redundancy'], ascending=True)
    create_pointplot(f"{bd}/performance_summary_allRedundancies_pointplot{'_special' if include_special_inputs else ''}.png", df.reset_index(), ci='sd',
                     x='labels', y='fscore', hue='redundancy', dodge=True, r=(0 if include_special_inputs else 0.4, 1), palette=palette_ident,
                     capsize=.1, scale=0.75, errwidth=1)
    create_pointplot(f"{bd}/performance_summary_allRedundancies_pointplot_noline{'_special' if include_special_inputs else ''}.png", df.reset_index(), ci='sd',
                     x='labels', y='fscore', hue='redundancy', dodge=True, r=(0 if include_special_inputs else 0.4, 1), palette=palette_ident,
                     capsize=.1, scale=0.75, errwidth=1, linestyles='')
    create_lineplot(f"{bd}/performance_summary_allRedundancies{'_special' if include_special_inputs else ''}.png", df.reset_index(), ci='sd', err_style='band',
                     x='labels', y='fscore', hue='redundancy', r=(0 if include_special_inputs else 0.4, 1), palette=palette_ident)


if __name__ == '__main__':
    ncv_min = 0
    n_cv = 10
    #cval = [3]
    nclasses = 15
    int2str_label = {ii: int2str_converter(ii, 'ctgt_j0251_v3') for ii in range(nclasses)}
    str2int_label = {int2str_converter(ii, 'ctgt_j0251_v3'): ii for ii in range(nclasses)}
    overwrite = True
    n_runs = 1

    state_dict_fname = 'state_dict.pth'
    wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/"
    bbase_dir = 'cajal/nvmescratch/users/arother/cnn_training/221010_cnn_training_dijkstra/'
    all_res_paths = set()
    for ctx, npts, use_syntype, cellshape_only, use_myelin in [
        #(20000, 50000, False, False, True), (20000, 50000, True, False, True),
        #(20000, 50000, True, True, False), (20000, 25000, True, False, True), (20000, 75000, True, False, True),
        #(20000, 5000, True, False, True), (4000, 25000, True, False, True),
        #(20000, 50000, True, False, False), (20000, 25000, True, False, False),
        (20000, 50000, True, False, False)
    ]:
        scale = ctx // 10
        skip_model = False
        if not use_myelin and not cellshape_only:
            base_dir = f'{bbase_dir}/myelin_ablation/celltype_pts{npts}_ctx{ctx}'
        else:
            base_dir = f'{bbase_dir}//celltype_pts{npts}_ctx{ctx}'
        if cellshape_only:
            base_dir += '_cellshape_only'
        if not use_syntype:  # ignore if cell shape only
            base_dir += '_no_syntype'
        base_dir = bbase_dir
        #mfold = base_dir + '/celltype_CV{}/celltype_pts_j0251v4_scale{}_nb{}_ctx{}_relu{}{}_gn_CV{}_eval{}/'
        #mfold = base_dir + '/celltype_pts_j0251v4_scale{}_nb{}_ctx{}_relu_gn_CV{}_eval0/'
        mfold = base_dir + f'/celltype_pts{npts}_ctx{ctx}' +'/celltype_CV{}/celltype_pts_j0251v4_scale{}_nb{}_ctx{}_relu{}{}_gn_CV{}_eval{}/'


        #mfold = base_dir
        for run in range(n_runs):
            for CV in range(ncv_min, n_cv):
            #for CV in cval:
                mylein_str = "_myelin" if use_myelin else ""
                mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                                              "_cellshapeOnly" if cellshape_only else mylein_str, CV, run)
                #mfold_complete = mfold.format(scale, npts, ctx, CV)
                #mfold_complete = mfold
                mpath = f'{mfold_complete}/{state_dict_fname}'
                if not os.path.isfile(mpath):
                    msg = f"'{mpath}' not found. Skipping entire eval run for {base_dir}."
                    raise ValueError(msg)
        if skip_model:
            continue
        # prepare GT
        check_train_ids = set()
        check_valid_ids = []
        for CV in range(ncv_min, n_cv):
        #for CV in cval:
            ccd = CellCloudDataJ0251(cv_val=CV)
            check_train_ids.update(set(ccd.splitting_dict['train']))
            check_valid_ids.extend(list(ccd.splitting_dict['valid']))
        #assert len(check_train_ids) == len(check_valid_ids)
        assert np.max(np.unique(check_valid_ids, return_counts=True)[1]) == 1
        target_names = [int2str_label[kk] for kk in range(nclasses)]
        csv_p = ccd.csv_p
        df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
        ssv_ids = df[:, 0].astype(np.uint64)
        if len(np.unique(ssv_ids)) != len(ssv_ids):
            raise ValueError('Multi-usage of IDs!')
        str_labels = df[:, 1]
        ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
        ssd_kwargs = dict(working_dir=wd)
        ssd = SuperSegmentationDataset(**ssd_kwargs)
        for redundancy in [1, 10, 20, 50][::-1]:
        #for redundancy in [10]:
            perf_res_dc = collections.defaultdict(list)  # collect for each run
            for run in range(n_runs):
                log = config.initialize_logging(f'log_eval{run}_sp{npts}k_redun{redundancy}', base_dir)
                log.info(f'\nStarting evaluation of model with npoints={npts}, eval. run={run}.\n'
                         f'GT data at wd={wd}\n')
                for CV in range(ncv_min, n_cv):
                #for CV in cval:
                    mylein_str = "_myelin" if use_myelin else ""
                    mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                        "_cellshapeOnly" if cellshape_only else mylein_str, CV, run)
                    #mfold_complete = mfold.format(scale, npts, ctx, CV)
                    mpath = f'{mfold_complete}/{state_dict_fname}'
                    print(mpath)
                    assert os.path.isfile(mpath)
                    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
                    assert loader_kwargs['npoints'] == npts
                    log.info(f'model_kwargs={mkwargs}')
                    ccd = CellCloudDataJ0251(cv_val=CV)
                    split_dc = ccd.splitting_dict
                    if use_myelin or cellshape_only:  # use old splits
                        split_dc['valid'] = valid_splits[CV]
                        del split_dc['train']
                    map_myelin = use_myelin
                    if map_myelin:
                        assert "_myelin" in mpath
                    if not map_myelin and '_cellshapeOnly' not in mpath:
                        assert "_myelin" not in mpath
                    if '_noSyntype' in mpath:
                        assert not use_syntype
                    if '_cellshapeOnly' in mpath:
                        assert cellshape_only
                    mkwargs['mpath'] = mpath
                    log.info(f'Using model "{mpath}" for cross-validation split {CV}.')
                    fname_pred = f'{os.path.split(mpath)[0]}/ctgt_v4_splitting_cv{CV}_redun{redundancy}_{run}_10fold_PRED.pkl'
                    assert fname_pred not in all_res_paths
                    all_res_paths.add(fname_pred)
                    # check pred if available
                    incorrect_pred = False
                    if os.path.isfile(fname_pred) and not overwrite:
                        os.path.isfile(fname_pred)
                        res_dc = basics.load_pkl2obj(fname_pred)
                        res_dc = dict(res_dc)  # convert to standard dict
                        incorrect_pred = (len(res_dc) != len(split_dc['valid'])) or (not np.all([k in split_dc['valid'] for k in res_dc]))
                        if incorrect_pred:
                            print(f'Wrong prediction stored at: {fname_pred}. Recomputing now.')
                    if overwrite or not os.path.isfile(fname_pred) or incorrect_pred:
                        res_dc = predict_celltype_gt(ssd_kwargs, mpath=mpath, bs=20,
                                                     nloader=10, device='cuda', seeded=True, ssv_ids=split_dc['valid'],
                                                     npredictor=2, use_test_aug=False,
                                                     loader_kwargs={'redundancy': redundancy, 'map_myelin': map_myelin,
                                                                    'use_syntype': use_syntype,
                                                                    'cellshape_only': cellshape_only},
                                                     **loader_kwargs)
                        incorrect_pred = (len(res_dc) != len(split_dc['valid'])) or (not np.all([k in split_dc['valid'] for k in res_dc]))
                        if incorrect_pred:
                            raise ValueError('Incorrect prediction.')
                        basics.write_obj2pkl(fname_pred, res_dc)
                valid_ids, valid_ls, valid_preds, valid_certainty = [], [], [], []

                for CV in range(ncv_min, n_cv):
                #for CV in cval:
                    ccd = CellCloudDataJ0251(cv_val=CV)
                    split_dc = ccd.splitting_dict
                    if use_myelin or cellshape_only:  # use old splits
                        split_dc['valid'] = valid_splits[CV]
                        del split_dc['train']
                    mylein_str = "_myelin" if use_myelin else ""
                    mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                                                  "_cellshapeOnly" if cellshape_only else mylein_str, CV, run)
                    #mfold_complete = mfold.format(scale, npts, ctx, CV)
                    mpath = f'{mfold_complete}/{state_dict_fname}'
                    assert os.path.isfile(mpath)
                    fname_pred = f'{os.path.split(mpath)[0]}/ctgt_v4_splitting_cv{CV}_redun{redundancy}_{run}_10fold_PRED.pkl'
                    res_dc = basics.load_pkl2obj(fname_pred)
                    res_dc = dict(res_dc)  # convert to standard dict
                    assert len(res_dc) == len(split_dc['valid'])
                    assert np.all([k in split_dc['valid'] for k in res_dc])
                    valid_ids_local, valid_ls_local, valid_preds_local = [], [], []
                    for ix, curr_id in enumerate(ssv_ids):
                        if curr_id not in split_dc['valid']:
                            continue
                        curr_l = ssv_labels[ix]
                        valid_ls.append(curr_l)
                        curr_pred, curr_cert = res_dc[curr_id]
                        valid_preds.append(curr_pred)
                        valid_certainty.append(curr_cert)
                        valid_ids.append(curr_id)
                valid_preds = np.array(valid_preds)
                valid_certainty = np.array(valid_certainty)
                valid_ls = np.array(valid_ls)
                valid_ids = np.array(valid_ids)
                log.info(f'Final prediction result for run {run} with {loader_kwargs} and {mkwargs}.')
                class_rep = classification_report(valid_ls, valid_preds, labels=np.arange(nclasses), target_names=target_names,
                                                  output_dict=True)
                for ii, k in enumerate(target_names):
                    perf_res_dc[f'fscore_class_{ii}'].append(class_rep[k]['f1-score'])
                perf_res_dc['fscore_macro'].append(f1_score(valid_ls, valid_preds, average='macro'))
                perf_res_dc['accuracy'].append(accuracy_score(valid_ls, valid_preds))
                perf_res_dc['cert_correct'].append(valid_certainty[valid_preds == valid_ls])
                perf_res_dc['cert_incorrect'].append(valid_certainty[valid_preds != valid_ls])
                log.info(classification_report(valid_ls, valid_preds, labels=np.arange(nclasses), target_names=target_names))
                conf_matrix = confusion_matrix(valid_ls, valid_preds, labels=np.arange(nclasses))
                log.info(conf_matrix)
                conf_pd = pd.DataFrame(conf_matrix, columns=target_names, index=target_names)
                conf_pd.to_csv(base_dir + "conf_matrix_%i.csv" % redundancy)
                log.info(f'Mean certainty correct:\t{np.mean(valid_certainty[valid_preds == valid_ls])}\n'
                         f'Mean certainty incorrect:\t{np.mean(valid_certainty[valid_preds != valid_ls])}')
                log.info(f'Incorrectly predicted IDs (ID, label, prediction): '
                         f'{[(ix, int2str_label[label], int2str_label[pred]) for ix, label, pred in zip(valid_ids[valid_preds != valid_ls], valid_ls[valid_preds != valid_ls], valid_preds[valid_preds != valid_ls])]}')
            # plot everything
            perf_res_dc = dict(perf_res_dc)
            model_tag = f'ctx{loader_kwargs["ctx_size"]}_nb{npts}'
            if cellshape_only:
                model_tag += 'cellshapeOnly'
            if not use_syntype:
                model_tag += 'noSyntype'
            if not use_myelin:
                model_tag += 'noMyelin'
            perf_res_dc['model_tag'] = model_tag
            perf_res_dc['ctx'] = ctx
            perf_res_dc['redundancy'] = redundancy
            perf_res_dc['npts'] = npts
            perf_res_dc['cellshape_only'] = cellshape_only
            perf_res_dc['use_syntype'] = use_syntype
            perf_res_dc['use_myelin'] = use_myelin

            basics.write_obj2pkl(f"{base_dir}/redun{redundancy}_prediction_results.pkl", perf_res_dc)
            fscores = np.concatenate([perf_res_dc[f'fscore_class_{ii}'] for ii in range(nclasses)] +
                                     [perf_res_dc[f'fscore_macro'], perf_res_dc['accuracy']]).squeeze()
            labels = np.concatenate([np.concatenate([[int2str_label[ii]] * n_runs for ii in range(nclasses)]),
                                     np.array(['f1_score_macro'] * n_runs + ['accuracy'] * n_runs)])

            df = pandas.DataFrame(data={'quantity': labels, 'f1score': fscores})
            create_catplot(f"{base_dir}/redun{redundancy}_performances.png", qs=df, x='quantity', y='f1score',
                           size=10)

            cert_correct = np.concatenate(perf_res_dc['cert_correct'])
            cert_incorrect = np.concatenate(perf_res_dc['cert_incorrect'])
            df = pandas.DataFrame(data={'quantity': ['correct'] * len(cert_correct) + ['incorrect'] * len(cert_incorrect),
                                        'certainty': np.concatenate([cert_correct, cert_incorrect]).squeeze()})
            create_catplot(f"{base_dir}/redun{redundancy}_certainty.png", qs=df, x='quantity', y='certainty',
                           add_boxplot=True, size=4)
    plot_performance_summary(bbase_dir)
    plot_performance_summary(bbase_dir, include_special_inputs=True)

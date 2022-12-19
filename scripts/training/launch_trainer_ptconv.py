from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    nfold = 10
    params = []
    cnn_script = '/cajal/nvmescratch/users/arother/dev/SyConn/syconn/cnn/cnn_celltype_ptcnv_j0251.py'

    for npoints, ctx in ([50000, 20000],):
        scale = int(ctx / 10)
        for run in range(3):
            base_dir = f'/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811' \
                       f'/celltype_training/221216_celltype_cross_val/celltype_pts{npoints}_ctx{ctx}'
            for cval in range(nfold):
                save_root = f'{base_dir}/celltype_CV{cval}/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx, scale_norm=scale,
                               use_bias=True)])

    params = list(basics.chunkify_successive(params, 1))

    batchjob_script(params, 'launch_trainer', n_cores=10, additional_flags='--time=7-0 --gres=gpu:1 --mem=400000 --cpus-per-task 30 -p p.share',
                    disable_batchjob=False,
                    batchjob_folder=f'/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/celltype_training/221216_celltype_cross_val/',
                    remove_jobfolder=False, overwrite=True)

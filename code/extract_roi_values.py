import pandas as pd
import os.path as op
import numpy as np
import nibabel as nib
from nilearn import image
from glob import glob
from scipy import stats

reli_df = pd.read_csv('../../behav/religiosity_complete.tsv', sep='\t', index_col=0)
subs = sorted(glob('../derivatives/firstlevel/sub-*'))

for i, sub_dir in enumerate(subs):
    sub = op.basename(sub_dir)
    print("Processing %s" % sub)
    
    for con in ['incongruent-congruent', 'incorrect-correct']:
        if con == 'incongruent-congruent':
            colname = 'congruence'
            froi = '../../rois/conflict_association-test_z_FDR_0.01.nii.gz'
        else:
            colname = 'correct'
            froi = '../../rois/error_association-test_z_FDR_0.01.nii.gz'

        for stat in ['betas', 'vars']:
            statname = 'beta' if stat == 'betas' else 'variance'
            statfile = op.join(sub_dir, 'contrast-%s_%s.nii.gz' % (con, stat))
            if op.isfile(statfile):
                acc_resamp = image.resample_to_img('../rois/ACC.nii.gz', statfile, interpolation='nearest')
                #acc_resamp.to_filename(op.join(sub_dir, 'ACC_native.nii.gz'))
                froi_resamp = image.resample_to_img(froi, statfile, interpolation='nearest')
                conj_roi = np.logical_and(acc_resamp.get_data() > 0, froi_resamp.get_data() > 0)
                conj_roi_img = nib.Nifti1Image(conj_roi.astype(int), affine=nib.load(statfile).affine)
                conj_roi_img.to_filename(op.join(sub_dir, 'final_roi_%s.nii.gz' % con))
                stat_data = nib.load(statfile).get_data()
                val = stat_data[conj_roi].mean()
                reli_df.loc[sub, '%s_%s' % (colname, statname)] = val

    if (i+1) % 20 == 0:
        for s in ['congruence', 'correct']:
            vals = reli_df.loc[:, '%s_beta' % s].values
            vals = vals[~np.isnan(vals)]
            print(stats.ttest_1samp(vals, 0))
reli_df.to_csv('../../behav/religiosity_complete_with_roi_values.tsv', sep='\t', index=True)

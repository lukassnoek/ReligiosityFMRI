import os
import os.path as op
import joblib as jl
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nistats.first_level_model import first_level_models_from_bids
from nistats.thresholding import map_threshold
from nistats.reporting import plot_design_matrix
from nilearn.plotting import plot_stat_map
from nilearn.image import math_img

plt.set_cmap('coolwarm')
DATASET_PATH = '../bids'
out = first_level_models_from_bids(
    dataset_path=DATASET_PATH,
    task_label='gstroop',
    space_label='MNI152NLin2009cAsym',
    img_filters=[],
    t_r=2.0,
    slice_time_ref=0.0,
    hrf_model='glover',
    drift_model=None,
    period_cut=128,
    drift_order=1, fir_delays=[0],
    min_onset=-24,
    mask=None,
    target_affine=None,
    target_shape=None,
    smoothing_fwhm=5,
    memory_level=1,
    standardize=False,
    signal_scaling=0,
    noise_model='ar1',
    verbose=100,
    n_jobs=2,
    minimize_memory=True,
    derivatives_folder='derivatives/fmriprep'
)

conf_cols = ['Cosine00', 'Cosine01', 'Cosine02', 'Cosine03', 'Cosine04', 'Cosine05',
             'X', 'Y', 'Z','RotX', 'RotY', 'RotZ']

mods, mod_imgs, mod_events, mod_conf = out
for i, conf in enumerate(mod_conf):
    conf[0] = conf[0].loc[:, conf_cols]
    mod_conf[i] = conf

for i, events in enumerate(mod_events):
    print("Adjusting design for file %s" % mod_imgs[i][0])
    df = events[0]
    df = df.fillna(99)
    df['response_hand'] = df['response_hand'].map({99: 'miss', 1: 'left_hand', 2: 'right_hand'})
    df = pd.concat([df.loc[:, [col, 'onset', 'duration', 'response_time']].rename({col: 'trial_type'}, axis=1) for col in ['trial_type', 'response_type', 'response_hand']]).sort_index()
    df.loc[df.trial_type.str.contains('hand'), 'onset'] += df.loc[df.trial_type.str.contains('hand'), 'response_time']
    df.loc[df.trial_type.str.contains('correct'), 'onset'] += df.loc[df.trial_type.str.contains('correct'), 'response_time']
    df = df.loc[df.trial_type != 'miss', :]
    df = df.loc[~df.trial_type.str.contains('hand'), :]
    #df = df.loc[df.trial_type.str.contains('hand'), :]
    #df['trial_type'] = df['response_hand']
    #df.loc[df.trial_type.str.contains('hand'), 'trial_type'] = 'resp'
    df = df.drop('response_time', axis=1)
    mod_events[i] = [df]

for m, m_img, m_events, m_conf in zip(mods, mod_imgs, mod_events, mod_conf):

    output_dir = op.join(DATASET_PATH, 'derivatives', 'firstlevel', 'sub-%s' % m.subject_label) 
    if not op.isdir(output_dir):
        os.makedirs(output_dir)

    mask = op.join(DATASET_PATH, 'derivatives', 'fmriprep', 'sub-%s' % m.subject_label, 'func',
                   'sub-%s_task-gstroop_acq-sequential_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz' % m.subject_label)
    m.mask = mask
    m.fit(m_img, events=m_events, confounds=m_conf)
    design_matrix = m.design_matrices_[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_design_matrix(design_matrix, ax=ax)
    fig.savefig(op.join(output_dir, 'design.png'))
    plt.close()    

    dm_corr = design_matrix.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(dm_corr, xticklabels=dm_corr.columns.values,
                     yticklabels=dm_corr.columns.values, ax=ax, vmin=-1, vmax=1)
    fig.savefig(op.join(output_dir, 'design_corr.png'))
    plt.close()

    for con in ['incongruent-congruent', 'incorrect-correct']:

        if con == 'incorrect-correct' and 'incorrect' not in design_matrix.columns:
            print("No incorrect trials for sub-%s!" % m.subject_label)
            continue

        out_dict = m.compute_contrast(con, output_type='all')
        zmap_img, pe_img, var_img = out_dict['z_score'], out_dict['effect_size'], out_dict['effect_variance']
        zmap_img_thr = map_threshold(zmap_img, mask_img=m.mask, level=0.05, height_control='fpr')[0]

        #test = map_threshold(math_img('pe / np.sqrt(var)', pe=pe_img, var=var_img), mask_img=m.mask, level=0.05, height_control='fpr')[0]
 
        for name, img in zip(['zvals_thr' , 'betas', 'vars'], [zmap_img_thr, pe_img, var_img]):
            img.to_filename(op.join(output_dir, 'contrast-%s_%s.nii.gz' % (con, name)))

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_stat_map(img, output_file=op.join(output_dir, 'contrast-%s_%s.png' % (con, name)),
                          display_mode='ortho', colorbar=True, figure=fig, axes=ax, title=con,
                          annotate=True, draw_cross=False)
            plt.close()

jl.dump(mods, op.join(DATASET_PATH, 'derivatives', 'firstlevel', 'firstlevel_models.jl'), compress=3)

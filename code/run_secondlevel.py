import os
import joblib as jl
import os.path as op
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold

deriv_dir = '../derivatives'
fl_dir = op.join(deriv_dir, 'firstlevel')
fl_models_file = op.join(fl_dir, 'firstlevel_models.jl')
fl_models = jl.load(fl_models_file)

sl_dir = op.join(deriv_dir, 'secondlevel')
if not op.isdir(sl_dir):
    os.makedirs(sl_dir)

slm = SecondLevelModel(
    mask=None,
    smoothing_fwhm=None,
    memory=None,
    memory_level=1,
    verbose=100,
    n_jobs=1,
    minimize_memory=True
)

for con in ['left_hand-right_hand']:#['incongruent-congruent', 'incorrect-correct']:
    
    if con == 'incorrect-correct':
        models = []
        for model in fl_models:
            if 'incorrect' in model.design_matrices_[0]:
                models.append(model)
    else:
        models = fl_models

    slm.fit(models)

    print("Computing %s contrast ..." % con)
    zmap = slm.compute_contrast(
        second_level_contrast=None,
        first_level_contrast=con,
        second_level_stat_type='t',
        output_type='z_score'
    )

    zmap_thr = map_threshold(stat_img=zmap, mask_img=None, level=0.05, height_control='fdr', cluster_threshold=0)[0]
    zmap.to_filename(op.join(sl_dir, 'zmap_%s.nii.gz' % con))
    zmap_thr.to_filename(op.join(sl_dir, 'zmap_thr_%s.nii.gz' % con))


import shutil
import os
import pandas as pd
import os.path as op
from glob import glob

reli_file = '../religiosity_raw.csv'
df = pd.read_csv(reli_file)
df.loc[:, 'nummer'] = 'sub-' + df.loc[:, 'nummer'].astype(str).str.pad(width=4, fillchar='0')
df = df.set_index('nummer')
df = df.loc[:, 'RELIGIOSITY_KEY']
df = df.dropna(how='any', axis=0)
print("After removing participants without religiosity: %i" % len(df))

# Add covariates
demogr_file = '../behav.tsv'
df_demo = pd.read_csv(demogr_file, sep='\t', index_col=0)
df_demo = df_demo.loc[:, ['age', 'gender', 'raven_score']].dropna(how='any', axis=0)
df = pd.concat((df, df_demo), axis=1, sort=True).dropna(how='any', axis=0)
print("After removing participants without demographics: %i" % len(df))

# Filter for actual MRI data
mris = [op.basename(mri) for mri in sorted(glob('../bids/sub*'))]
df = df.loc[df.index.intersection(mris), :]
print("After removing participants without mri: %i" % len(df))

logs = [op.basename(log).split('_')[0] for log in sorted(glob('../logs/clean/sub-*'))]
df = df.loc[df.index.intersection(logs), :]
print("After removing participants without logfile: %i" % len(df))

for col in ['age', 'raven_score', 'RELIGIOSITY_KEY']:
    df.loc[:, col] = (df.loc[:, col] - df.loc[:, col].mean()) / df.loc[:, col].std()

df.loc[:, 'gender'] = df.loc[:, 'gender'].map({1: 'Male', 2: 'Female'})
df = pd.concat((df, pd.get_dummies(df.loc[:, 'gender'])), axis=1)
df = df.drop(labels='gender', axis=1)

# Filter on behavioral performance
TO_REMOVE = ['sub-0010', 'sub-0030', 'sub-0056', 'sub-0057', 'sub-0069', 'sub-0080', 'sub-0115', 'sub-0137', 'sub-0194', 'sub-0201', 'sub-0216']
#df = df.drop(TO_REMOVE, axis=0)
print("After removing participants with wrong stroop-data: %i" % df.shape[0])

df.to_csv('../religiosity_complete.tsv', sep='\t', index=True)

for sub in df.index:
    log = '../logs/clean/%s_task-gstroop_events.tsv' % sub
    assert(op.isfile(log))

    if op.isdir('../bids/%s' % sub):
        shutil.copyfile(log, '../bids/%s/func/%s' % (sub, op.basename(log)))

# Remove subs with no logfile
bids_subs = sorted(glob('../bids/sub*'))
for sub in bids_subs:
    bsub = op.basename(sub)
    if bsub not in df.index:
         shutil.rmtree(sub)

fmriprep_subs = sorted(glob('../bids/derivatives/fmriprep/sub-????'))
for sub in fmriprep_subs:
    bsub = op.basename(sub)
    if bsub not in df.index:
        shutil.rmtree(sub)

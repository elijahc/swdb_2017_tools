# Set drive path to the brain observatory cache located on hard drive
drive_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'

# Import standard libs
import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt
import load_by_stim as lbs
import plotting as c_plt
from matplotlib.collections import LineCollection
import scipy.ndimage.filters as filt

from swdb2017.brain_observatory.util.cell_specimen_ops import get_run_mod_cells
# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# Set stimlus of interest
stim = 'static_gratings'
# Get all run modulated neurons
#run_mod_cells = get_run_mod_cells(boc, stimuli = stim)  ## based on Saskia's criteria

targeted_structures = ['VISp']
cre_line = ['Rbp4-Cre_KL100']

# Get all ophys experiments with eye tracking data, for spont period
exps_ = boc.get_ophys_experiments(stimuli = ['natural_scenes'], simple = False, targeted_structures=targeted_structures)
exps = []
for i, exp in enumerate(exps_):
    if (exps_[i]['fail_eye_tracking']==False):
        exps.append(exps_[i])

exp_id = exps[1]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id = exp_id)
meta_data = data_set.get_metadata()

# get df/f traces for static_gratings
f_df, pr_df, sg_cell_ids = lbs.get_grating_specific_traces(data_set, False)
nTrials = f_df.shape[0]
stim_id = '90.0,0.08,0.0'
# choose one image (for now) and make numpy array
dff_sg = np.concatenate(f_df[stim_id][0:nTrials].dropna(), axis=1)

#pr_ns = np.concatenate(pr_df['90.0,0.08,0.0'][0:nTrials], axis=0)

# Get mean response over all presentations of given stim for each cell
dff_sg_m = np.mean(dff_sg, axis = 1)

# Check on cell by cell basis for run mod, create trial by trial run mod indices
# and save into single column data frame
rm_df = pd.DataFrame([], index = range(0, nTrials), columns = [stim_id])
for i in range(0, nTrials):
    if ~np.isnan(np.mean(np.mean(f_df[stim_id][i]))):
        trial_m = np.mean(f_df[stim_id][i], axis=1)
        rm_index = (trial_m - dff_sg_m)/(trial_m + dff_sg_m)
        rm_df[stim_id][i] = list(rm_index)


rm_df = np.array((rm_df[stim_id]))  #np.concatenate(rm_df[stim_id][0:nTrials].dropna().transpose(), axis = 0)
print(np.asmatrix(rm_df).shape)

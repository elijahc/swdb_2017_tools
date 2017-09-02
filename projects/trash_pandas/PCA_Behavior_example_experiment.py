'''
Use this script to analyze the relationship between principle components and
behavioral readouts for a given experiment.

Specify the experiments by changing the cre_line, targeted_structures, etc.
Loop will generate list of all experiments meeting above criteria and containing
pupil tracking. From that list, choose an index for analyis. If stim_type =
'natural_scenes' you may also specifiy a list of natural_scenes to be analyzed.
'''


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
from PCA_Behavior_Analysis import PCA_batch
import extract_pupil_features as epf
import extract_running_features as err
import seaborn as sns

# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)


# Get list of all stimuli
stim = boc.get_all_stimuli()
structure = ['VISrl']
cre_line = []


# Get all ophys experiments with eye tracking data, for spont period
exps_ = boc.get_ophys_experiments(stimuli = ['natural_scenes'], simple = False, targeted_structures=structure)
exps = []

stim_type = 'natural_scenes'
images = [0]

for i, exp in enumerate(exps_):
    if (exps_[i]['fail_eye_tracking']==False):
        exps.append(exps_[i])

## choose experiment for analysis
exp_id = exps[1]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id = exp_id)
cell_ids = data_set.get_cell_specimen_ids()
meta_data = data_set.get_metadata()
if len(cell_ids) <= 20:    # Get rid of experiments with less than 20 cells
    sys.exit('less than 20 cells in experiment')

if stim_type == 'spont':
    pca, behavior = PCA_batch(data_set, stim_type = 'spont')

    corr = behavior['corr_mat'].T
    targeted_structure = 'VISam'
    cre_line = meta_data['genotype']
    plt.figure()
    plt.subplot(211)
    plt.title('exp_id: %s, targeted_structures: %s, cre_line: %s' %(exp_id, targeted_structure, cre_line))
    ax = sns.heatmap(corr)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    plt.subplot(212)
    plt.title('PCs needed to explain 50 percent of variance: %s' %pca['fraction_pcs'])
    plt.plot(pca['var_explained'], '.')

elif stim_type == 'natural_scenes':
    for image in images:
        pca, behavior = PCA_batch(data_set, stim_type = 'natural_scenes', images = image)

        corr = behavior['corr_mat'].T
        targeted_structure = 'VISam'
        cre_line = meta_data['genotype']
        plt.figure()
        plt.subplot(211)
        plt.title('exp_id: %s, targeted_structures: %s, cre_line: %s \n image: %s' %(exp_id, targeted_structure, cre_line, image))
        ax = sns.heatmap(corr)
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        for item in ax.get_yticklabels():
            item.set_rotation(0)
        plt.subplot(212)
        plt.title('PCs needed to explain 50 percent of variance: %s' %pca['fraction_pcs'])
        plt.plot(pca['var_explained'], '.')

plt.show()

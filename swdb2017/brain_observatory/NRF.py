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
from collections import Counter
from tqdm import tqdm
from util.pearson_corr_coeff import pearson_corr_coeff

#import seaborn as sns

'''
Function activity of populatio and calculates its "neural receptive
field" for each neuron. Essentially a reverse correlation between single neuron
activity and population activity.
'''

def NRF(exp, t_win):
    '''
    Inputs:
    exp: A data frame containing one given experiment from "get_ophys_experiments"
    t_win: The time window over which to correlate activity of the population
            and a given neuron (in ms)

    Returns:
    NRF_df: A data frame of cells in which each element of the data frame contains
    a cell X time matrix representing that cell's Neural Receptive Field
    '''

    # For now, will do this only with spont data
    f_df, pr_df, spont_cell_ids, t = lbs.get_spont_specific_fluorescence_traces(exp, False)
    step = Counter(np.ediff1d(t)).most_common(1)[0][0]
    fs = 1/(step)
    t_lag = int(np.floor(t_win/step))
    nTrials = len(f_df['spont'])
    dff_spont = np.concatenate(f_df['spont'][0:nTrials], axis=1)
    pr_spont = np.concatenate(pr_df['spont'][0:nTrials], axis = 0)
    nrf_df = np.empty((dff_spont.shape[0], dff_spont.shape[0]-1, t_lag)) #pd.DataFrame([], index = range(0, dff_spont.shape[0]), columns = ['NRF'])
    for neuron in (np.arange(0, dff_spont.shape[0])):           #dff_spont.shape[0]):
        n1 = dff_spont[neuron,:]
        dff = np.delete(dff_spont, neuron, axis=0)
        nrf = np.zeros((dff.shape[0], t_lag))
        for i in range(0, t_lag):
            Css = np.matmul(dff,dff.transpose())/dff.shape[1]
            nrf[:,i] = (np.matmul(np.linalg.inv(Css), np.dot(n1, np.roll(dff,i).T)/dff.shape[1]))
        nrf_df[neuron][:,:] = nrf
    lag = np.arange(0,t_lag*(step), step)
    return nrf_df, lag


# Set drive path to the brain observatory cache located on hard drive
drive_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'

from swdb2017.brain_observatory.util.cell_specimen_ops import get_run_mod_cells
# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# Get list of all stimuli
stim = boc.get_all_stimuli()
# Select brain region of interest and cre lines
targeted_structures = ['VISp']
cre_line = 'Rbp4-Cre_KL100'

# Get all ophys experiments with eye tracking data, for spont period
exps_ = boc.get_ophys_experiments(stimuli = ['natural_scenes'], simple = False, targeted_structures=targeted_structures)
exps = []
for i, exp in enumerate(exps_):
    if (exps_[i]['fail_eye_tracking']==False):
        exps.append(exps_[i])

## Test PCA with one of the experiments in exps
exp_id = exps[1]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id = exp_id)
x, t = NRF(data_set, 15)

fig, ax = plt.subplots(10,10, sharex = True, sharey = True)

for i, a in zip(range(0,91), ax.flatten()):
    nCells = x[i].shape[0]
    im = a.imshow(np.flip(x[i],1), aspect = 'auto', cmap = 'plasma', extent=[max(t), 0, nCells, 0])
fig.text(0.5, 0.04, 'lag (s)', ha='center')
fig.text(0.04, 0.5, 'neurons', va='center', rotation='vertical')
fig.colorbar(im, ax = ax.ravel().tolist())
plt.show()

c_mat = np.zeros((90,90))
for i in range(0, 90):
    for j in range(0,90):
        c_mat[i,j] = pearson_corr_coeff(np.delete(x[i], j, axis=0), np.delete(x[j], i, axis=0))
plt.figure()
plt.imshow(np.triu(c_mat, k=1), cmap = 'plasma')
plt.colorbar()
plt.show()

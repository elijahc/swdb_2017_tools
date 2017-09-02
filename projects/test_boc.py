# Set drive path to the brain observatory cache located on hard drive
drive_path = '~/media/charlie/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'

# Import standard libs
import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt

# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
print manifest_file

boc = BrainObservatoryCache(manifest_file=manifest_file)

############   Get some information about the data contained in boc ###########

# List of all targeted structures in visual cortex
targeted_structures = boc.get_all_targeted_structures()

# Download a list of all imaging depths
depths = boc.get_all_imaging_depths()

# Download a list of all cre driver lines
cre_lines = boc.get_all_cre_lines()

# Download a list of all stimuli
stims = boc.get_all_stimuli()

#### Choose visual area to look at
visual_area = 'VISp'
cre_line = 'Rbp4-Cre_KL100'

exps = boc.get_experiment_containers(targeted_structures=[visual_area], cre_lines=[cre_line])

expt_container_id = 555040113 ## Taken from Allen website - picked one with eye tracking

expt_session_info = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id])

# Get session id for session with natural images
session_id = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id], stimuli = ['natural_scenes'])[0]['id']


# Now, get data for this session of this experiment
data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)

# save pupil data
pr = data_set.get_pupil_size()

# get df/f traces for this session
cell_ids = data_set.get_cell_specimen_ids()
fl = data_set.get_fluorescence_traces(cell_ids)
t = fl[0]       # time stamps
r = fl[1][:,:]  # fluroescence responses

plt.figure()
plt.plot(pr[1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(r)
ax.set_aspect('auto')




### singluar value decompostition of fluroescence traces
'''
import scipy.sparse.linalg as la
print(r.shape)
U, S, V = la.svds(r, k=63)

print(U.shape)
print(S.shape)
print(V.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(V)
ax.set_aspect('auto')

plt.figure()
plt.imshow(U)


plt.show()
'''

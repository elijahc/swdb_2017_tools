import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from trash_cache import TrashCache


'''
---------------------------------------------------------------------------
Script for analyzing PCA data that is stored in the trash_cache
-------------------------------------------------------------------------
'''

stim_type = "natural_scenes"
images = [1,10,20,30,40,50]
# ------------------- plotting and analysis for natural scenes -----------------
if stim_type == 'natural_scenes':

    for image in images:
        manifest_path = os.path.join('/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/nsPCA/split_by_run', str(image))
        fig_path = os.path.join('/home/charlie/Desktop/PCA_plots/nat_scenes/split_by_run',str(image))
        os.makedirs(fig_path)

        tc = TrashCache(os.path.join(manifest_path, 'trash_cache_manifest.json'))
        exps = os.listdir(os.path.join(manifest_path, 'exps'))

        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        drive_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'
        manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
        boc = BrainObservatoryCache(manifest_file=manifest_file)
        # create plot of variance explained across brain regions
        fr_pm = []
        fr_p = []
        fr_rl = []
        fr_al = []
        fr_am = []
        fr_l = []

        fr_pm_run = []
        fr_p_run = []
        fr_rl_run = []
        fr_al_run = []
        fr_am_run = []
        fr_l_run = []

        fr_pm_paired = [[],[]]
        fr_p_paired = [[],[]]
        fr_rl_paired = [[],[]]
        fr_al_paired = [[],[]]
        fr_am_paired = [[],[]]
        fr_l_paired = [[],[]]

        fig, ax = plt.subplots(1,2)
        for exp in exps:
            spontPCA_data = tc.load_experiments([exp])
            #print(spontPCA_data[0].keys())
            target_area = spontPCA_data[0]['meta_data']['targeted_structure']

            if  target_area == 'VISpm':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_pm.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_pm_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-r')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-r')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_pm_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_pm_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])

                else:
                    continue
            elif target_area == 'VISp':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_p.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_p_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-y')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-y')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_p_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_p_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])
                else:
                    continue
            elif target_area == 'VISrl':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_rl.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_rl_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-g')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-g')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_rl_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_rl_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])
                else:
                    continue
            elif target_area == 'VISal':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_al.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_al_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-p')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-p')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_al_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_al_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])
                else:
                    continue
            elif target_area == 'VISam':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_am.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_am_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-k')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-k')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_am_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_am_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])
                else:
                    continue
            elif target_area == 'VISl':
                if spontPCA_data[0]['Fraction of PCs'] != []:
                    fr_l.append(spontPCA_data[0]['Fraction of PCs'])
                if spontPCA_data[0]['Fraction of PCs_run'] != []:
                    fr_l_run.append(spontPCA_data[0]['Fraction of PCs_run'])
                if spontPCA_data[0]['var_explained'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
                    ax[0].plot(x, spontPCA_data[0]['var_explained'], '-b')
                if spontPCA_data[0]['var_explained_run'] != []:
                    x = np.linspace(0,1, len(spontPCA_data[0]['var_explained_run']))
                    ax[1].plot(x, spontPCA_data[0]['var_explained_run'], '-b')
                if (spontPCA_data[0]['Fraction of PCs_run'] != [] and spontPCA_data[0]['Fraction of PCs'] != []):
                    print('pair')
                    fr_l_paired[0].append(spontPCA_data[0]['Fraction of PCs'])
                    fr_l_paired[1].append(spontPCA_data[0]['Fraction of PCs_run'])
                else:
                    continue


        ax[0].set_xlabel('Fraction of Dimensions')
        ax[0].set_ylabel('Variance explained')
        ax[0].set_title('Stationary trials, image: %s' %image)
        ax[1].set_xlabel('Fraction of Dimensions')
        ax[1].set_ylabel('Variance explained')
        ax[1].set_title('Run trials, image: %s' %image)
        plt.savefig(os.path.join(fig_path, 'var.svg'))

        data = [fr_p, fr_l, fr_al, fr_pm, fr_am, fr_rl]
        plt.figure()
        plt.boxplot(data)
        plt.title('Fraction of dimensions need to explain 50 percent of variance \n image: %s, Stationary' %image)
        plt.ylabel('Fraction of dimensions')
        plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
        plt.savefig(os.path.join(fig_path, 'var_explained_stationary.svg'))

        data = [fr_p_run, fr_l_run, fr_al_run, fr_pm_run, fr_am_run, fr_rl_run]
        plt.figure()
        plt.boxplot(data)
        plt.title('Fraction of dimensions need to explain 50 percent of variance \n image: %s, Running' %image)
        plt.ylabel('Fraction of dimensions')
        plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
        plt.savefig(os.path.join(fig_path, 'var_explained_run.svg'))

        data = [fr_p_paired, fr_l_paired, fr_al_paired, fr_pm_paired, fr_am_paired, fr_rl_paired]
        print(fr_p_paired)
        plt.figure()
        for i, item in enumerate(data):
            if item[0] !=[]:
                xpoints = np.hstack((i*np.ones(len(item[0])), (i+0.5)*np.ones(len(item[0]))))
                plt.plot(xpoints, item, '-b')


        data_set = boc.get_ophys_experiment_data(ophys_experiment_id = int(exp))
        stim_template = data_set.get_stimulus_template('natural_scenes')
        plt.figure()
        plt.imshow(stim_template[image], cmap = 'gray')
        plt.title("%s" %image)
        plt.savefig(os.path.join(fig_path, 'image.svg'))

# --------------------- Plotting for spont analysis ---------------------------

elif stim_type == 'spont':
    manifest_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/spontPCA/'
    fig_path = '/home/charlie/Desktop/PCA_plots/spont/'
    os.makedirs(fig_path)

    tc = TrashCache(os.path.join(manifest_path, 'trash_cache_manifest.json'))
    exps = os.listdir(os.path.join(manifest_path, 'exps'))

    # create plot of variance explained across brain regions
    fr_pm = []
    fr_p = []
    fr_rl = []
    fr_al = []
    fr_am = []
    fr_l = []
    p_corr_VISpm = []
    p_corr_VISp = []
    p_corr_VISrl = []
    p_corr_VISal = []
    p_corr_VISam = []
    p_corr_VISl = []
    r_corr_VISpm = []
    r_corr_VISp = []
    r_corr_VISrl = []
    r_corr_VISal = []
    r_corr_VISam = []
    r_corr_VISl = []
    plt.figure()
    for exp in exps:
        spontPCA_data = tc.load_experiments([exp])
        #print(spontPCA_data[0].keys())
        target_area = spontPCA_data[0]['meta_data']['targeted_structure']
        print(spontPCA_data[0]['corr_mat'].T.keys())

        if  target_area == 'VISpm':
            fr_pm.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISpm.append(max(abs(spontPCA_data[0]['corr_mat'].T['pupil smooth'])))
            r_corr_VISpm.append(max(abs(spontPCA_data[0]['corr_mat'].T['running speed smooth'])))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-r')
        elif target_area == 'VISp':
            fr_p.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISp.append(max(abs(spontPCA_data[0]['corr_mat'].T['pupil smooth'])))
            r_corr_VISp.append(max(abs(spontPCA_data[0]['corr_mat'].T['running speed smooth'])))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-y')
        elif target_area == 'VISrl':
            fr_rl.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISrl.append(max(abs(spontPCA_data[0]['corr_mat'].T['pupil smooth'])))
            r_corr_VISrl.append(max(abs(spontPCA_data[0]['corr_mat'].T['running speed smooth'])))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-g')
        elif target_area == 'VISal':
            fr_al.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISal.append(max(abs(spontPCA_data[0]['corr_mat'].T['pupil smooth'])))
            r_corr_VISal.append(max(abs(spontPCA_data[0]['corr_mat'].T['running speed smooth'])))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-p')
        elif target_area == 'VISam':
            fr_am.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISam.append(max(abs(spontPCA_data[0]['corr_mat'].T['pupil smooth'])))
            r_corr_VISam.append(max(abs(spontPCA_data[0]['corr_mat'].T['running speed smooth'])))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-k')
        elif target_area == 'VISl':
            fr_l.append(spontPCA_data[0]['Fraction of PCs'])
            p_corr_VISl.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
            r_corr_VISl.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
            x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
            plt.plot(x, spontPCA_data[0]['var_explained'], '-b')


    plt.xlabel('Fraction of Dimensions')
    plt.ylabel('Variance explained')
    plt.savefig(os.path.join(fig_path, 'var.svg'))

    data = [fr_p, fr_l, fr_al, fr_pm, fr_am, fr_rl]
    plt.figure()
    plt.boxplot(data)
    plt.title('Fraction of dimensions need to explain 50 percent of variance \n spont')
    plt.ylabel('Fraction of dimensions')
    plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
    plt.savefig(os.path.join(fig_path, 'var_explained.svg'))

    data = [p_corr_VISp, p_corr_VISl, p_corr_VISal, p_corr_VISpm, p_corr_VISam, p_corr_VISrl]
    plt.figure()
    plt.boxplot(data)
    plt.title('PC correlation with pupil area \n sponts')
    plt.ylabel('Pearson correlation coefficient')
    plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
    plt.savefig(os.path.join(fig_path, 'pup_corr.svg'))

    data = [r_corr_VISp, r_corr_VISl, r_corr_VISal, r_corr_VISpm, r_corr_VISam, r_corr_VISrl]
    plt.figure()
    plt.boxplot(data)
    plt.title('PC correlation with running speed \n sponts')
    plt.ylabel('Pearson correlation coefficient')
    plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
    plt.savefig(os.path.join(fig_path, 'run_corr.svg'))

plt.show()

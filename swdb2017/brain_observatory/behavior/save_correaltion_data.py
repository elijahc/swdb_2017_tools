
#Imports
import swdb2017.brain_observatory.behavior.correlate_fluor_behavior as cfb
import swdb2017.brain_observatory.behavior.correlation_matrix as cm

from trash_cache import TrashCache
tc = TrashCache(manifest_fp='/home/fionag/tpc/trash_cache_manifest.json')

def save_behavior_correlations(boc, tc, experiment_ids=[], features=[], figure=False):
    """
    Save pearson correlation coefficients for behavior-behavior correlations, and behavior-fluorescence trace
    correlations (natural scenes and spontaneous activity only) from Allen Brain Observatory data.

    Parameters:
    ----------
    boc : BrainObservatory Cache instance
    tc : TrashCache instance
    experiment_ids : list
        list of ophys experiment ids to get correlations for
    features : list
        list of behavioral features to get correlations for. Options = 'pupil_area_rate',
        'saccade_rate', 'pupil_area_smooth', 'running_speed_smooth', 'running_rate_smooth'
    figure : boolean
        if True, will save an image of the correlation matrix to the TrashCache

    Returns:
    -------
    dict : saves information (correlation dataframes, metadata, figure (optional) as a dictionary to the TrashCache

    """
    for i in experiment_ids:
        # Load data
        exp_data = boc.get_ophys_experiment_data(i)
        exp_meta = exp_data.get_metadata()

        # Calculate correlations
        if exp_meta['session_type'] == 'three_session_B':
            ns_behavior_corr_df = cfb.corr_ns_behavior(exp_data, raw=False)
        else:
            ns_behavior_corr_df = ['no_data']

        spont_behavior_corr_df = cfb.corr_spont_behavior(exp_data, raw=False)
        behavior_behavior_corr_df = cm.get_correlations_from_features(boc, features, exp_id=i, figure=figure)

        # Save data
        tc.save_experiments([{'id': i, 'vars': [{'name': 'ns_behavior_correlation', 'data': ns_behavior_corr_df},
                                                {'name': 'spont_behavior_correlation', 'data': spont_behavior_corr_df},
                                                {'name': 'behavior_behavior_correlation',
                                                 'data': behavior_behavior_corr_df},
                                                {'name': 'metadata', 'data': exp_meta}]
                              }])
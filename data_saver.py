# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import os
import pickle
import numpy as np
import json
import pdb
from datetime import datetime

def save_results_single(rs, save_path, trainer):
    """
    Save the results 

    Parameters
    ----------
    rs : class ResultSet
        Contains the different results to be saved
    save_path : str
        Path where to save the data.

    Returns
    -------
    None.

    """

    os.makedirs(save_path)

    with open(save_path + "/train_labels_orig.pickle", 'wb') as outfile:
        pickle.dump(rs.train_labels_orig, outfile)

    with open(save_path + "/distribution_train.pickle", 'wb') as outfile:
        pickle.dump(rs.classes_count, outfile)

    with open(save_path+"/parameters.json", 'w') as outfile:
        json.dump(rs.parameters, outfile)

    np.save(save_path + "/evaluation_original", rs.eval_orig)
    np.save(save_path+'/var_original_classes_prediction', rs.classes_pred_orig)
    np.save(save_path+'/var_original_accuracy', rs.acc_orig)
    np.save(save_path+'/classes_templates', rs.classes_templates)

    if trainer.training_type!="uniform":
        with open(save_path + "/train_labels_shfl.pickle", 'wb') as outfile:
            pickle.dump(rs.train_labels_shfl, outfile)
    
        np.save(save_path + "/evaluation_shuffled", rs.eval_shfl)
        np.save(save_path+'/var_shuffle_classes_prediction', rs.classes_pred_shfl)
        np.save(save_path+'/var_shuffle_accuracy', rs.acc_shfl)

    # np.save(save_path+'/autocorr_original', rs.atc_orig)
    # np.save(save_path+'/autocorr_shuffle', rs.atc_shfl)

    print('Saved all results to {0:s} subfolders'.format(save_path))


def save_results(rs, save_root):
    """
    Save the results 

    Parameters
    ----------
    rs : class ResultSet
        Contains the different results to be saved
    save_path : str
        Path where to save the data.

    Returns
    -------
    None.

    """

    save_folder = "T{0:.3f}_Memory{1:d}_{2:s}".format(rs.T, rs.memory_sz, datetime.now().strftime("%y%m%d_%H%M%S"))
    save_path = save_root + save_folder
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/train_labels_orig.pickle", 'wb') as outfile:
        pickle.dump(rs.train_labels_orig, outfile)

    with open(save_path + "/distribution_train.pickle", 'wb') as outfile:
        pickle.dump(rs.classes_count, outfile)

    with open(save_path+"/parameters.json", 'w') as outfile:
        json.dump(rs.parameters, outfile)

    with open(save_path + "/labels_heatmap_orig.pickle", 'wb') as outfile:
        pickle.dump(rs.lbls_htmp_orig, outfile)

    if rs.save_um_distances:
        np.save(save_path + "/evaluation_original", rs.eval_orig)

    np.save(save_path+'/var_original_classes_prediction', rs.classes_pred_orig)
    np.save(save_path+'/var_original_accuracy', rs.acc_orig)

    np.save(save_path+'/classes_templates', rs.classes_templates)

    if rs.enable_shuffling:
        for block_size_shuffle in rs.train_labels_shfl.keys():
            save_path = save_root + save_folder + '/shuffle_%d/'%(block_size_shuffle)
            os.makedirs(save_path)

            with open(save_path + "/train_labels_shfl.pickle", 'wb') as outfile:
                pickle.dump(rs.train_labels_shfl[block_size_shuffle], outfile)

            with open(save_path + "/labels_heatmap_shfl.pickle", 'wb') as outfile:
                pickle.dump(rs.lbls_htmp_shfl[block_size_shuffle], outfile)

            if rs.save_um_distances:
                np.save(save_path + "/evaluation_shuffled", rs.eval_shfl[block_size_shuffle])
            np.save(save_path+'/var_shuffle_classes_prediction', rs.classes_pred_shfl[block_size_shuffle])
            np.save(save_path+'/var_shuffle_accuracy', rs.acc_shfl[block_size_shuffle])

    print('Saved all results to {0:s} subfolders'.format(save_root))


def save_orig_results(rs, save_path):
    """
    Saves only the results for the original sequence
    To be called right after training on the original sequence 

    Parameters
    ----------
    rs : class ResultSet
        Contains the different results to be saved
    save_path : str
        Path where to save the data.

    Returns
    -------
    None.

    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/train_labels_orig.pickle", 'wb') as outfile:
        pickle.dump(rs.train_labels_orig, outfile)

    with open(save_path + "/distribution_train.pickle", 'wb') as outfile:
        pickle.dump(rs.classes_count, outfile)

    with open(save_path+"/parameters.json", 'w') as outfile:
        json.dump(rs.parameters, outfile)

    with open(save_path + "/labels_heatmap_orig.pickle", 'wb') as outfile:
        pickle.dump(rs.lbls_htmp_orig, outfile)

    if rs.save_um_distances:
        np.save(save_path + "/evaluation_original", rs.eval_orig)

    np.save(save_path+'/var_original_classes_prediction', rs.classes_pred_orig)
    np.save(save_path+'/var_original_accuracy', rs.acc_orig)

    np.save(save_path+'/classes_templates', rs.classes_templates)

    print('Saved results for original sequence to {0:s}'.format(save_path))


def save_shuffle_results(rs, save_path, shfl_sz):
    """
    Saves only the results for the original sequence
    To be called right after training on the original sequence 

    Parameters
    ----------
    rs : class ResultSet
        Contains the different results to be saved
    save_path : str
        Path where to save the data.
    shfl_sz: int
        Shuffle size for which results are to be saved. This should be a key in the rs shuffle dicts

    Returns
    -------
    None.

    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if rs.enable_shuffling:
        if shfl_sz in rs.train_labels_shfl.keys():
            save_path = save_path + '/shuffle_%d/'%(shfl_sz)
            os.makedirs(save_path)

            with open(save_path + "/train_labels_shfl.pickle", 'wb') as outfile:
                pickle.dump(rs.train_labels_shfl[shfl_sz], outfile)

            with open(save_path + "/labels_heatmap_shfl.pickle", 'wb') as outfile:
                pickle.dump(rs.lbls_htmp_shfl[shfl_sz], outfile)

            if rs.save_um_distances:
                np.save(save_path + "/evaluation_shuffled", rs.eval_shfl[shfl_sz])
            np.save(save_path+'/var_shuffle_classes_prediction', rs.classes_pred_shfl[shfl_sz])
            np.save(save_path+'/var_shuffle_accuracy', rs.acc_shfl[shfl_sz])

            print('Saved all results for shuffle size {0:d} to {1:s}'.format(shfl_sz, save_path))

        else:
            print('Could not find shuffle size {0:d} in the result set dicts... No result was saved for this shuffle size'.format(shfl_sz))

    else:
        print('This result set does not produce shuffled sequences. No data was saved for shuffling scenario.')
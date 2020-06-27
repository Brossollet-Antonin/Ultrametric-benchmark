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
        for block_size_shuffle in rs.eval_shfl.keys():
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
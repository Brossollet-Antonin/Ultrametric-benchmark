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

def save_results(rs, save_path, trainer):
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


def save_results_multiblocks(rs, save_root):
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

    for block_size_shuffle in rs.eval_shfl.keys():
        save_folder = "T%.3f_Memory%d_block%d_%s" % (T, memory_sz, block_size_shuffle, datetime.now().strftime("%y%m%d_%H%M%S"))
        save_path = save_root + save_folder
        os.makedirs(save_path)

        with open(save_path + "/train_labels_shfl.pickle", 'wb') as outfile:
            pickle.dump(rs.train_labels_shfl[block_size_shuffle], outfile)

        np.save(save_path + "/evaluation_shuffled", rs.eval_shfl[block_size_shuffle])
        np.save(save_path+'/var_shuffle_classes_prediction', rs.classes_pred_shfl[block_size_shuffle])
        np.save(save_path+'/var_shuffle_accuracy', rs.acc_shfl[block_size_shuffle])

    print('Saved all results to {0:s} subfolders'.format(save_path))


def chunk_and_save(arr, max_size):
    return


def save_results_by_chunks(rs, save_path):
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

    outfile = open(save_path + "/train_data_orig.pickle", 'wb')
    pickle.dump(rs.train_data_orig, outfile)
    outfile.close()

    outfile = open(save_path + "/train_labels_orig.pickle", 'wb')
    pickle.dump(rs.train_labels_orig, outfile)
    outfile.close()


    outfile = open(save_path + "/train_data_shfl.pickle", 'wb')
    pickle.dump(rs.train_data_shfl, outfile)
    outfile.close()

    outfile = open(save_path + "/train_labels_shfl.pickle", 'wb')
    pickle.dump(rs.train_labels_shfl, outfile)
    outfile.close()

    outfile = open(save_path + "/distribution_train.pickle", 'wb')
    pickle.dump(rs.classes_count, outfile)
    outfile.close()

    outfile = open(save_path+"/parameters.pickle", 'wb')
    pickle.dump(rs.parameters, outfile)
    outfile.close()

    np.save(save_path + "/evaluation_original", rs.eval_orig)
    np.save(save_path + "/evaluation_shuffled", rs.eval_shfl)

    np.save(save_path+'/var_original_classes_prediction', rs.classes_pred_orig)
    np.save(save_path+'/var_shuffle_classes_prediction', rs.classes_pred_shfl)
    np.save(save_path+'/var_original_accuracy', rs.acc_orig)
    np.save(save_path+'/var_shuffle_accuracy', rs.acc_shfl)

    # np.save(save_path+'/autocorr_original', rs.atc_orig)
    # np.save(save_path+'/autocorr_shuffle', rs.atc_shfl)

    print('Saved all results to {0:s} subfolders'.format(save_path))
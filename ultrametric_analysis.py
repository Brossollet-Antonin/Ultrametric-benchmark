# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:48:43 2019

@author: Antonin
"""

import os
import pdb
from copy import deepcopy
import random
import numpy as np
import torch

from utils import verbose, get_lbl_distr
from trainer import Trainer
import neuralnet
import sequence_generator_temporal
import time

import data_saver
from datetime import datetime

def train_sequenceset(trainer, args, block_sizes, rs, save_root):

    rs.sequence_type = args.sequence_type
    rs.enable_shuffling = args.enable_shuffling
    rs.save_um_distances = args.save_um_distances

    rs.classes_templates = trainer.dataset.patterns
    save_folder = "T{0:.3f}_Memory{1:d}_{2:s}".format(rs.T, rs.memory_sz, datetime.now().strftime("%y%m%d_%H%M%S"))

    # Define a family of models, all based on the same architecture template
    trainer.assign_model(deepcopy(trainer.network_tmpl))

    # rs.eval_orig will contain, for each test, an analysis of the ultrametric distance between the labels and the predictions
    eval_orig = trainer.evaluate_hierarchical()
    if args.save_um_distances:
        rs.eval_orig = [eval_orig,]

    # Counting the number of correct responses per classes before the training
    rs.acc_orig = np.array([[eval_orig[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, dim 1 of diagnos_original contains the accuracies at different levels
    nbr_test_samples = trainer.dataset.class_sz_test*(trainer.dataset.branching**trainer.dataset.depth)    # number of test examples
    rs.lbls_htmp_orig = np.zeros((args.test_nbr, trainer.n_classes))

    classes_correct = np.zeros(len(trainer.dataset.test_data))
    for k in range(nbr_test_samples):
        classes_correct[int(eval_orig[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the k-th test example

    rs.classes_pred_orig = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

    #-----------------------------------#
    #---- Data sequence generation -----#
    #-----------------------------------#
    verbose('Data generation...', args.verbose)
    trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
    if hasattr(trainer, 'rates_vector'):
        rs.parameters["Timescales"]=trainer.rates_vector.tolist():
    else:
        rs.parameters["Timescales"] = []
    rs.train_labels_orig = trainer.train_sequence

    verbose('...done\n', args.verbose, 2)

    for test_id in range(args.test_nbr):
        training_range = (test_id*args.test_stride, (test_id+1)*args.test_stride)     #Part of the sequence on which the training will be done
        verbose('Training network on original sequence...', args.verbose, 2)

        trainer.train(
            mem_sz = trainer.memory_size,
            training_range = training_range,
            verbose_lvl = args.verbose
            )

        verbose('...done\nComputing performance for original sequence...', args.verbose, 2)

        eval_orig = trainer.evaluate_hierarchical()
        if args.save_um_distances:
            rs.eval_orig.append(eval_orig)
        rs.lbls_htmp_orig[test_id,:] = get_lbl_distr(trainer.train_sequence, training_range[0], training_range[1], trainer.n_classes)

        verbose('...done\n', args.verbose, 2)

        original_accuracy_current = eval_orig[0][0]      # Recover the standard accuracy
        original_accuracy_current = np.array([[original_accuracy_current, (test_id+1)*args.test_stride]])
        rs.acc_orig = np.append(rs.acc_orig, original_accuracy_current, axis=0)

        classes_correct = np.zeros(len(trainer.dataset.test_data))
        for k in range(nbr_test_samples):
            classes_correct[int(eval_orig[1][k][0])] +=1
        classes_correct = np.array([[classes_correct, (test_id+1)*args.test_stride]])
        rs.classes_pred_orig = np.append(rs.classes_pred_orig, classes_correct, axis=0)

        verbose(
            'Accuracy on original sequence at pos {seq_pos:d} ({n_test_spls:d} test images): {acc:.2f}%'.format(
                seq_pos = training_range[1],
                n_test_spls = nbr_test_samples,
                acc= original_accuracy_current[0][0]
            ), args.verbose
        )

    rs.classes_count = [0 for k in range(len(trainer.dataset.train_data))]
    for k in rs.train_labels_orig:
        rs.classes_count[k] += 1

    data_saver.save_orig_results(rs, os.path.join(save_root,save_folder))

    if rs.enable_shuffling:
        verbose("--- Start shuffle training ---", args.verbose)
        # Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the
        # sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)

        trainer.assign_model(deepcopy(trainer.network_tmpl))
        eval_shfl = trainer.evaluate_hierarchical()
        acc_shfl = np.array([[eval_shfl[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_shuffle contains the accuracies at different levels

        # Counting the number of correct responses per classes before the training
        classes_correct = np.zeros(len(trainer.dataset.test_data))
        for k in range(nbr_test_samples):
            classes_correct[int(eval_shfl[1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example

        rs.train_labels_shfl = {block_size: [] for block_size in block_sizes}
        rs.classes_pred_shfl = {block_size: np.array([[classes_correct, 0]]) for block_size in block_sizes} # This array will stock the prediction of the network during the training
        if args.save_um_distances:
            rs.eval_shfl = {block_size: [eval_shfl,] for block_size in block_sizes}
        rs.acc_shfl = {block_size: acc_shfl for block_size in block_sizes}
        rs.lbls_htmp_shfl = {}

        # trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
        for block_size_shuffle in block_sizes:

            rs.lbls_htmp_shfl[block_size_shuffle] = np.zeros((args.test_nbr, trainer.n_classes))
            
            for test_id in range(args.test_nbr):
                trainer.assign_model(deepcopy(trainer.network_tmpl))
                training_range = (test_id*args.test_stride, (test_id+1)*args.test_stride)
                shuffled_sequence = trainer.shuffle_block_partial(block_size_shuffle, training_range[1])

                #trainer.train(seq=shuffled_sequence, mem_sz=trainer.memory_size, lr=args.lr, momentum=0.5, training_range=training_range)
                trainer.train(
                    mem_sz=trainer.memory_size,
                    training_range=(0, training_range[1]),
                    seq = shuffled_sequence,
                    verbose_lvl = args.verbose
                )
                eval_shfl = trainer.evaluate_hierarchical()
                if args.save_um_distances:
                    rs.eval_shfl[block_size_shuffle].append(eval_shfl)
                shuffle_accuracy_current = eval_shfl[0][0]      # Recover the standard accuracy
                shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (test_id+1)*args.test_stride]])
                rs.acc_shfl[block_size_shuffle] = np.append(rs.acc_shfl[block_size_shuffle], shuffle_accuracy_current, axis=0)

                rs.lbls_htmp_shfl[block_size_shuffle][test_id,:] = get_lbl_distr(trainer.train_sequence, training_range[0], training_range[1], trainer.n_classes)

                classes_correct = np.zeros(len(trainer.dataset.test_data))
                for k in range(nbr_test_samples):
                    classes_correct[int(eval_shfl[1][k][0])] +=1
                classes_correct = np.array([[classes_correct, (test_id+1)*args.test_stride]])
                rs.classes_pred_shfl[block_size_shuffle] = np.append(rs.classes_pred_shfl[block_size_shuffle], classes_correct, axis=0)
                verbose(
                    'Accuracy on shuffled sequence (block size {block_size:d}) at pos {seq_pos:d} ({n_test_spls:d} test images): {acc:.2f}%'.format(
                        block_size = block_size_shuffle,
                        seq_pos = training_range[1],
                        n_test_spls = nbr_test_samples,
                        acc= shuffle_accuracy_current[0][0]
                    ), args.verbose
                )

            rs.train_labels_shfl[block_size_shuffle] = shuffled_sequence

            data_saver.save_shuffle_results(rs, os.path.join(save_root,save_folder), block_size_shuffle)

    return rs
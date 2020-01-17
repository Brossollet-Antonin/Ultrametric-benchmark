# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:48:43 2019

@author: Antonin
"""

import pdb
from copy import deepcopy
import random
import numpy as np
import torch

from local_tools import verbose
import control
from trainer import Trainer
import neuralnet
import sequence_generator_temporal

class ResultSet:
    def __init__(self):
        pass


def ultrametric_analysis(trainer, args, block_size_shuffle):

    rs = ResultSet()

    trainer.network = trainer.network_orig
    rs.eval_orig = trainer.evaluate_hierarchical()

    rs.acc_orig = np.array([[rs.eval_orig[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, dim 1 of diagnos_original contains the accuracies at different levels
    nbr_test_samples = trainer.dataset.class_sz_test*(trainer.dataset.branching**trainer.dataset.depth)    # number of test examples
    classes_correct = np.zeros(len(trainer.dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
    # Compting the number of correct responses per classes before the training

    for k in range(nbr_test_samples):
        classes_correct[int(rs.eval_orig[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the k-th test example

    rs.classes_pred_orig = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

    verbose('Data generation...', args)
    trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training

    rs.train_data_orig, rates, rs.train_labels_orig = trainer.train_sequence

    verbose('...done\n', args)

    rs.atc_orig = sequence_generator_temporal.sequence_autocor(rs.train_labels_orig, n_labels=trainer.dataset.num_classes)


    for i in range(args.test_nbr):
        training_range = (i*args.test_stride, (i+1)*args.test_stride)     #Part of the sequence on which the training will be done
        verbose('Training network on original sequence...', args, 2)

        trainer.train(
            mem_sz = trainer.memory_size,
            lr = args.lr,
            momentum = 0.5,
            training_range = training_range
            )

        verbose('...done\nComputing performance for original sequence...', args, 2)

        rs.eval_orig = trainer.evaluate_hierarchical()

        verbose('...done\n', args, 2)

        original_accuracy_current = rs.eval_orig[0][0]      # Recover the standard accuracy
        original_accuracy_current = np.array([[original_accuracy_current, (i+1)*args.test_stride]])
        rs.acc_orig = np.append(rs.acc_orig, original_accuracy_current, axis=0)

        classes_correct = np.zeros(len(trainer.dataset.test_data))
        for k in range(nbr_test_samples):
            classes_correct[int(rs.eval_orig[1][k][0])] +=1
        classes_correct = np.array([[classes_correct, (i+1)*args.test_stride]])
        rs.classes_pred_orig = np.append(rs.classes_pred_orig, classes_correct, axis=0)

        verbose('Accuracy of the network on the {0:d} test images: {1:.2f}%'.format(nbr_test_samples, original_accuracy_current[0][0]), args)



    verbose("--- Start shuffle training ---", args)
    # Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the
    # sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)

    trainer.network = trainer.network_shfl
    rs.eval_shfl = trainer.evaluate_hierarchical()
    rs.acc_shfl = np.array([[rs.eval_shfl[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_shuffle contains the accuracies at different levels
    classes_correct = np.zeros(len(trainer.dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
    # Compting the number of correct responses per classes before the training

    rs.atc_shfl = []

    for k in range(nbr_test_samples):
        classes_correct[int(rs.eval_shfl[1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example
    rs.classes_pred_shfl = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

    # trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
    
    for i in range(args.test_nbr):
        training_range = (i*args.test_stride, (i+1)*args.test_stride)
        control_data_shuffle, _, control_labels_shuffle = control.shuffle_block_partial(trainer.train_sequence, block_size_shuffle, training_range[1])

        rs.atc_shfl.append(sequence_generator_temporal.sequence_autocor(control_labels_shuffle, n_labels=trainer.dataset.num_classes))

        control.train(trainer.network_shfl, trainer, control_data_shuffle, mem_sz=trainer.memory_size,
                                 batch_sz=trainer.batch_sz, lr=args.lr, momentum=0.5, training_range=training_range)
        rs.eval_shfl = trainer.evaluate_hierarchical()
        shuffle_accuracy_current = rs.eval_shfl[0][0]      # Recover the standard accur  acy
        shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (i+1)*args.test_stride]])
        rs.acc_shfl = np.append(rs.acc_shfl, shuffle_accuracy_current, axis=0)

        classes_correct = np.zeros(len(trainer.dataset.test_data))
        for k in range(nbr_test_samples):
            classes_correct[int(rs.eval_shfl[1][k][0])] +=1
        classes_correct = np.array([[classes_correct, (i+1)*args.test_stride]])
        rs.classes_pred_shfl = np.append(rs.classes_pred_shfl, classes_correct, axis=0)

        verbose('Accuracy of the (shuffle) network on the {0:d} test images: {1:.2f}%'.format(nbr_test_samples, shuffle_accuracy_current[0][0]), args)

    rs.train_data_shfl, rs.train_labels_shfl = control_data_shuffle, control_labels_shuffle

    rs.classes_count = [0 for k in range(len(trainer.dataset.train_data))]
    for k in rs.train_labels_orig:
        rs.classes_count[k] += 1

    return rs



def ultrametric_analysis_across_blocksizes(trainer, args, block_sizes):

    rs = ResultSet()

    trainer.network = trainer.network_orig
    rs.eval_orig = trainer.evaluate_hierarchical()

    rs.acc_orig = np.array([[rs.eval_orig[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, dim 1 of diagnos_original contains the accuracies at different levels
    nbr_test_samples = trainer.dataset.class_sz_test*(trainer.dataset.branching**trainer.dataset.depth)    # number of test examples
    classes_correct = np.zeros(len(trainer.dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
    # Compting the number of correct responses per classes before the training

    for k in range(nbr_test_samples):
        classes_correct[int(rs.eval_orig[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the k-th test example

    rs.classes_pred_orig = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

    verbose('Data generation...', args)
    trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training

    rs.train_data_orig, rates, rs.train_labels_orig = trainer.train_sequence

    verbose('...done\n', args)

    rs.atc_orig = sequence_generator_temporal.sequence_autocor(rs.train_labels_orig, n_labels=trainer.dataset.num_classes)


    for i in range(args.test_nbr):
        training_range = (i*args.test_stride, (i+1)*args.test_stride)     #Part of the sequence on which the training will be done
        verbose('Training network on original sequence...', args, 2)

        trainer.train(
            mem_sz = trainer.memory_size,
            lr = args.lr,
            momentum = 0.5,
            training_range = training_range
            )

        verbose('...done\nComputing performance for original sequence...', args, 2)

        rs.eval_orig = trainer.evaluate_hierarchical()

        verbose('...done\n', args, 2)

        original_accuracy_current = rs.eval_orig[0][0]      # Recover the standard accuracy
        original_accuracy_current = np.array([[original_accuracy_current, (i+1)*args.test_stride]])
        rs.acc_orig = np.append(rs.acc_orig, original_accuracy_current, axis=0)

        classes_correct = np.zeros(len(trainer.dataset.test_data))
        for k in range(nbr_test_samples):
            classes_correct[int(rs.eval_orig[1][k][0])] +=1
        classes_correct = np.array([[classes_correct, (i+1)*args.test_stride]])
        rs.classes_pred_orig = np.append(rs.classes_pred_orig, classes_correct, axis=0)

        verbose('Accuracy of the network on the {0:d} test images: {1:.2f}%'.format(nbr_test_samples, original_accuracy_current[0][0]), args)



    verbose("--- Start shuffle training ---", args)
    # Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the
    # sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)

    shfl_networks = {}
    rs.eval_shfl = {}
    rs.acc_shfl = {}
    rs.atc_shfl = {}
    rs.classes_pred_shfl = {}
    rs.train_data_shfl = {}
    rs.train_labels_shfl = {}

    for block_size_shuffle in block_sizes:
        shfl_networks.append(trainer.network_shfl)
        trainer.network = shfl_networks[-1]
        rs.eval_shfl[block_size_shuffle] = trainer.evaluate_hierarchical()
        rs.acc_shfl[block_size_shuffle] = np.array([[rs.eval_shfl[block_size_shuffle][0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_shuffle contains the accuracies at different levels
        classes_correct = np.zeros(len(trainer.dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
        # Compting the number of correct responses per classes before the training

        rs.atc_shfl[block_size_shuffle] = []

        for k in range(nbr_test_samples):
            classes_correct[int(rs.eval_shfl[block_size_shuffle][1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example
        rs.classes_pred_shfl[block_size_shuffle] = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

        # trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
        
        for i in range(args.test_nbr):
            training_range = (i*args.test_stride, (i+1)*args.test_stride)
            control_data_shuffle, _, control_labels_shuffle = control.shuffle_block_partial(trainer.train_sequence, block_size_shuffle, training_range[1])

            rs.atc_shfl[block_size_shuffle].append(sequence_generator_temporal.sequence_autocor(control_labels_shuffle, n_labels=trainer.dataset.num_classes))

            control.train(trainer.network_shfl, trainer, control_data_shuffle, mem_sz=trainer.memory_size,
                                     batch_sz=trainer.batch_sz, lr=args.lr, momentum=0.5, training_range=training_range)
            rs.eval_shfl[block_size_shuffle] = trainer.evaluate_hierarchical()
            shuffle_accuracy_current = rs.eval_shfl[block_size_shuffle][0][0]      # Recover the standard accur  acy
            shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (i+1)*args.test_stride]])
            rs.acc_shfl[block_size_shuffle] = np.append(rs.acc_shfl[block_size_shuffle], shuffle_accuracy_current, axis=0)

            classes_correct = np.zeros(len(trainer.dataset.test_data))
            for k in range(nbr_test_samples):
                classes_correct[int(rs.eval_shfl[block_size_shuffle][1][k][0])] +=1
            classes_correct = np.array([[classes_correct, (i+1)*args.test_stride]])
            rs.classes_pred_shfl[block_size_shuffle] = np.append(rs.classes_pred_shfl, classes_correct, axis=0)

            verbose('Accuracy of the (shuffle) network on the {0:d} test images: {1:.2f}%'.format(nbr_test_samples, shuffle_accuracy_current[0][0]), args)

        rs.train_data_shfl[block_size_shuffle], rs.train_labels_shfl[block_size_shuffle] = control_data_shuffle, control_labels_shuffle

        rs.classes_count = [0 for k in range(len(trainer.dataset.train_data))]
        for k in rs.train_labels_orig:
            rs.classes_count[k] += 1

        return rs
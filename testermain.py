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

import testing
import control
from trainer import train
import neuralnet
from evaluation import evaluate_hierarchical
import sequence_generator_temporal_noself as sequence_generator_temporal





diagnos_original = evaluate_hierarchical(netfc_original, trainer, device)

original_accuracy = np.array([[diagnos_original[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, dim 1 of diagnos_original contains the accuracies at different levels
nbr_test_samples = dataset.class_sz_test*(dataset.branching**dataset.depth)    # number of test examples
classes_correct = np.zeros(len(dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training

for k in range(nbr_test_samples):
    classes_correct[int(diagnos_original[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the k-th test example 

original_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

if args.verbose:
    print('Data generation...')
train_data_rates = trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
train_data, rates, train_labels = train_data_rates[0], train_data_rates[1], train_data_rates[2]

if args.verbose:
    print('...done\n')
# original_autocorr_function = sequence_generator_temporal.sequence_autocor(train_data)

original_autocorr_function = sequence_generator_temporal.sequence_autocor(train_labels, n_labels=dataset.num_classes)
for i in range(args.test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)     #Part of the sequence on which the training will be done
    if args.verbose > 1:
        print('Training network on original sequence...')
    train(netfc_original, trainer, train_data_rates, mem_sz=memory_sz, 
                     batch_sz=trainer.task_sz_nbr, lr=args.lr, momentum=0.5, training_range=training_range)
    if args.verbose > 1:
        print('...done\n')
        print('Computing performance for original sequence...')
    diagnos_original = evaluate_hierarchical(netfc_original, trainer, device)
    if args.verbose > 1:
        print('...done\n')
    original_accuracy_current = diagnos_original[0][0]      # Recover the standard accuracy
    original_accuracy_current = np.array([[original_accuracy_current, (i+1)*test_stride]])
    original_accuracy = np.append(original_accuracy, original_accuracy_current, axis=0)   
    
    classes_correct = np.zeros(len(dataset.test_data))   
    for k in range(nbr_test_samples):
        classes_correct[int(diagnos_original[1][k][0])] +=1
    classes_correct = np.array([[classes_correct, (i+1)*test_stride]])
    original_classes_prediction = np.append(original_classes_prediction, classes_correct, axis=0)
    
    print('Accuracy of the network on the %d test images: %.2f %%' % (nbr_test_samples, original_accuracy_current[0][0]))



print("--- Start shuffle training ---")
#netfc_shuffle = neuralnet.Net_CNN(dataset.data_origin)
#netfc_shuffle.to(device)



# Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the 
# sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)

diagnos_shuffle = evaluate_hierarchical(netfc_shuffle, trainer, device)
shuffle_accuracy = np.array([[diagnos_shuffle[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_shuffle contains the accuracies at different levels
classes_correct = np.zeros(len(dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training

shuffle_autocorr_functions = []

for k in range(nbr_test_samples):
    classes_correct[int(diagnos_shuffle[1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example 
shuffle_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training
    
train_data_rates = trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
train_data, rates, train_labels_sfl = train_data_rates[0], train_data_rates[1], train_data_rates[2]
for i in range(args.test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    control_data_shuffle, _, control_labels_shuffle = control.shuffle_block_partial(train_data_rates, block_size_shuffle, training_range[1])
    # shuffle_autocorr_functions.append(sequence_generator_temporal.sequence_autocor(control_data_shuffle))
    
    shuffle_autocorr_functions.append(sequence_generator_temporal.sequence_autocor(control_labels_shuffle, n_labels=dataset.num_classes))
    control.shuffle_sequence(netfc_shuffle, trainer, control_data_shuffle, mem_sz=memory_sz, 
                             batch_sz=trainer.task_sz_nbr, lr=args.lr, momentum=0.5, training_range=training_range)
    diagnos_shuffle = evaluate_hierarchical(netfc_shuffle, trainer, device)
    shuffle_accuracy_current = diagnos_shuffle[0][0]      # Recover the standard accur  acy
    shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (i+1)*test_stride]])
    shuffle_accuracy = np.append(shuffle_accuracy, shuffle_accuracy_current, axis=0)   
    
    classes_correct = np.zeros(len(dataset.test_data))   
    for k in range(nbr_test_samples):
        classes_correct[int(diagnos_shuffle[1][k][0])] +=1
    classes_correct = np.array([[classes_correct, (i+1)*test_stride]])
    shuffle_classes_prediction = np.append(shuffle_classes_prediction, classes_correct, axis=0)
    
    print('Accuracy of the (shuffle) network on the %d test images: %.2f %%' % (nbr_test_samples, shuffle_accuracy_current[0][0]))




compteur = [0 for k in range(len(dataset.train_data))]
for k in train_labels_sfl:
    compteur[k] += 1
    
#│exec(open("./sequence_plot_MNISTCIFAR.py").read())    
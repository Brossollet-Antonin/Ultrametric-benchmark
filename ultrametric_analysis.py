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
from trainer import Trainer
import sequencegenerator as sg
import sequenceshuffler as ss




trainer.network = netfc_original
eval_original = trainer.evaluate_hierarchical()

original_accuracy = np.array([[eval_original[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, dim 1 of eval_original contains the accuracies at different levels
nbr_test_samples = trainer.dataset.class_sz_test*(trainer.dataset.branching**trainer.dataset.depth)    # number of test examples
classes_correct = np.zeros(len(dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training

for k in range(nbr_test_samples):
    classes_correct[int(eval_original[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the k-th test example 

original_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training

verbose('Data generation...', args)
trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training

train_data, rates, train_labels = trainer.train_sequence

verbose('...done\n', args)

original_autocorr_function = sg.sequence_autocor(train_labels, n_labels=trainer.dataset.num_classes)


for i in range(args.test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)     #Part of the sequence on which the training will be done
    verbose('Training network on original sequence...', args, 2)

    trainer.train(
        mem_sz = memory_sz,
        batch_sz = trainer.task_sz_nbr,
        lr = args.lr,
        momentum = 0.5,
        training_range = training_range
        )

    verbose('...done\nComputing performance for original sequence...', args, 2)

    eval_original = trainer.evaluate_hierarchical()

    verbose('...done\n', args, 2)

    original_accuracy_current = eval_original[0][0]      # Recover the standard accuracy
    original_accuracy_current = np.array([[original_accuracy_current, (i+1)*test_stride]])
    original_accuracy = np.append(original_accuracy, original_accuracy_current, axis=0)   
    
    classes_correct = np.zeros(len(trainer.dataset.test_data))   
    for k in range(nbr_test_samples):
        classes_correct[int(eval_original[1][k][0])] +=1
    classes_correct = np.array([[classes_correct, (i+1)*test_stride]])
    original_classes_prediction = np.append(original_classes_prediction, classes_correct, axis=0)
    
    print('Accuracy of the network on the %d test images: %.2f %%' % (nbr_test_samples, original_accuracy_current[0][0]))



print("--- Start shuffle training ---")
# Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the 
# sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)

trainer.network = netfc_shuffle
eval_shuffle = trainer.evaluate_hierarchical()
shuffle_accuracy = np.array([[eval_shuffle[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of eval_shuffle contains the accuracies at different levels
classes_correct = np.zeros(len(trainer.dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training

shuffle_autocorr_functions = []

for k in range(nbr_test_samples):
    classes_correct[int(eval_shuffle[1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example 
shuffle_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training
    
trainer.make_train_sequence()  #Stock rates (if not a random process) and data for training
train_data, rates, train_labels_sfl = trainer.train_sequence

seqshfl = ss.SequenceShuffler()
for i in range(args.test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    control_data_shuffle, _, control_labels_shuffle = seqshfl.shuffle_block_partial(trainer.train_sequence, block_size_shuffle, training_range[1])
    
    shuffle_autocorr_functions.append(sg.sequence_autocor(control_labels_shuffle, n_labels=trainer.dataset.num_classes))
    seqshfl.shuffle_sequence(netfc_shuffle, trainer, control_data_shuffle, mem_sz=memory_sz, 
                             batch_sz=trainer.task_sz_nbr, lr=args.lr, momentum=0.5, training_range=training_range)
    eval_shuffle = trainer.evaluate_hierarchical()
    shuffle_accuracy_current = eval_shuffle[0][0]      # Recover the standard accur  acy
    shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (i+1)*test_stride]])
    shuffle_accuracy = np.append(shuffle_accuracy, shuffle_accuracy_current, axis=0)   
    
    classes_correct = np.zeros(len(trainer.dataset.test_data))   
    for k in range(nbr_test_samples):
        classes_correct[int(eval_shuffle[1][k][0])] +=1
    classes_correct = np.array([[classes_correct, (i+1)*test_stride]])
    shuffle_classes_prediction = np.append(shuffle_classes_prediction, classes_correct, axis=0)
    
    print('Accuracy of the (shuffle) network on the %d test images: %.2f %%' % (nbr_test_samples, shuffle_accuracy_current[0][0]))




compteur = [0 for k in range(len(trainer.dataset.train_data))]
for k in train_labels_sfl:
    compteur[k] += 1
    
#│exec(open("./sequence_plot_MNISTCIFAR.py").read())    
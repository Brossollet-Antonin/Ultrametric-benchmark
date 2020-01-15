# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:37:35 2019

@author: Antonin
"""


import algo 
import torch
import neuralnet
import numpy as np
import testing
import control
import random
from copy import deepcopy
import diagnosis
import sequence_generator_temporal_noself as sequence_generator_temporal


device = torch.device("cpu")
print(device)
#dataset = artificial_dataset.artificial_dataset(depth=depth, branching=data_branching, data_sz=200, class_sz_train=20000, 
#                                                class_sz_test=1000, ratio_type='exponnential', ratio_value=2)
#dataset.create()
trainer = algo.training('temporal correlation', 'reservoir sampling', dataset=dataset,
                        task_sz_nbr=minibatches,      
                        tree_depth=depth, preprocessing=False, device=device, sequence_length=sequence_length, energy_step=step, T=T, 
                        tree_branching=tree_branching, proba_transition=proba_transition)



 
netfc_original = neuralnet.Net_CNN(dataset.data_origin)
netfc_original.to(device)


diagnos_original = diagnosis.hierarchical_error(netfc_original, trainer, device)
original_accuracy = np.array([[diagnos_original[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_original contains the accuracies at different levels
nbr_test_samples = dataset.class_sz_test*(tree_branching**depth)    # number of test examples
classes_correct = np.zeros(len(dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training
for k in range(nbr_test_samples):
    classes_correct[int(diagnos_original[1][k][0])] += 1     # The value in the array correspond to the prediction of the network for the i-th test example 
original_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training
    
train_data_rates=trainer.train_data()  #Stock rates (if not a random process) and data for training
train_data, rates, train_sequence = train_data_rates[0], train_data_rates[1], train_data_rates[2]
for i in range(test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    algo.learning_ER(netfc_original, trainer, train_data_rates, mem_sz=memory_sz, 
                     batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
    diagnos_original = diagnosis.hierarchical_error(netfc_original, trainer, device)
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
netfc_shuffle = neuralnet.Net_CNN(dataset.data_origin)
netfc_shuffle.to(device)

# Shuffle the training sequence in block of a choosen length (try to use a length of blocks that divise the length of the 
# sequence to be sure to train on the full sequence, have one small block to take that into account is not implemented # TODO)
control_data_shuffle = control.shuffle_block(train_data, block_size_shuffle)

diagnos_shuffle = diagnosis.hierarchical_error(netfc_shuffle, trainer, device)
shuffle_accuracy = np.array([[diagnos_shuffle[0][0], 0]])     # Will contain the accuracy through training and the number of train samples seen, the first dim of diagnos_shuffle contains the accuracies at different levels
nbr_test_samples = dataset.class_sz_test*(tree_branching**depth)    # number of test examples
classes_correct = np.zeros(len(dataset.test_data))     # Array of size the number of classes to stock the current count of prediction
# Compting the number of correct responses per classes before the training
for k in range(nbr_test_samples):
    classes_correct[int(diagnos_shuffle[1][k][0])] +=1     # The value in the array correspond to the prediction of the network for the i-th test example 
shuffle_classes_prediction = np.array([[classes_correct, 0]])   # This array will stock the prediction of the network during the training
    
train_data_rates=trainer.train_data()  #Stock rates (if not a random process) and data for training
train_data, rates, train_sequence = train_data_rates[0], train_data_rates[1], train_data_rates[2]
for i in range(test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    control.shuffle_sequence(netfc_shuffle, trainer, control_data_shuffle, mem_sz=memory_sz, 
                             batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
    diagnos_shuffle = diagnosis.hierarchical_error(netfc_shuffle, trainer, device)
    shuffle_accuracy_current = diagnos_shuffle[0][0]      # Recover the standard accuracy
    shuffle_accuracy_current = np.array([[shuffle_accuracy_current, (i+1)*test_stride]])
    shuffle_accuracy = np.append(shuffle_accuracy, shuffle_accuracy_current, axis=0)   
    
    classes_correct = np.zeros(len(dataset.test_data))   
    for k in range(nbr_test_samples):
        classes_correct[int(diagnos_shuffle[1][k][0])] +=1
    classes_correct = np.array([[classes_correct, (i+1)*test_stride]])
    shuffle_classes_prediction = np.append(shuffle_classes_prediction, classes_correct, axis=0)
    
    print('Accuracy of the (shuffle) network on the %d test images: %.2f %%' % (nbr_test_samples, shuffle_accuracy_current[0][0]))





#shuffle_accuracy = np.array([[testing.testing_final(netfc_shuffle, dataset, device)[0], 0]])
##control_data_shuffle = deepcopy(train_data)
##random.shuffle(control_data_shuffle)
#control_data_shuffle = control.shuffle_block(train_data, 10000)
#for i in range(test_nbr):
#    training_range = (i*test_stride, (i+1)*test_stride)
#    control.shuffle_sequence(netfc_shuffle, trainer, control_data_shuffle, mem_sz=memory_sz, 
#                             batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
#    shuffle_accuracy_current, _ = testing.testing_final(netfc_shuffle, dataset, device)
#    shuffle_accuracy_current = np.array([[shuffle_accuracy_current, i*test_stride]])
#    shuffle_accuracy = np.append(shuffle_accuracy, shuffle_accuracy_current, axis=0)

#
#print("--- Start labels training ---")
#netfc_labels = neuralnet.Net_CNN(dataset.data_origin)
#netfc_labels.to(device)
#
#labels_accuracy = [testing.testing_final(netfc_labels, dataset, device)[0]]
#permutation = [i for i in range(max(train_sequence)+1)]
#random.shuffle(permutation)
#control_sequence = [permutation[i] for i in train_sequence]
#control_data_labels = sequence_generator_temporal.training_sequence(control_sequence, trainer.dataset)
#for i in range(test_nbr):
#    training_range = (i*test_stride, (i+1)*test_stride)
#    control.shuffle_labels(netfc_labels, trainer, control_data_labels, mem_sz=memory_sz, 
#                           batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
#    labels_accuracy_current, _ = testing.testing_final(netfc_labels, dataset, device)
#    labels_accuracy.append(labels_accuracy_current)
#
    
#netfc_labels2 = neuralnet.Net_FCL()
#netfc_labels2.to(device)
#control_sequence_labels2, labels_accuracy2 = control.shuffle_labels(netfc_labels2, trainer, train_sequence, mem_sz=memory_sz, 
#                                                                    batch_sz=10, lr=0.01, momentum=0.5)




compteur = [0 for k in range(len(dataset.train_data))]
for k in train_sequence:
    compteur[k] += 1
    
#│exec(open("./sequence_plot_MNISTCIFAR.py").read())    
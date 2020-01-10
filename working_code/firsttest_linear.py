# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:35:15 2019

@author: Antonin
"""


import artificial_dataset
import algo 
import torch
import neuralnet
import testing
import control
import random
from copy import deepcopy
import sequence_generator_temporal_noself as sequence_generator_temporal


device = torch.device("cpu")
print(device)
#dataset = artificial_dataset.artificial_dataset(depth=depth, branching=data_branching, data_sz=200, class_sz_train=20000, 
#                                                class_sz_test=1000, ratio_type='exponnential', ratio_value=2)
#dataset.create()
trainer = algo.training('temporal correlation', 'reservoir sampling', dataset=dataset,
                        task_sz_nbr=10,      
                        tree_depth=depth, preprocessing=False, device=device, sequence_length=sequence_length, energy_step=step, T=T, 
                        tree_branching=tree_branching, proba_transition=proba_transition)



 
netfc_original = neuralnet.Net_FCRU()
netfc_original.to(device)

original_accuracy = [testing.testing_final(netfc_original, dataset, device)[0]]
train_data_rates=trainer.train_data()  #Stock rates (if not a random process) and data for training
train_data, rates, train_sequence = train_data_rates[0], train_data_rates[1], train_data_rates[2]
for i in range(test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    algo.learning_ER(netfc_original, trainer, train_data_rates, mem_sz=memory_sz, 
                     batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
    original_accuracy_current, test_data = testing.testing_final(netfc_original, dataset, device)
    original_accuracy.append(original_accuracy_current)    


print("--- Start shuffle training ---")
netfc_shuffle = neuralnet.Net_FCRU()
netfc_shuffle.to(device)

shuffle_accuracy = [testing.testing_final(netfc_shuffle, dataset, device)[0]]
control_data_shuffle = deepcopy(train_data)
random.shuffle(control_data_shuffle)
for i in range(test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    control.shuffle_sequence(netfc_shuffle, trainer, control_data_shuffle, mem_sz=memory_sz, 
                             batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
    shuffle_accuracy_current, _ = testing.testing_final(netfc_shuffle, dataset, device)
    shuffle_accuracy.append(shuffle_accuracy_current)


print("--- Start labels training ---")
netfc_labels = neuralnet.Net_FCRU()
netfc_labels.to(device)

labels_accuracy = [testing.testing_final(netfc_labels, dataset, device)[0]]
permutation = [i for i in range(max(train_sequence)+1)]
random.shuffle(permutation)
control_sequence = [permutation[i] for i in train_sequence]
control_data_labels = sequence_generator_temporal.training_sequence(control_sequence, trainer.dataset)
for i in range(test_nbr):
    training_range = (i*test_stride, (i+1)*test_stride)
    control.shuffle_labels(netfc_labels, trainer, control_data_labels, mem_sz=memory_sz, 
                           batch_sz=10, lr=0.01, momentum=0.5, training_range=training_range)
    labels_accuracy_current, _ = testing.testing_final(netfc_labels, dataset, device)
    labels_accuracy.append(labels_accuracy_current)
#
    
#netfc_labels2 = neuralnet.Net_FCL()
#netfc_labels2.to(device)
#control_sequence_labels2, labels_accuracy2 = control.shuffle_labels(netfc_labels2, trainer, train_sequence, mem_sz=memory_sz, 
#                                                                    batch_sz=10, lr=0.01, momentum=0.5)




compteur = [0 for k in range(len(dataset.train_data))]
for k in train_sequence:
    compteur[k] += 1
    
exec(open("./sequence_plot.py").read())    
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:37:08 2019

@author: Antonin
"""

import artificial_dataset
import algo 
import torch
import neuralnet
import testing
import control


device = torch.device("cpu")
print(device)
#dataset = artificial_dataset.artificial_dataset(depth=depth, branching=data_branching, data_sz=200, class_sz_train=20000, 
#                                                class_sz_test=1000, ratio_type='exponnential', ratio_value=2)
#dataset.create()
trainer = algo.training('temporal correlation', 'reservoir sampling', dataset=dataset,
                        task_sz_nbr=10,      
                        tree_depth=depth, preprocessing=False, device=device, sequence_length=sequence_length, T=T, tree_branching=tree_branching)




netfc_original = neuralnet.Net_FCRU()
netfc_original.to(device)

train_data = []
test_data = []
train_sequence = []
original_accuracy = []
for i in range(test_nbr):
    test_stride = int(sequence_length/test_nbr)
    training_range = (i*test_stride, (i+1)*test_stride)
    train_data_current, rates, train_sequence_current = algo.learning_ER(netfc_original, trainer, mem_sz=memory_sz, 
                                                     batch_sz=10, lr=0.01, momentum=0.5, training_range)
    original_accuracy_current, test_data_current = testing.testing_final(netfc_original, dataset, device)
    train_data = train_data + train_data_current
    train_sequence = train_sequence + train_sequence_current
    test_data = test_data + test_data_current
    original_accuracy = original_accuracy + original_accuracy_current    


netfc_shuffle = neuralnet.Net_FCRU()
netfc_shuffle.to(device)
control_sequence_shuffle, shuffle_accuracy = control.shuffle_sequence(netfc_shuffle, trainer, train_data, mem_sz=memory_sz, 
                                                                      batch_sz=10, lr=0.01, momentum=0.5)

netfc_labels1 = neuralnet.Net_FCRU()
netfc_labels1.to(device)
control_sequence_labels1, labels_accuracy1 = control.shuffle_labels(netfc_labels1, trainer, train_sequence, mem_sz=memory_sz, 
                                                                    batch_sz=10, lr=0.01, momentum=0.5)
#
#netfc_labels2 = neuralnet.Net_FCRU()
#netfc_labels2.to(device)
#control_sequence_labels2, labels_accuracy2 = control.shuffle_labels(netfc_labels2, trainer, train_sequence, mem_sz=memory_sz, 
#                                                                    batch_sz=10, lr=0.01, momentum=0.5)




compteur = [0 for k in range(len(dataset.train_data))]
for k in train_sequence:
    compteur[k] += 1
    
exec(open("./sequence_plot.py").read())    
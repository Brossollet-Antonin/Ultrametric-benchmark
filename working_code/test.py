# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:35:54 2019

@author: Antonin
"""

import algo
import neuralnet
import testing
import torch
import torchvision
import control

import time
start_time = time.time()

memory_sz = 0
epoch = 2
step = 0
minibatches = 10

device = torch.device("cpu")
print(device)
 
trainer = algo.training('spatial correlation', 'reservoir sampling', task_sz_nbr=10, T=1/13, train_epoch=2, energy_step=3, device=device)

netfc = neuralnet.Net_FCRU()
netfc.to(device)

train_data, rates, train_sequence = algo.learning_ER(netfc, trainer, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)


test_data = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)
original_accuracy = testing.testing_final(netfc, test_data, device, (8,9) if trainer.training_type=='temporal correlation' else ())

#netfc = neuralnet.Net_FCRU()
#train_data, rates, train_sequence = algo.learning_naive(netfc, trainer, (train_data, rates, train_sequence), batch_sz=64, lr=0.01, momentum=0.5)
#naive_accuracy = testing.testing_final(netfc, test_data, device, (8,9) if trainer.training_type=='temporal correlation' else ())


netfc = neuralnet.Net_FCRU()
netfc.to(device)
control_sequence_shuffle, shuffle_accuracy = control.shuffle_sequence(netfc, trainer, train_data, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)

netfc = neuralnet.Net_FCRU()
netfc.to(device)
control_sequence_labels, labels_accuracy = control.shuffle_labels(netfc, trainer, train_data, train_sequence, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)


#
#netfc = neuralnet.Net_FCRU()
#netfc.to(device)
#control_sequence_labels2, labels_accuracy2 = control.shuffle_labels(netfc, trainer, train_data, train_sequence, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)
#
#
#
#netfc = neuralnet.Net_FCRU()
#netfc.to(device)
#control_sequence_labels3, labels_accuracy3 = control.shuffle_labels(netfc, trainer, train_data, train_sequence, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)
#
#
#
#netfc = neuralnet.Net_FCRU()
#netfc.to(device)
#control_sequence_labels4, labels_accuracy4 = control.shuffle_labels(netfc, trainer, train_data, train_sequence, mem_sz=memory_sz, batch_sz=10, lr=0.01, momentum=0.5)
#


print("--- %s seconds ---" % (time.time() - start_time))


# plot the sequences
exec(open("./sequence_plot.py").read()) 



#import sequence_generator_spatial
#cor_original = sequence_generator_spatial.sequence_autocor(train_sequence)
#cor_labels = sequence_generator_spatial.sequence_autorco(control_sequence_labels)
#plt.figure()
#plt.loglog(cor_original[:3000])
#plt.title("Autocorrelation original sequence")
#plt.figure()
#plt.plot(cor_labels)
#plt.title("Autocorrelation shuffled labels sequence") 


#
#device = torch.device("cpu")
#print(device)
#
#trainer = algo.training('spatial correlation', 'reservoir sampling', train_epoch=3, task_sz_nbr=64, T=1/13, energy_step=3, device=device)
#
#netfc = neuralnet.Net_CNN()
#netfc.to(device)
#
#train_data, rates, train_sequence = algo.learning_ER(netfc, trainer, mem_sz=memory_sz, batch_sz=64, lr=0.01, momentum=0.5)
#test_data = torch.utils.data.DataLoader(
#  torchvision.datasets.MNIST('/files/', train=False, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#  batch_size=1, shuffle=True)
#original_accuracy = testing.testing_final(netfc, test_data, device, (8,9) if trainer.training_type=='temporal correlation' else ())
#netfc = neuralnet.Net_CNN()
#netfc.to(device)
#control_sequence_shuffle, shuffle_accuracy = control.shuffle_sequence(netfc, trainer, train_data, mem_sz=memory_sz, batch_sz=64, lr=0.01, momentum=0.5)
#
#netfc = neuralnet.Net_CNN()
#netfc.to(device)
#control_sequence_labels, labels_accuracy = control.shuffle_labels(netfc, trainer, train_data, train_sequence, mem_sz=memory_sz, batch_sz=64, lr=0.01, momentum=0.5)
#
#
#print("--- %s seconds ---" % (time.time() - start_time))
#
#
## plot the sequences
#exec(open("./sequence_plot.py").read()) 
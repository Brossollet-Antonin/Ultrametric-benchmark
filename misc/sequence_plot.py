# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:13:15 2019

@author: Antonin
"""

import matplotlib.pyplot as plt
import numpy as np

seq=[]
for k in train_data:
    seq.append(k[1].item())
   
plt.figure(figsize=(9,5))
plt.plot(train_sequence, label='Original')
plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d \n Energy step: %d Temperature: %.3f Branching: %d" % (original_accuracy[-1], memory_sz, minibatches, epoch, step, T, tree_branching))
plt.legend()
plt.savefig(savepath + save_folder + "/original.png")

#plt.figure(figsize=(9,5))
#x = range(test_nbr)
#plt.plot(x, original_accuracy)
#plt.title("Accuracy evolution during training")
#plt.savefig(savepath+"T%.3f Depth%d Memory%d Branching%d/original_accuracy_evolution.png" % (T, depth, memory_sz, tree_branching))



seq_control_shuffle=[]
for k in control_data_shuffle:
    seq_control_shuffle.append(k[1].item())
    
plt.figure(figsize=(9,5))   
plt.plot(seq_control_shuffle, label='Control Shuffle')
plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d \n Energy step: %d Temperature: %.3f Branching : %d" % (shuffle_accuracy[-1], memory_sz, minibatches, epoch, step, T, tree_branching))
plt.legend()
plt.savefig(savepath + save_folder + "/shuffle.png")

#plt.figure(figsize=(9,5))
#x = range(test_nbr)
#plt.plot(x, shuffle_accuracy)
#plt.title("Accuracy evolution during training")
#plt.savefig(savepath+"T%.3f Depth%d Memory%d Branching%d/shuffle_accuracy_evolution.png" % (T, depth, memory_sz, tree_branching))



seq_control_labels=[]
for k in control_data_labels:
    seq_control_labels.append(k[1].item())
    
plt.figure(figsize=(9,5))
plt.plot(seq_control_labels, label='Control Labels')
plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d \n Energy step: %d Temperature: %.3f Branching: %d" % (labels_accuracy[-1], memory_sz, minibatches, epoch, step, T, tree_branching))
plt.legend()
plt.savefig(savepath + save_folder + "/labels.png")

plt.figure(figsize=(9,5))
x = np.linspace(0, sequence_length, test_nbr+1)
plt.ylim(0, 105)
plt.plot(x, original_accuracy, label='Original')
plt.plot(x, shuffle_accuracy, label='Shuffle')
plt.plot(x, labels_accuracy, label='Labels')
plt.title("Accuracy evolution during training")
plt.xlabel('Number of train examples')
plt.ylabel('Accuracy in %')
plt.legend()
plt.savefig(savepath + save_folder + "/accuracy.png")

plt.figure(figsize=(18,10))
x = range(len(compteur))
plt.bar(x, compteur)
plt.title("Classes distribution \n Memory size: %d Minibatches size: %d Epoch: %d \n Energy step: %d Temperature: %.3f Branching: %d" %(memory_sz, minibatches, epoch, step, T, tree_branching))
plt.savefig(savepath + save_folder + "/distribution.png")


#
#
#seq_control_labels2=[]
#for k in control_sequence_labels2:
#    seq_control_labels2.append(k[1].item())
#    
#plt.figure()
#plt.plot(seq_control_labels2, label='Control Labels 2')
#plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d Energy step: %d" % (labels_accuracy2, memory_sz, minibatches, epoch, step))
#plt.legend()
#
#
#seq_control_labels3=[]
#for k in control_sequence_labels3:
#    seq_control_labels3.append(k[1].item())
#    
#plt.figure()
#plt.plot(seq_control_labels3, label='Control Labels 3')
#plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d Energy step: %d" % (labels_accuracy3, memory_sz, minibatches, epoch, step))
#plt.legend()
#
#
#seq_control_labels4=[]
#for k in control_sequence_labels4:
#    seq_control_labels4.append(k[1].item())
#    
#plt.figure()
#plt.plot(seq_control_labels4, label='Control Labels 4')
#plt.title("Accuracy : %.2f %% \n Memory size: %d Minibatches size: %d Epoch: %d Energy step: %d" % (labels_accuracy4, memory_sz, minibatches, epoch, step))
#plt.legend()






#    
#    
#plt.plot(train_sequence, label='Original')
#plt.legend()
#
#
#plt.figure()   
#plt.plot(control_sequence_shuffle, label='Control Shuffle')
#plt.legend()
#
#plt.figure()
#plt.plot(control_sequence_labels, label='Control Labels')
#plt.legend
#
#
#compteur1 = 10*[0]
#for k in train_sequence:
#    compteur1[k]+=1
#    
#compteur2 = 10*[0]
#for k in control_sequence_shuffle:
#    compteur2[k]+=1
#
#compteur3 = 10*[0]
#for k in control_sequence_labels:
#    compteur3[k]+=1
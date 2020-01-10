# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:46:11 2019

@author: Antonin
"""



import os
import hidden_layers as hidden
import diagnosis
import numpy as np
#import matplotlib.pyplot as plt
import artificial_dataset




#Temperature = np.arange(2.9, 0, -0.3)
Temperature=[1, 1/2, 1/3, 1/5, 1/4, 1/6]

# Warning: must verify that depth_seq < depth_data for all the cases 
depth_tree = [3]
nbr_run = 10
memory_sz = 0  
epoch = 0
minibatches_list = [10, 5, 50, 100, 1000]
step = 1
tree_branching_list = [2]
proba_transition = 1e-4
sequence_length = 1000000    
test_nbr = 50
test_stride = int(sequence_length/test_nbr)
block_size_shuffle_list = [1, 10, 50, 100, 1000, 10000]



uniqueID = 0
for run in range(nbr_run):
    for minibatches in minibatches_list:
for depth in depth_tree: 
    for tree_branching in tree_branching_list:
        data_branching = tree_branching
        dataset = artificial_dataset.artificial_dataset(data_origin='CIFAR10')
        for T in Temperature: 
            savepath = "./Results/CIFAR10_testing/CNN_length%d_minibatches%d/" % (sequence_length, minibatches)
            data_branching = tree_branching
            save_folder = "T%.3f Depth%d Memory%d Branching%d %d" % (T, depth, memory_sz, tree_branching, uniqueID)
            os.makedirs(savepath + save_folder)
    
            parameters = np.array([[T, depth, tree_branching, sequence_length, minibatches, block_size_shuffle, test_nbr, step, memory_sz],
                                   ["Temperature", "Tree Depth", "Tree Branching", "Sequence Length", "Minibatches Size", 
                                    "Size Blocks Shuffle", "Number of tests", "Energy Step", "Replay Memory Size"]])
        
            exec(open("./testerMNIST.py").read()) 
            
            
            diagnos_original = diagnosis.hierarchical_error(netfc_original, trainer, device)
            diagnos_shuffle = diagnosis.hierarchical_error(netfc_shuffle, trainer, device)
            #diagnos_labels = diagnosis.hierarchical_error(netfc_labels, trainer, device)
            
#            np.savetxt(savepath + save_folder + "/accuracy_original.txt", diagnos_original[0])
#            np.savetxt(savepath + save_folder + "/accuracy_shuffle.txt", diagnos_shuffle[0])
           # np.savetxt(savepath + save_folder + "/accuracy_labels.txt", diagnos_labels[0])

#            x = np.arange((tree_branching**depth)*dataset.class_sz_test)
#            ymax = tree_branching**depth
#            plt.figure(figsize=(18,10)) 
#            plt.plot(diagnos_original[1][:, 2], '.b')
#            plt.plot(diagnos_original[1][:, 3], '.r')
#            plt.plot(diagnos_original[1][:, 4], '.k')
#            plt.legend(('0', '1', '2'))
#            plt.axis(ymin=-2, ymax=ymax)
#            plt.title("Pred GT diff original ")
#            plt.savefig(savepath + save_folder + "/diff_original.png")
#            
#            plt.figure(figsize=(18,10)) 
#            plt.plot(diagnos_shuffle[1][:, 2], '.b')
#            plt.plot(diagnos_shuffle[1][:, 3], '.r')
#            plt.plot(diagnos_shuffle[1][:, 4], '.k')
#            plt.legend(('0', '1', '2'))
#            plt.title("Pred GT diff shuffle")
#            plt.axis(ymin=-2, ymax=ymax)
#            plt.savefig(savepath + save_folder + "/diff_shuffle.png")
            
#            plt.figure(figsize=(18,10))
#            plt.plot(diagnos_labels[1][:, 2], '.b')
#            plt.plot(diagnos_labels[1][:, 3], '.r')
#            plt.plot(diagnos_labels[1][:, 4], '.k')
#            plt.legend(('0', '1', '2'))
#            plt.title("Pred GT diff labels")
#            plt.axis(ymin=-2, ymax=ymax)    
#            plt.savefig(savepath + save_folder + "/diff_labels.png")
            
                
#            L = len(dataset.test_data)
#            compteur_diag_original=np.zeros(L)
#            compteur_diag_shuffle=np.zeros(L)
##            compteur_diag_labels=np.zeros(L)
#            nbr_test = dataset.class_sz_test*(tree_branching**depth)
#            for i in range(nbr_test):
#                compteur_diag_original[int(diagnos_original[1][i][0])] += 1
#                compteur_diag_shuffle[int(diagnos_shuffle[1][i][0])] += 1
##                compteur_diag_labels[int(diagnos_labels[1][i][0])] += 1
#            y = np.arange(L)
#            
#            plt.figure(figsize=(18,10))
#            plt.bar(y, compteur_diag_original)
#            plt.title("Predicted classes repartition Original")
#            plt.savefig(savepath + save_folder + "/pred_distrib_original.png")
#            
#            plt.figure(figsize=(18,10))
#            plt.bar(y, compteur_diag_shuffle)
#            plt.title("Predicted classes repartition Shuffle")
#            plt.savefig(savepath + save_folder + "/pred_distrib_shuffle.png")
            
#            plt.figure(figsize=(18,10))
#            plt.bar(y, compteur_diag_labels)
#            plt.title("Predicted classes repartition Labels")
#            plt.savefig(savepath + save_folder + "/pred_distrib_labels.png")
            #Saving important data
            exec(open("./save_data.py").read())
            
#            if len(Temperature)*len(depth_tree)*len(tree_branching_list) > 1:
#                plt.close('all')
            uniqueID += 1 
            

 
#ploting_GT = False
#ploting_pred = False
#
#original_GT = hidden.plot_hidden_GT(netfc_original, dataset, device, 'Original', ploting_GT)
#shuffle_GT = hidden.plot_hidden_GT(netfc_shuffle, dataset, device, 'Shuffle', ploting_GT)
#labels_GT = hidden.plot_hidden_GT(netfc_labels1, dataset, device, 'Labels', ploting_GT)
#      
#
#
#shuffle_pred = hidden.plot_hidden_pred(netfc_shuffle, dataset, device, 'Shuffle', ploting_pred)
#original_pred = hidden.plot_hidden_pred(netfc_original, dataset, device, 'Original', ploting_pred)
#labels_pred = hidden.plot_hidden_pred(netfc_labels1, dataset, device, 'Labels', ploting_pred)

#diagnos_original = diagnosis.test_error(netfc_original, dataset, device)
#diagnos_shuffle = diagnosis.test_error(netfc_shuffle, dataset, device)
#diagnos_labels = diagnosis.test_error(netfc_labels1, dataset, device)
    
    
    

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:24:07 2020

@author: Simon
"""


import pdb
import itertools
from copy import deepcopy

import random
from random import shuffle
from random import seed

from operator import add
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import memory
import sequence_generator_temporal
import sequence_generator_spatial
import rates_correlation
import preprocessing
from neuralnet import ContinualLearner

from utils import verbose


def mem_SGD(net, mini_batch, lr, momentum, device, optimizer=None):
    # Instanciates optimizer and computes the loss for mini_batch
    # Backpropagates
    # Returns avg loss on mini-batch
    inputs, labels = mini_batch
    inputs, labels = inputs.to(device), labels.to(device=device)
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
    outputs = net(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


class ClassGraph:
    """
    A general structure that will a graph between labels so as to allow generation of a learning sequence by random walk on this graph
    """
    def __init__(self, n_classes, transition_matrix):
        self.n_classes = n_classes
        self.transition_matrix = transition_matrix

class UltrametricTree (ClassGraph):
    """
    A particular case of ClassGraph where the graph is a tree, where an ultrametric topology is well-defined.
    """

    def __init__(self):
        super.__init__()
        


class Trainer:
    """
    Defined by tuple (dataset, neural_network, sequence_type).
    Training type : random, ultrametric, uniform, spatial_correlation,
    ladder_blocks1, ladder_blocks2, random_blocks1, random_blocks2, random_blocks2_2freq
    memory sampling : reservoir sampling, ring buffer

    random: random sequence
    ultrametric: random walk on ultrametric tree
    uniform: uniform transition matrix
    spatial_correlation: unmaintened
    ladder_blocks1: sequence of following digits, repeted split_length_list[0]
    times (0000000...01111...122222...)
    ladder_blocks2: like ladder_blocks1 but two adjacent digits in each blocks,
    with equiprobability between each two digits (01100101100...10222332333...)
    random_blocks1: blocks of the same random digit repeted split_length_list[0]
    times (33333333....3555555...5666666666)
    random_blocks2: blocks of two adjacent random digits repeted
    split_length_list[0] with equiprobability to switch between two digits
    times (334334433433...345655566...69898899898...)
    random_blocks2_2freq: like random_blocks2 but with two different timescales
    of switches, lenghts stored in split_length_list
    """
    def __init__(self, dataset, network, training_type, memory_sampling, memory_sz, lr, momentum, criterion=None, optimizer=None, batch_sz=None,
                 sequence_first=0, sequence_length=60000, min_visit=0, energy_step=1, T=0.4,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 preprocessing=True, proba_transition=1e-3, dynamic_T_thr=0, split_length_list=None):

        self.dataset = dataset
        self.network_tmpl = network
        self.network = deepcopy(self.network_tmpl)
        self.training_type = training_type

        self.batch_sz = batch_sz
        self.memory_sampling = memory_sampling
        self.memory_size = memory_sz

        self.lr = lr
        self.momentum = momentum

        self.rates_vector = np.array([])

        if optimizer is None:
            optimizer = 'sgd'
        self.optimizer_type = optimizer

        if criterion is None:
            criterion = 'cross_entropy'
        if criterion.lower()=='l1':
            self.criterion = nn.L1Loss()
        if criterion.lower()=='l2' or criterion.lower()=='mse':
            self.criterion = nn.MSELoss()
        if criterion.lower()=='nll':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.sequence_first = sequence_first
        self.sequence_length = sequence_length
        self.energy_step = energy_step
        self.T = T
        self.tree_depth = dataset.depth
        self.device = device
        self.min_visit = min_visit
        self.preprocessing = preprocessing
        self.tree_branching = dataset.branching
        self.proba_transition = proba_transition
        self.dynamic_T_thr = dynamic_T_thr
        self.n_classes = self.tree_branching**self.tree_depth

        if self.training_type=="ladder_blocks1" or self.training_type=="ladder_blocks2":
            self.split_block_lengths = split_length_list
        if self.training_type=="random_blocks1" or self.training_type=="random_blocks2" or self.training_type=="random_blocks2_2freq":
            self.split_block_lengths = split_length_list

    def assign_model(self, model):
        self.network = model
        
        if self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        elif self.optimizer_type == 'adagrad':
            self.optimizer = optim.Adagrad(self.network.parameters(), lr=self.lr)


    def mem_optim(self, mini_batch):
        # Instanciates optimizer and computes the loss for mini_batch
        # Backpropagates
        # Returns avg loss on mini-batch
        inputs, labels = mini_batch
        inputs, labels = inputs.to(self.device), labels.to(device=self.device)
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        ### Additional losses for method-specific continual learning:
        # Add EWC-loss
        if self.network.ewc_lambda>0:
            ewc_loss = self.network.ewc_loss()
            loss += self.network.ewc_lambda * ewc_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def heuristic_temperature(self, rate_law, dT=0.01):
        # Only relevant when self.training_type = 'ultrametric'
        T = 0.01
        rates = sequence_generator_temporal.setting_rates(self.energy_step, T, self.tree_depth, self.tree_branching, rate_law, force_switch=False)
        threshold = 10/self.sequence_length
        while min(rates) < threshold:
            T += dT
            rates = sequence_generator_temporal.setting_rates(self.energy_step, T, self.tree_depth, self.tree_branching, rate_law, force_switch=False)
        return T


    def generate_batch(self, train_sequence, train_range):
        """
        Generates a data batch using train_range=[first_train_id, last_train_id), taking the first self.batch_sz elements of that range
        Note that the interval above is closed on the left and open on the right, meaning that first_train_id is included, but last_train_id never is.
        If first_train_id+self.batch_sz >= last_train_id, then a batch containing [first_train_id, last_train_id) is returned
        Else a batch containing [first_train_id, first_train_id+self.batch_sz) is returned
        """
        if train_range[0] >= train_range[1]:
            return (), None

        first_train_id = train_range[0]
        last_train_id = min(first_train_id+self.batch_sz, train_range[1])
        batch_sz = last_train_id - first_train_id

        train_labels = train_sequence[first_train_id:last_train_id]
        first_couple = next(self.data_iterator[train_labels[0]])
        train_data = first_couple[0]
        train_tensorlabels = first_couple[1]
        for seq_locid in range(1,batch_sz):
            next_couple = next(self.data_iterator[train_labels[seq_locid]])
            train_data = torch.cat((train_data, next_couple[0]))
            train_tensorlabels = torch.cat((train_tensorlabels, next_couple[1]))

        train_range = (first_train_id, last_train_id)
        train_batch = (train_data, train_tensorlabels)
        return train_range, train_batch


    def make_train_sequence(self):

        if self.training_type=="random": # actually not random, this should be renamed 'stair' or something
            j = 0
            train_data=[]
            iterator = [itertools.cycle(self.dataset.train_data[k]) for k in range(len(self.dataset.train_data))]
            for i in range(self.sequence_length):
                if j == len(self.dataset.train_data):
                    j = 0
                train_data.append(next(iterator[j]))
                j += 1
            random.shuffle(train_data)
            self.train_sequence = train_data

        elif self.training_type=="ultrametric":
            if self.dynamic_T_thr > 0:
                #new_T = self.heuristic_temperature(self.dynamic_T_thr, rate_law='power')
                new_T = self.heuristic_temperature(rate_law='power')
                print('Adapting temperature to be {0:.2f} to enforce minimum visitation constraint\nPrevious temperature was {1:.2f}'.format(new_T, self.T))
                self.T = new_T

            seqgen = sequence_generator_temporal.TempCorr_SequenceGenerator()
            train_sequence, rates_vector = seqgen.generate_labels(
                self.sequence_first,
                self.sequence_length,
                self.energy_step,
                self.T,
                self.tree_depth,
                self.tree_branching,
                self.min_visit
            )
            self.train_sequence = train_sequence
            self.rates_vector = rates_vector


        elif self.training_type=="spatial_correlation":

            data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
            if self.preprocessing:
                preprocessor = preprocessing.train_preproc()
                rates_matrix = preprocessing.rates_preproc(preprocessor, data, self.T, 10)
            else:
                data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
                rates_matrix = rates_correlation.rates_cor(data, self.T, 10)

            train_sequence = sequence_generator_spatial.um_sequence_generator(
                self.sequence_first,
                rates_matrix,
                self.sequence_length,
                data,
                minimum_classcount = self.min_visit
            )

            self.train_sequence = train_sequence
            self.rates_matrix = rates_matrix


        elif self.training_type=="uniform":
            seqgen = sequence_generator_temporal.Uniform_SequenceGenerator(self.tree_branching, self.tree_depth)

            train_sequence, rates_vector = seqgen.generate_labels(
                self.sequence_first,
                self.sequence_length
                )

            self.train_sequence = train_sequence
            self.rates_vector = rates_vector


        elif self.training_type=="ladder_blocks1":
            n_classes = self.dataset.branching**self.dataset.depth

            n_splits = self.sequence_length // self.split_block_lengths[0]
            seq_rest = self.sequence_length % self.split_block_lengths[0]
            train_sequence = []

            for splt_id in range(n_splits): # MNIST patterns are numbers from 0 to 9
                train_sequence.extend([splt_id%n_classes for k in range(self.split_block_lengths[0])])
            if seq_rest > 0:
                train_sequence.extend([n_splits%n_classes for k in range(seq_rest)])

            self.train_sequence = train_sequence


        elif self.training_type=="random_blocks1":
            n_classes = self.dataset.branching**self.dataset.depth

            n_splits = self.sequence_length // self.split_block_lengths[0]
            seq_rest = self.sequence_length % self.split_block_lengths[0]
            train_sequence = []

            for splt_id in range(n_splits): # MNIST patterns are numbers from 0 to 9
                rand_splt_id = random.randint(0, n_classes-1)
                train_sequence.extend([rand_splt_id for k in range(self.split_block_lengths[0])])
            if seq_rest > 0:
                rand_splt_id = random.randint(0, n_classes-1)
                train_sequence.extend([n_splits%n_classes for k in range(seq_rest)])

            self.train_sequence = train_sequence


        elif self.training_type=="ladder_blocks2":
            n_classes = self.dataset.branching**self.dataset.depth

            n_splits = self.sequence_length // self.split_block_lengths[0]
            seq_rest = self.sequence_length % self.split_block_lengths[0]

            train_sequence = []

            for splt_id in range(n_splits): # MNIST patterns are numbers from 0 to 9
                # Initiate transition rates based on splitID
                lbls = [(2*splt_id)%n_classes, (2*splt_id+1)%n_classes]
                cl_ids = np.random.randint(0, 2, size=self.split_block_lengths[0])
                train_sequence.extend([lbls[cl_ids[k]] for k in range(self.split_block_lengths[0])])
            if seq_rest > 0:
                lbls = [(2*n_splits)%n_classes, (2*n_splits+1)%n_classes]
                cl_ids = np.random.randint(0, 2, size=seq_rest)
                train_sequence.extend([lbls[cl_ids[k]] for k in range(seq_rest)])

            self.train_sequence = train_sequence


        elif self.training_type=="random_blocks2":
            n_classes = self.dataset.branching**self.dataset.depth

            n_splits = self.sequence_length // self.split_block_lengths[0]
            seq_rest = self.sequence_length % self.split_block_lengths[0]
            train_sequence = []

            for splt_id in range(n_splits): # MNIST patterns are numbers from 0 to 9
                # Initiate transition rates based on splitID
                rand_splt_id = random.randint(0, (n_classes//2)-1)
                lbls = [(2*rand_splt_id)%n_classes, (2*rand_splt_id+1)%n_classes]
                cl_ids = np.random.randint(0, 2, size=self.split_block_lengths[0])
                train_sequence.extend([lbls[cl_ids[k]] for k in range(self.split_block_lengths[0])])
            if seq_rest > 0:
                rand_splt_id = random.randint(0, (n_classes//2)-1)
                lbls = [(2*rand_splt_id)%n_classes, (2*rand_splt_id+1)%n_classes]
                cl_ids = np.random.randint(0, 2, size=seq_rest)
                train_sequence.extend([lbls[cl_ids[k]] for k in range(seq_rest)])

            self.train_sequence = train_sequence

        elif self.training_type=="random_blocks2_2freq":
            n_classes = self.dataset.branching**(self.dataset.depth-1)

            n_long_splits = self.sequence_length // self.split_block_lengths[1]
            seq_rest_long = self.sequence_length % self.split_block_lengths[1]
            n_short_splits = self.split_block_lengths[1] // self.split_block_lengths[0]
            n_shorts_splits_in_rest = self.split_block_lengths[1] // seq_rest_long
            seq_rest_short = self.split_block_lengths[1] % seq_rest_long
            train_sequence = []

            for l_splt_id in range(n_long_splits):
                rand_l_splt_id = random.randint(0, 1)
                for s_splt_id in range(n_short_splits):
                    rand_s_splt_id = random.randint(0, (n_classes//2)-1)
                    lbls = [n_classes*rand_l_splt_id+(2*rand_s_splt_id)%n_classes, n_classes*rand_l_splt_id+(2*rand_s_splt_id+1)%n_classes]
                    cl_ids = np.random.randint(0, 2, size=self.split_block_lengths[0])
                    train_sequence.extend([lbls[cl_ids[k]] for k in range(self.split_block_lengths[0])])

            self.train_sequence = train_sequence


        else:
            raise NotImplementedError("training type not supported")

        self.data_iterator = [itertools.cycle(self.dataset.train_data[i]) for i in range(len(self.dataset.train_data))]


    def train(self, mem_sz, training_range, seq=None, method='sgd', verbose_lvl=0):
        """
        Train a network on the specified training protocol.

        Parameters
        ----------
        net : neural network
            The neural network to train.
        training
            Trainer object parametrizing the training protocol.
        control_data
            Data used to do the training.
        mem_sz : int
            Size of the memory.
        batch_sz : int
            Size of the mini-batches for the training.
        lr : int
            Learning rate.
        momentum : int
            Momentum for the SGD.
        training_range : list
            Range of the data to train on. training_range[0] is the begining
            sample, training_range[1] is the end sample
        verbose_lvl: int
            verbove level used to decide whether to output performance throughout training

        Returns
        -------
        None.

        """
        if self.training_type in ("ultrametric", "ladder_blocks1", "ladder_blocks2", "random_blocks1", "random_blocks2", "random_blocks2_2freq", "spatial_correlation", "random", "uniform"):
            # Set model in training-mode
            self.network.train()
            # For temporal and spatial correlation tests
            if seq is not None:
                train_sequence = seq
            else:
                train_sequence = self.train_sequence

            first_train_id = training_range[0]
            last_train_id = training_range[1]
            running_loss = 0.0

            memory_list = [next(self.data_iterator[train_sequence[0]])]
            # Define mini-batches of size training.batch_sz and SGD and update the memory for each of them
            batch_range, batch = self.generate_batch(train_sequence, (first_train_id, last_train_id))
            while batch is not None:
                # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
                if isinstance(self.network, ContinualLearner) and (self.network.ewc_lambda>0):
                    verbose("Estimating Fisher information on last {:d} elements".format(self.batch_sz), verbose_lvl, 2)
                    self.network.estimate_fisher(batch)

                # Sample elements from memory at random and add it to the mini batch expect for the first iteration
                if first_train_id != 0 and mem_sz != 0:
                    # Sample the memory. We could choose to reduce the number of elements taken from memory, should make things more difficult
                    sample_memory = memory.sample_memory(memory_list, self.batch_sz)
                    train_batch = [
                        torch.cat((batch[0], sample_memory[0])),
                        torch.cat((batch[1], sample_memory[1]))
                    ]
                else:
                    train_batch = batch

                # Perform loss computation (including CL method-specific loss), and optimization step on the batch and memory
                running_loss += self.mem_optim(train_batch)

                # Update memory
                memory_list = memory.reservoir(memory_list, mem_sz, first_train_id, batch) if self.memory_sampling == "reservoir sampling" else memory.ring_buffer(memory, mem_sz, first_train_id, batch)

                # Get new batch
                first_train_id += self.batch_sz
                batch_range, batch = self.generate_batch(train_sequence, (first_train_id, last_train_id))
                if first_train_id % (1000*self.batch_sz) == 999*self.batch_sz:
                    verbose('[%d] loss: %.4f' % (first_train_id//self.batch_sz + 1, running_loss/1000), verbose_lvl, 2)
                    running_loss = 0.0


            verbose("--- Finished Experience Replay training on {0:d}-{1:d} ---".format(training_range[0], training_range[1]), verbose_lvl, 2)

        else:
            raise NotImplementedError("sequence generation not recognized.")


    def evaluate(self):
        # Set model in training-mode
        self.network.eval()
        # Return the accuracy, the predicted and real label for the whole test set and the difference between the two
        # Creation of the testing sequence
        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data))]
        for i in range(len(self.dataset.test_data)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))
        shuffle(test_sequence)

        correct = 0
        total = 0
        # Array which will contain the predicted output, the ground truth and the difference of the two
        result = np.zeros((len(test_sequence), 3))

        with torch.no_grad():
            for i, data in enumerate(test_sequence):
                samples, labels = data
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.network(samples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                result[i][0] = predicted
                result[i][1] = labels
        result[:, 2] = np.abs(result[:, 0] - result[:, 1])
        return (100*correct/total, result)



    def evaluate_hierarchical(self):
        # Return the accuracy, the predicted and GT and the distance between them for every hierachical level
        # Creation of the testing sequence
        if self.dataset.data_origin=='MNIST' or self.dataset.data_origin=='CIFAR10':
            excluded_labels = [8, 9]
        elif self.dataset.data_origin=='CIFAR100':
            excluded_labels = range(64, 100)

        else:
            excluded_labels = []

        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data) - len(excluded_labels))]   # Create the test sequence with only labels on which the network as been trained on
        for i in range(len(self.dataset.test_data) - len(excluded_labels)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))

        shuffle(test_sequence)

        # Array which will contain the accuracies at the different hierarchical levels,
        # the predicted output, the ground truth and the difference of the two
        result = [[], np.zeros((len(test_sequence), 3 + self.tree_depth))]
        with torch.no_grad():
            for i, data in enumerate(test_sequence):
                samples, labels = data
                if labels not in excluded_labels:
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    outputs = self.network(samples)
                    _, predicted = torch.max(outputs.data, 1)
                    result[1][i][0] = predicted
                    result[1][i][1] = labels
        zero = np.zeros(len(test_sequence))
        # Compute the difference between prediction and GT for every hierarchical level
        for i in range(2, self.tree_depth + 3):
            result[1][:, i] = np.abs((result[1][:, 0]//(self.tree_branching**(i-2)))
                                - (result[1][:, 1]//(self.tree_branching**(i-2))))
            result[0].append((np.sum(result[1][:, i] == zero)/len(test_sequence))*100)

        #pdb.set_trace()
        return result


    def shuffle_block_partial(self, block_size, end):
        """
        Block shuffle a data sequence up to a certain point.

        The part of the sequence aft

        Parameters
        ----------
        train_data : list of torch.Tensor
            List of the ordered samples used in the training process.
        block_size : int
            Size of the block for the shuffle.
        end: int
            Indice of the last sample of the sequence to shuffle

        Returns
        -------
        control_data_block : list of torch.Tensor
            List of the samples shuffled by blocks.

        """
        n_extra_labels = len(self.train_sequence)%block_size
        block_indices = list(range(len(self.train_sequence[:end])//block_size))
        shuffle(block_indices)
        block_indices += list(range(len(self.train_sequence[:end])//block_size, len(self.train_sequence)//block_size))     #add the rest of the data unshuffled to have everything work smoothly with older code. Not optimal but simpler

        idx_shuffled = list(
            map(
                add,
                np.repeat([block_size*k for k in block_indices], block_size).tolist(),
                list(range(block_size))*len(block_indices)
            )
        )

        shuffled_labels = [self.train_sequence[k] for k in idx_shuffled]
        shuffled_labels += self.train_sequence[-n_extra_labels:]

        return shuffled_labels


    def testing_final(self, verbose_lvl=1):
        # Define which label to exclude in the testing phase (for simplicity, to have a simple tree structure, certain labels have to be excluded
        # depending on the branching and depth of the tree)
        if self.dataset.data_origin=='MNIST' or self.dataset.data_origin=='CIFAR10':
            excluded_labels = [8, 9]
        elif self.dataset.data_origin=='CIFAR100':
            excluded_labels = range(64, 100)

        else:
            excluded_labels = []
        # Creation of the testing sequence
        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data))]
        for i in range(len(self.dataset.test_data)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))
        shuffle(test_sequence)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_sequence:
                samples, labels = data
                if labels not in excluded_labels:
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    outputs = self.network(samples)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        verbose('Accuracy of the network on the %d test images: %.2f %%' % (total, 100 * correct / total), verbose_lvl, 2)
        return (100*correct/total, test_sequence)

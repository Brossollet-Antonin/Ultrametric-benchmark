# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:09:45 2019

@author: Antonin
"""

import torch
import torchvision
import numpy as np
import random
from copy import deepcopy
import pdb


def count_differences(seq1, seq2):
    return np.sum([0.25*(seq1[k] - seq2[k])**2 for k in range(len(seq1))])


def sort_dataset(dataset, train):
    # If True, return sorted train set, else return sorted test set
    if dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
            batch_size=1, shuffle=True)
    elif dataset == 'CIFAR10':
         train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])),
            batch_size=1, shuffle=True)
    elif dataset =='CIFAR100':
         train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])),
            batch_size=1, shuffle=True)

    train_data_sorted = [[] for i in range(10)] if (dataset=='MNIST' or dataset=='CIFAR10') else [[] for i in range(100)]
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        train_data_sorted[int(labels.item())].append(data)
    return train_data_sorted


def sort_MNIST(train):
    # If True, return sorted train set, else return sorted test set
    train_data_sorted= [[] for i in range(10)]
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=train, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ])),
        batch_size=1, shuffle=True)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        train_data_sorted[int(labels)].append(data)
    return train_data_sorted


class Dataset:

    """Contains artifical dataset parameters and data.

    Parameters
    ----------
    data_origin : str
        Dataset used.
    data_sz : int, default=1
        Length of data vectors generated for an artifical dataset.
    tree_depth : int, default=3
        Depth of the ultrametric tree used to generate artificial dataset.
    class_sz_train : int, default=0
        Number of samples generated for each label for the training dataset
        an artificial dataset.
    class_sz_test : int, default=0
        Number of samples generated for each label for the testing dataset for
        an artificial dataset.
    ratio_type : str, default='linear'
        Method used to generate the artificial dataset, specified how many bits
        will be flipped between each level of the ultrametric tree.
    ratio_value : int, default=5
        Number of flipped bits between each level of the tree. If
        ratio_type='linear' ratio_value bits will be flipped, if
        ratio_type='exponnential' (1/ratio_value)**depth bits will be flipped.
    noise_level : int, default=1
        Noise level to generate the samples for each class. noise_level bits
        will be flipped randomly to generate each sample of the class.

    Attributes
    ----------
    data_origin : str
        Dataset used.
    data_sz : int
        Length of data vectors generated for an artifical dataset.
    depth : int
        Depth of the ultrametric tree used to generate artificial dataset.
    class_sz_train : int
        Number of samples generated for each label for the training dataset for
        an artificial dataset.
    class_sz_test : int
        Number of samples generated for each label for the testing dataset for
        an artificial dataset.
    ratio_type : str
        Method used to generate the artificial dataset, specified how many bits
        will be flipped between each level of the ultrametric tree.
    ratio_value : int
        Number of flipped bits between each level of the tree. If
        ratio_type='linear' ratio_value bits will be flipped, if
        ratio_type='exponnential' (1/ratio_value)**depth bits will be flipped.
    noise_level : int
        Noise level to generate the samples for each class. noise_level bits
        will be flipped randomly to generate each sample of the class.
    branching : int
        Branching ratio of the ultrametric dataset associated with the dataset.
    num_classes : int
        Number of classes in the dataset.
    n_axes : int
        Number of axes of the data (e.g. 2 for 2D images).
    n_in_channels : int
        Number of input channels (e.g. 1 for B&W images, 3 for color images).
    train_data : list of list of torch.Tensor
        List containing the lists of tensors of the train data of the dataset,
        classed by labels (e.g. train_data[2][5] is the 5 tensor of the 2
        class of the dataset).
    test_data : list of list of torch.Tensor
        List containing the lists of tensors of the test data of the dataset,
        classed by labels (e.g. train_data[2][5] is the 5 tensor of the 2
        class of the dataset).

    """

    def __init__(self, data_origin, data_sz=0, tree_depth=3, class_sz_train=0,
                 class_sz_test=0, ratio_type='linear', ratio_value=5,
                 noise_level=1, shuffle_classes=True):
        self.data_origin = data_origin
        self.class_sz_train = class_sz_train
        self.data_sz = data_sz
        self.class_sz_test = class_sz_test
        self.ratio_type = ratio_type
        self.ratio_value = ratio_value
        self.noise_level = noise_level
        self.shuffle_classes = shuffle_classes

        if data_origin == 'artificial':
            self.depth, self.branching = tree_depth, 2
            self.num_classes = self.branching**self.depth
            self.n_axes, self.n_in_channels = 1, 1
        elif data_origin == 'MNIST':
            self.depth, self.branching = 3, 2
            self.num_classes = 10
            self.n_axes, self.n_in_channels = 2, 1
            self.data_sz = (28**2)
        elif data_origin == 'CIFAR10':
            self.depth, self.branching = 3, 2
            self.num_classes = 10
            self.n_axes, self.n_in_channels = 2, 3
            self.data_sz = (32**2)*3
        elif data_origin == 'CIFAR100':
            self.depth, self.branching = 6, 2
            self.num_classes = 100
            self.n_axes, self.n_in_channels = 2, 3
            self.data_sz = (32**2)*3

        if 'artificial' not in data_origin:
            self.train_data = sort_dataset(dataset=data_origin, train=True)
            self.test_data = sort_dataset(dataset=data_origin, train=False)
            self.patterns = []
            if data_origin=='MNIST':
                self.class_sz_test = 892    # The class 5 of MNIST as only 892 test samples
            elif data_origin=='CIFAR10':
                self.class_sz_test = 1000
        else:
            if self.num_classes > 500:
                self.class_sz_test = 20000//self.num_classes
            elif self.num_classes > 100:
                self.class_sz_test = 5000//self.num_classes
            else:
                self.class_sz_test = 300
            #self.create()
            self.create_power(self.shuffle_classes)


    def create(self, shuffle_labels=False):
        # Creating an initial random tensor of size data_sz, of +1 and -1
        initial = torch.randint(2,(self.data_sz,), dtype=torch.float)
        initial = initial*2.0 -1.0

        # List to stock the parents to allow the creation of the different
        parents_mem = [initial]
        self.patterns = [[] for k in range(self.depth + 1)]
        self.patterns[0].append(parents_mem)
        train_data = [[] for i in range(self.branching**self.depth)]
        test_data = [[] for i in range(self.branching**self.depth)]
        label = 0
        d = 0
        b = 0
        while label < self.branching**self.depth:
            if b == self.branching:
                parents_mem.pop()
                b = 0
                d -= 1
            next_parent = deepcopy(parents_mem[-1])

            if self.ratio_type == 'linear':
                for s in range(self.ratio_value):
                    ind_mod = random.randint(0, self.data_sz -1)
                    next_parent[ind_mod] = next_parent[ind_mod]*(-1)
            elif self.ratio_type == 'exponnential':
                for s in range(int(self.data_sz*(1/self.ratio_value)**(d+1))):
                    ind_mod = random.randint(0, self.data_sz -1)
                    next_parent[ind_mod] = next_parent[ind_mod]*(-1)
            else:
                raise NotImplementedError("Supported modes are for the moment 'linear' and 'exponnential'")
            # Switch the value of the ind_mod digit

            parents_mem.append(next_parent)
            d += 1
            self.patterns[d].append(next_parent)

            # If at the bottom of the tree, create the random samples
            if d == self.depth:

            # ToDo: Add option to shuffle all leaves randomly to destroy link between temporal and spatial correlations
                b += 1
                # Creating train set
                for k in range(self.class_sz_train):
                    new_example_train = deepcopy(parents_mem[-1])
                    # Add noise
                    for s in range(self.noise_level):
                        ind_mod_ex_train = random.randint(0, self.data_sz -1)
                        new_example_train[ind_mod_ex_train] = new_example_train[ind_mod_ex_train]*(-1)
                    # Add mini batch size and number of channel to have a correctly formated tensor for training
                    new_example_train = torch.unsqueeze(new_example_train, 0)
                    new_example_train = torch.unsqueeze(new_example_train, 0)
                    train_data[label].append([new_example_train, torch.tensor([label])])
                # Creating test set
                for k in range(self.class_sz_test):
                    new_example_test = deepcopy(parents_mem[-1])
                    # Add noise
                    for s in range(self.noise_level):
                        ind_mod_ex_test = random.randint(0, self.data_sz -1)
                        new_example_test[ind_mod_ex_test] = new_example_test[ind_mod_ex_test]*(-1)
                    # Add mini batch size and number of channel to have a correctly formated tensor for training
                    new_example_test = torch.unsqueeze(new_example_test, 0)
                    new_example_test = torch.unsqueeze(new_example_test, 0)
                    test_data[label].append([new_example_test, torch.tensor([label])])

                label += 1
                d -= 1

        self.train_data = train_data
        self.test_data = test_data
        if shuffle_labels:
            random.shuffle(self.train_data)
            random.shuffle(self.test_data)


    def create_power(self, shuffle_labels=True):
        """
        Generate the artificial dataset.

        Parameters
        ----------
        shuffle_labels : Bool, default=True
            True if the labels of the created dataset have to be shuffled after
            being created (to destroy the link between temporal and spacial
            correlations).

        Returns
        -------
        None.

        """

        # Creating an initial random tensor of size data_sz, of +1 and -1
        initial = torch.randint(2,(self.data_sz,), dtype=torch.float)
        initial = initial*2.0 -1.0

        self.patterns = [[] for d in range(self.depth+1)]
        self.patterns[0].append(initial)

        for d in range(1, self.depth+1):
            for pat_id in range(self.branching**d):
                template = deepcopy(self.patterns[d-1][pat_id//2])
                self.patterns[d].append(template)
                for s in range(self.ratio_value):
                    ind_mod = random.randint(0, self.data_sz -1)
                    self.patterns[d][pat_id][ind_mod] = self.patterns[d][pat_id][ind_mod]*(-1)

        train_data = [[] for i in range(self.branching**self.depth)]
        test_data = [[] for i in range(self.branching**self.depth)]

        for label in range(self.branching**self.depth):
            template = self.patterns[self.depth][label]
            for k in range(self.class_sz_train):
                new_example_train = deepcopy(template)
                # Add noise
                for s in range(self.noise_level):
                    ind_mod_ex_train = random.randint(0, self.data_sz -1)
                    new_example_train[ind_mod_ex_train] = new_example_train[ind_mod_ex_train]*(-1)
                # Add mini batch size and number of channel to have a correctly formated tensor for training
                new_example_train = torch.unsqueeze(new_example_train, 0)
                new_example_train = torch.unsqueeze(new_example_train, 0)
                train_data[label].append([new_example_train, torch.tensor([label])])

            for k in range(self.class_sz_test):
                new_example_test = deepcopy(template)
                # Add noise
                for s in range(self.noise_level):
                    ind_mod_ex_test = random.randint(0, self.data_sz -1)
                    new_example_test[ind_mod_ex_test] = new_example_test[ind_mod_ex_test]*(-1)
                # Add mini batch size and number of channel to have a correctly formated tensor for training
                new_example_test = torch.unsqueeze(new_example_test, 0)
                new_example_test = torch.unsqueeze(new_example_test, 0)
                test_data[label].append([new_example_test, torch.tensor([label])])

        self.train_data = train_data
        self.test_data = test_data

        if shuffle_labels:
            random.shuffle(self.train_data)
            random.shuffle(self.test_data)

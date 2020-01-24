# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import os
import pickle
import numpy as np
import json
import pdb

class ResultSet_Single:
    """Contains the results of a simulation

    Parameters
    ----------
    dataroot : str
        Path of the root where data are stored.
    datapath : str
        Path to the data.
        
    Attribute
    ---------
    dataroot : str
        Path of the root where data are stored.
    datapath : str
        Path to the data.    
    
    """
    
    def __init__(self, dataroot, datapath):
        self.dataroot = dataroot
        self.datapath = datapath
        
    def load_analytics(self, load_atc=False):
        self.train_data_orig = {}
        self.train_labels_orig = {}
        self.train_data_shfl = {}
        self.train_labels_shfl = {}
        self.dstr_train = {}
        self.params = {}
        self.atc_orig = {}
        self.atc_shfl = {}
        self.eval_orig = {}
        self.eval_shfl = {}
        self.var_acc_orig = {}
        self.var_acc_shfl = {}
        self.var_pred_orig = {}
        self.var_pred_shfl = {}
    
        os.chdir(self.dataroot+'/'+self.datapath)

        self.help = {} # will contain general information about stored analytics
            
        self.help['train_data_orig'] = """
        Type: list    Stored as: pickle
        Contains the training data inputs, for the original training sequence
        """
        file = open('train_data_orig.pickle', 'rb')
        self.train_data_orig = pickle.load(file)
        file.close()

        self.help['train_labels_orig'] = """
        Type: list    Stored as: pickle
        Contains the training labels, cast between 0 and N_labels, for the original training sequence
        """
        file = open('train_labels_orig.pickle', 'rb')
        self.train_labels_orig = pickle.load(file)
        file.close()

        self.help['train_data_shfl'] = """
        Type: list    Stored as: pickle
        Contains the training data inputs, for the shuffled training sequence
        """
        file = open('train_data_shfl.pickle', 'rb')
        self.train_data_shfl = pickle.load(file)
        file.close()

        self.help['train_labels_shfl'] = """
        Type: list    Stored as: pickle
        Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
        """
        file = open('train_labels_shfl.pickle', 'rb')
        self.train_labels_shfl = pickle.load(file)
        file.close()

        self.help['distribution_train'] = """
        Type: list    Stored as: pickle
        Counts, for each label, the corresponding number of training example
        """
        file = open('distribution_train.pickle', 'rb')
        self.dstr_train = pickle.load(file)
        file.close()

        self.help['parameters'] = """
        Type: list    Stored as: pickle
        Counts, for each label, the corresponding number of training example
        """
        file = open('parameters.pickle', 'rb')
        self.params = pickle.load(file)
        file.close()

        if load_atc:
	        self.help['autocorr_original.npy'] = """
	        Type: array    Stored as: npy
	        The autocorrelation function as computed by statsmodels.tsa.stattools.act
	        """
	        self.atc_orig = np.load('autocorr_original.npy')
	            
	        self.help['autocorr_shuffle.npy'] = """
	        Type: array    Stored as: npy
	        A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
	        """
	        self.atc_shfl = np.load('autocorr_shuffle.npy')

        self.help['diagnostic_original.npy'] = """
        Type: array    Stored as: npy
        [0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
        [1][0] contains the GT pointwise to the testing sequence
        [1][1] contains the prediction pointwise to the testing sequence
        [1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
        """
        self.eval_orig = np.load('diagnostic_original.npy', allow_pickle=True)
        self.eval_shfl = np.load('diagnostic_shuffle.npy', allow_pickle=True)

        self.help['var_original_accuracy.npy'] = """
        Type: array    Stored as: npy
        [0] Average accuracy over full test sequence
        [1:test_nbr] Average accuracy over each test run
        """
        self.var_acc_orig = np.load('var_original_accuracy.npy')
        self.var_acc_shfl = np.load('var_shuffle_accuracy.npy')

        self.help['var_original_classes_prediction.npy'] = """
        Type: array    Stored as: npy
        [0:test_nbr] Contains, for each test run, the composition of the test sampl,
        as well as the progress of training as the max training ID scanned at the time of the test run
        """
        self.var_pred_orig = np.load('var_original_classes_prediction.npy', allow_pickle=True)
        self.var_pred_shfl = np.load('var_shuffle_classes_prediction.npy', allow_pickle=True)



class ResultSet:
	def __init__(self, dataroot, datapaths):
		self.dataroot = dataroot
		self.datapaths = datapaths
		
	def load_analytics(self, load_data=False, load_atc=False, load_shuffle=True):

		self.train_data_orig = {}
		self.train_labels_orig = {}
		self.train_data_shfl = {}
		self.train_labels_shfl = {}
		self.dstr_train = {}
		self.params = {}
		self.atc_orig = {}
		self.atc_shfl = {}
		self.eval_orig = {}
		self.eval_shfl = {}
		self.var_acc_orig = {}
		self.var_acc_shfl = {}
		self.var_pred_orig = {}
		self.var_pred_shfl = {}
	
		self.help = {}

		self.help['train_labels_orig'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the original training sequence
				"""
		self.help['train_labels_shfl'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
				"""
		self.help['distribution_train'] = """
				Type: list    Stored as: pickle
				Counts, for each label, the corresponding number of training example
				"""
		self.help['parameters'] = """
				Type: list    Stored as: JSON
				Refers the different parameters and hyperparameters used for this set of simulations
				"""

		self.help['diagnostic_original.npy'] = """
				Type: array    Stored as: npy
				[0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
				[1][0] contains the GT pointwise to the testing sequence
				[1][1] contains the prediction pointwise to the testing sequence
				[1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
				"""
		self.help['var_original_accuracy.npy'] = """
				Type: array    Stored as: npy
				[0] Average accuracy over full test sequence
				[1:test_nbr] Average accuracy over each test run
				"""
		self.help['var_original_classes_prediction.npy'] = """
				Type: array    Stored as: npy
				[0:test_nbr] Contains, for each test run, the composition of the test sampl,
				as well as the progress of training as the max training ID scanned at the time of the test run
				"""

		if load_data:
			self.help['train_data_orig'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the original training sequence
					"""
			self.help['train_data_shfl'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the shuffled training sequence
					"""
		else:
			self.help['train_data_orig'] = """
					Unavailable. load_data set to False
					"""
			self.help['train_data_shfl'] = """
					Unavailable. load_data set to False
					"""

		if load_atc:
			self.help['autocorr_original.npy'] = """
					Type: array    Stored as: npy
					The autocorrelation function as computed by statsmodels.tsa.stattools.act
					"""
			self.help['autocorr_shuffle.npy'] = """
					Type: array    Stored as: npy
					A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
					"""
		else:
			self.help['autocorr_original.npy'] = """
					Unavailable. load_atc set to False
					"""
			self.help['autocorr_shuffle.npy'] = """
					Unavailable. load_atc set to False
					"""

		for params, datapath_list in self.datapaths.items():
			
			self.train_labels_orig[params] = []
			self.train_labels_shfl[params] = []
			self.dstr_train[params] = []
			self.params[params] = []
			self.eval_orig[params] = []
			self.eval_shfl[params] = []
			self.var_acc_orig[params] = []
			self.var_acc_shfl[params] = []
			self.var_pred_orig[params] = []
			self.var_pred_shfl[params] = []

			if load_data:
				self.train_data_orig[params] = []
				self.train_data_shfl[params] = []

			if load_atc:
				self.atc_orig[params] = []
				self.atc_shfl[params] = []

			for datapath in datapath_list:
				os.chdir(self.dataroot+'/'+datapath)

				with open('train_labels_orig.pickle', 'rb') as file:
					self.train_labels_orig[params].append(pickle.load(file))

				with open('distribution_train.pickle', 'rb') as file:
					self.dstr_train[params].append(pickle.load(file))
				
				with open('parameters.json', 'r') as file:
					self.params[params].append(json.load(file))

				self.eval_orig[params].append(np.load('evaluation_original.npy', allow_pickle=True))
				self.var_acc_orig[params].append(np.load('var_original_accuracy.npy'))
				self.var_pred_orig[params].append(np.load('var_original_classes_prediction.npy', allow_pickle=True))

				if load_shuffle:
					with open('train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[params].append(pickle.load(file))
					self.eval_shfl[params].append(np.load('evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[params].append(np.load('var_shuffle_accuracy.npy'))
					self.var_pred_shfl[params].append(np.load('var_shuffle_classes_prediction.npy', allow_pickle=True))

				if load_data:
					print("Loading data for {0:s}...".format(datapath))

					with open('train_data_orig.pickle', 'rb') as file:
						self.train_data_orig[params].append(pickle.load(file))

					with open('train_data_shfl.pickle', 'rb') as file:
						self.train_data_shfl[params].append(pickle.load(file))
					
					print("...done")

				if load_atc:
					self.atc_orig[params].append(np.load('autocorr_original.npy'))
					self.atc_shfl[params].append(np.load('autocorr_shuffle.npy'))

		if not load_data:
			print("load_data set to False. Data sequences not loaded.")
		if not load_atc:
			print("load_atc set to False. Autocorrelations not loaded.")
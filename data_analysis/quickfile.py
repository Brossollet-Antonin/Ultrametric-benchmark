# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:17:24 2019

@author: Antonin
"""

import matplotlib.pyplot as plt
import numpy as np 
import pickle 

obj=pickle.load(open(rpath, 'rb'))

obj=np.load(rpath, allow_pickle=True)

plt.plot(var_shuffle_accuracy[:,1], var_shuffle_accuracy[:,0])
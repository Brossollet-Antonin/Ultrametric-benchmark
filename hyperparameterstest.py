# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:17:54 2019

@author: Antonin
"""

import numpy as np
import os
#import matplotlib.pyplot as plt

savepath = "./Results/"

#Temperature = np.arange(1, 3, 0.2)
Temperature = [0.5]

# Warning: must verify that depth_seq < depth_data for all the cases
depth_tree = [3]
memory_sz = 0
epoch = 0
minibatches = 10
step = 3
tree_branching_list = [4]


for depth in depth_tree: 
    for T in Temperature:
        for tree_branching in tree_branching_list:
            data_branching = tree_branching
            os.makedirs(savepath+"T%.1f Depth%d Memory%d Branching%d" % (T, depth, memory_sz, tree_branching))
            exec(open("./firsttest.py").read()) 
            plt.close("all")
        
        


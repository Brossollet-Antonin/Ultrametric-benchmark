# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:39:14 2020

@author: Simon
"""

import numpy as np

def verbose(message, args, lvl=1):
	if args.verbose >= lvl:
		print(message)

def base_conv(value, base):
    # Equivalent to bin(value) but for an arbitrary base. Return a string in the given base
    res = ''
    while value > 0:
        res = str(value % base) + res 
        value = value//base
    return res

def make_ohe(y, n_labels):
    ohe = np.zeros((len(y), n_labels))    
    ohe[np.arange(len(y)),y] = 1
    return ohe
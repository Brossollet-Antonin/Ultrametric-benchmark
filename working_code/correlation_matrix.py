# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:33:04 2019

@author: Antonin
"""

import numpy as np

XY_original = np.zeros((50,32))
for k in range(len(train_data)):
    XY_original = XY_original + np.transpose(train_data[k][0].numpy()[0]) @ netfc_original(train_data[k][0]).detach().numpy()

XY_original = XY_original / len(train_data)
plt.figure()
plt.imshow(XY_original)
plt.title('Original')
plt.colorbar()


XY_shuffle = np.zeros((50,32))
for k in range(len(control_sequence_shuffle)):
    XY_shuffle = XY_shuffle + np.transpose(control_sequence_shuffle[k][0].numpy()[0]) @ netfc_shuffle(control_sequence_shuffle[k][0]).detach().numpy()

XY_shuffle = XY_shuffle / len(control_sequence_shuffle)
plt.figure()
plt.imshow(XY_shuffle)
plt.title('Shuffle')
plt.colorbar()



XY_labels = np.zeros((50,32))
for k in range(len(control_sequence_labels1)):
    XY_labels = XY_labels + np.transpose(control_sequence_labels1[k][0].numpy()[0]) @ netfc_original(control_sequence_labels1[k][0]).detach().numpy()

XY_labels = XY_labels / len(control_sequence_labels1)
plt.figure()
plt.imshow(XY_labels)
plt.title('Labels')
plt.colorbar()


SVD_original=np.linalg.svd(XY_original)
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import os
import pickle
import numpy as np

if args.verbose:
    print('Entering data saving script.\nSaving to {0:s}'.format(savepath+save_folder))

os.makedirs(savepath + save_folder)

filename = savepath + save_folder + "/train_data_orig.pickle"
outfile = open(filename, 'wb')
pickle.dump(train_data, outfile)
outfile.close

filename = savepath + save_folder + "/train_labels_orig.pickle"
outfile = open(filename, 'wb')
pickle.dump(train_labels, outfile)
outfile.close()


seq_control_shuffle=[]
for k in control_data_shuffle:
    seq_control_shuffle.append(k[1].item())

filename = savepath + save_folder + "/train_data_shfl.pickle"
outfile = open(filename, 'wb')
pickle.dump(control_data_shuffle, outfile)
outfile.close

filename = savepath + save_folder + "/train_labels_shfl.pickle"
outfile = open(filename, 'wb')
pickle.dump(train_labels_sfl, outfile)
outfile.close()


filename = savepath + save_folder + "/distribution_train.pickle"
outfile = open(filename, 'wb')
pickle.dump(compteur, outfile)
outfile.close()

np.save(savepath+save_folder+"/parameters", parameters)

np.save(savepath + save_folder + "/evaluation_original", eval_original)
np.save(savepath + save_folder + "/evaluation_shuffle", eval_shuffle)

np.save(savepath+save_folder+'/var_original_classes_prediction', original_classes_prediction)
np.save(savepath+save_folder+'/var_shuffle_classes_prediction', shuffle_classes_prediction)
np.save(savepath+save_folder+'/var_original_accuracy', original_accuracy)
np.save(savepath+save_folder+'/var_shuffle_accuracy', shuffle_accuracy)

np.save(savepath+save_folder+'/autocorr_original', original_autocorr_function)
np.save(savepath+save_folder+'/autocorr_shuffle', shuffle_autocorr_functions)

if args.verbose >= 2:
    print('Saved all arrays to {0:s} subfolders'.format(savepath+save_folder))

if args.verbose >= 2:
    print('Saved train data to {0:s} subfolders'.format(savepath+save_folder))
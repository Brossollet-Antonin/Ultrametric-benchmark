# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import pickle
import numpy as np

if args.verbose:
    print('Entering data saving script.\nSaving to {0:s}'.format(savepath+save_folder))



filename = savepath + save_folder + "/original"
outfile = open(filename, 'wb')
pickle.dump(train_sequence, outfile)
outfile.close()


seq_control_shuffle=[]
for k in control_data_shuffle:
    seq_control_shuffle.append(k[1].item())

filename = savepath + save_folder + "/shuffle"
outfile = open(filename, 'wb')
pickle.dump(seq_control_shuffle, outfile)
outfile.close()

filename = savepath + save_folder + "/distribution_train"
outfile = open(filename, 'wb')
pickle.dump(compteur, outfile)
outfile.close()

np.save(savepath+save_folder+"/parameters", parameters)

np.save(savepath + save_folder + "/diagnostic_original", diagnos_original)
np.save(savepath + save_folder + "/diagnostic_shuffle", diagnos_shuffle)

np.save(savepath+save_folder+'/var_original_classes_prediction', original_classes_prediction)
np.save(savepath+save_folder+'/var_shuffle_classes_prediction', shuffle_classes_prediction)
np.save(savepath+save_folder+'/var_original_accuracy', original_accuracy)
np.save(savepath+save_folder+'/var_shuffle_accuracy', shuffle_accuracy)

np.save(savepath+save_folder+'/autocorr_original', original_autocorr_function)
np.save(savepath+save_folder+'/autocorr_shuffle', shuffle_autocorr_functions)

if args.verbose >= 2:
    print('Saved all arrays to {0:s} subfolders'.format(savepath+save_folder))

filename = savepath + save_folder + "/data_shuffle"
outfile = open(filename, 'wb')
pickle.dump(control_data_shuffle, outfile)
outfile.close

filename = savepath + save_folder + "/train_data"
outfile = open(filename, 'wb')
pickle.dump(train_data, outfile)
outfile.close

if args.verbose >= 2:
    print('Saved train data to {0:s} subfolders'.format(savepath+save_folder))
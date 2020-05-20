import os, sys
sys.path.append('../')

import numpy as np
from scipy.io import savemat
import pdb
import utils
import ast, json
from distutils.dir_util import copy_tree
import argparse

paths = utils.get_project_paths()
um_root = paths['simus']+'1toM/artificial_32/FCL20/'

parser = argparse.ArgumentParser(paths['misc']+'sort_results_byratio.py', description='Sorting simlation file by number of flips in the data patterns sequences')
parser.add_argument('--simuset_root', help="Simulation set folder to sort")
parser.add_argument('--seq_len', type=int, default=200, help="Length of the data pattern sequences")

args = parser.parse_args()

simus_root = os.listdir(args.simuset_root)
bynflips_sorter = {}
for simu_subdir in simus_root:
    # For all nflips simulated, list the corresponding simulation subfolders
    with open(args.simuset_root+'/'+simu_subdir+'/parameters.json', 'r') as param_file:
        simu_params = json.loads(param_file.read())
    try:
        flips_ratio = float(simu_params["Flips ratio"])
    except:
        pdb.set_trace()
    nflips = int(flips_ratio*args.seq_len)
    if nflips not in bynflips_sorter.keys():
        bynflips_sorter[nflips] = []
    bynflips_sorter[nflips].append(simu_subdir)

# Create variations of simuset directories specific to each nflips
for nflips in bynflips_sorter.keys():
    new_folder = args.simuset_root+'_ratio{}'.format(nflips)
    os.mkdir(new_folder)
    for simu_subdir in bynflips_sorter[nflips]:
        copy_tree(args.simuset_root+'/'+simu_subdir, new_folder+'/'+simu_subdir)


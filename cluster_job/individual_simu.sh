#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…

module load anaconda

#Command to execute Python program
python3 ${15} --nnarch FCL --data_tree_depth ${1} --hidden_sizes ${2} \
--seqlength ${3} --split_length ${4} -T ${5} --seqtype ${6} \
--optimizer ${7} --dataset ${8} --nbrtest ${9} --data_flips_rate ${10} --shuffle_classes ${11} --nonlin ${12} --lr ${13} --blocksz ${14} --verbose 1

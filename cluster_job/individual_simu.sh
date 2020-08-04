#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…

module load anaconda

#Command to execute Python program
python3 ${1} --dataset ${2} --data_tree_depth ${3} -T ${4} --nnarch {5} --hidden_sizes ${6} \
--optimizer ${7} --nonlin ${8} --lr ${9} --seqtype ${10} --seqlength ${11} --split_length ${12} \
--nbrtest ${13} --data_flips_rate ${14} --shuffle_classes ${15} --blocksz ${16} --verbose 1 \
  

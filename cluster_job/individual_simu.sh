#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…

module load anaconda

#Command to execute Python program
python3 ${5} --dataset ${6} --data_tree_depth ${7} -T ${8} --nnarch ${9} --hidden_sizes ${10} \
--optimizer ${11} --nonlin ${12} --lr ${13} --seqtype ${14} --seqlength ${15} --split_length ${16} \
--nbrtest ${17} --data_flips_rate ${18} --shuffle_classes ${19} --blocksz ${20} --verbose 1 \
  

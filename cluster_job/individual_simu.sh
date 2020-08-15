#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…

module load anaconda

#Command to execute Python program
python3 ${6} --dataset ${7} --data_tree_depth ${8} -T ${9} --nnarch ${10} --hidden_sizes ${11} \
--optimizer ${12} --nonlin ${13} --lr ${14} --seqtype ${15} --seqlength ${16} --split_length ${17} \
--nbrtest ${18} --data_flips_rate ${19} --shuffle_classes ${20} --blocksz ${21} ${22} --verbose 1 \
  

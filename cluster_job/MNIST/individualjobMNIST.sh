#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="MNIST_UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --gres=gpu

module load anaconda

#Command to execute Python program
python3 ${10} --nnarch FCL --hidden_sizes ${1} --gpu \
--seqlength ${2} --split_length ${3} -T ${4} --seqtype ${5} \
--optimizer ${6} --dataset ${7} --nbrtest ${8} --blocksz ${9}

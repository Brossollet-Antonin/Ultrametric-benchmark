#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="MNIST ultrametric" # The job name.
#SBATCH -c 2 # The number of cpu cores to use.
#SBATCH --time=8:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb # The memory the job will use per cpu core.
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=antoninbrossollet@gmail.com

module load anaconda

#Command to execute Python program
python ../main.py --dataset MNIST --nnarch FCL --hidden_sizes 256 --seqlength 1200000 --split_length 1000 \
--minibatch ${1} --blocksz ${2} -T ${3} --seqtype ${4}

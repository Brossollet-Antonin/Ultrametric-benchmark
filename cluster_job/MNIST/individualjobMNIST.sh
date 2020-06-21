#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="MNIST_UMRB" # The job name.
#SBATCH -c 2 # The number of cpu cores to use.
#SBATCH --time=35:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb # The memory the job will use per cpu core.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=antoninbrossollet@gmail.com
#SBATCH --gres=gpu

module load anaconda

#Command to execute Python program
python3 /rigel/theory/users/ab4877/Ultrametric-benchmark/main.py --dataset MNIST --nnarch FCL --hidden_sizes ${1} --gpu \
--seqlength ${2} --split_length ${3} --nbrtest 150 \
-T ${4} --seqtype ${5} --optimizer ${6} --blocksz ${7}

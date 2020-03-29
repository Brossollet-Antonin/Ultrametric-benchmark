#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="artificial7" # The job name.
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=5:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=6gb # The memory the job will use per cpu core.
#SBATCH --gres=gpu

module load anaconda

#Command to execute Python program
python ../main.py -v --gpu --dataset artificial --data_tree_depth 7 --seqtype ${1} --data_seq_size 100 --nbrtest 100 --nnarchi CNN -T 0.15 0.4 0.65 --blocksz 10 100 1000

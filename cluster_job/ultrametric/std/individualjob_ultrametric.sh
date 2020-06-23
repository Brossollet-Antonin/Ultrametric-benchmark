#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="ARTIF_UMRB" # The job name.
#SBATCH -c 2 # The number of cpu cores to use.
#SBATCH --time=35:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

module load anaconda
source activate /rigel/theory/users/sl4744/anaconda3/envs/ultrametric

#Command to execute Python program
python3 /rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py --dataset artificial --data_tree_depth ${1} --nnarch FCL --hidden_sizes ${2} \
--seqlength ${3} --split_length ${4} --nbrtest 300 \
-T ${5} --seqtype ${6} --shuffle_classes ${7} --data_flips_rate ${8} --blocksz ${9}

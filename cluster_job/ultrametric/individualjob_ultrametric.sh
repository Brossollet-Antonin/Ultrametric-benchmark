#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --export=NONE
#SBATCH --job-name="ultrabenchmark" # The job name.
#SBATCH -c 2 # The number of cpu cores to use.
#SBATCH --time=40:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

module load anaconda
source activate /rigel/theory/users/sl4744/anaconda3/envs/ultrametric

#Command to execute Python program
python3 /rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py --dataset artificial --data_tree_depth ${1} --nnarch FCL --hidden_sizes ${2} \
--seqlength 300000 --split_length ${3} --nbrtest 300 \
-T ${4} --seqtype ${5} --shuffle_classes ${6} --data_flips_rate ${7} --blocksz ${8} ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}

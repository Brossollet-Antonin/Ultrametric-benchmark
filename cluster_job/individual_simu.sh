#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,…

echo "Runtime requested: ${1}"
echo "CPUs per task: ${2}"
echo "Mail user: ${3}"
echo "Memory per CPU: ${4}"
echo "Path to main program: ${5}"
echo "Dataset: ${6}"
echo "Tree depth: ${7}"
echo "Temperature: ${8}"
echo "Network architecture: ${9}"
echo "Hidden size: ${10}"
echo "Optimizer: ${11}"
echo "Nonlinearity: ${12}"
echo "Learning rate: ${13}"
echo "Sequence type: ${14}"
echo "Sequence length: ${15}"
echo "Split length: ${16}"
echo "Number of tests: ${17}"
echo "Data flips rate: ${18}"
echo "Shuffle classes:: ${19}"
echo "Block sizes: ${20}"

#Command to execute Python program
python3 ${5} --dataset ${6} --data_tree_depth ${7} -T ${8} --nnarch ${9} --hidden_sizes ${10} \
--optimizer ${11} --nonlin ${12} --lr ${13} --seqtype ${14} --seqlength ${15} --split_length ${16} \
--nbrtest ${17} --data_flips_rate ${18} --shuffle_classes ${19} --blocksz ${20} --verbose 1 \
  

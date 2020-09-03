#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,�~@�

echo "Runtime requested: ${1}"
echo "CPUs per task: ${2}"
echo "Mail user: ${3}"
echo "Memory per CPU: ${4}"
echo "GPU usage: ${5}"
echo "Path to main program: ${6}"
echo "Dataset: ${7}"
echo "Tree depth: ${8}"
echo "Temperature: ${9}"
echo "Network architecture: ${10}"
echo "Hidden size: ${11}"
echo "Optimizer: ${12}"
echo "Nonlinearity: ${13}"
echo "Learning rate: ${14}"
echo "Sequence type: ${15}"
echo "Sequence length: ${16}"
echo "Split length: ${17}"
echo "Number of tests: ${18}"
echo "Data flips rate: ${19}"
echo "Shuffle classes:: ${20}"
echo "Block sizes: ${21}"
echo "Resume simulations: ${22}"

#Command to execute Python program
python3 ${6} --dataset ${7} --data_tree_depth ${8} -T ${9} --nnarch ${10} --hidden_sizes ${11} \
--optimizer ${12} --nonlin ${13} --lr ${14} --seqtype ${15} --seqlength ${16} --split_length ${17} \
--nbrtest ${18} --data_flips_rate ${19} --shuffle_classes ${20} --blocksz ${21} --use_orig ${22} --gpu ${5} --verbose 1 \
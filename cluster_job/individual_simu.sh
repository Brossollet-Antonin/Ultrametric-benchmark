#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="UMRB" # The job name.
#SBATCH --mail-type=START,END,FAIL,TIME_LIMIT_80                  # Mail notification of the events concerning the job : start time, end time,�~@�

echo "Path to main program: ${1}"
echo "Dataset: ${2}"
echo "Tree depth: ${3}"
echo "Temperature: ${4}"
echo "Network architecture: ${5}"
echo "Hidden size: ${6}"
echo "Optimizer: ${7}"
echo "Nonlinearity: ${8}"
echo "Learning rate: ${9}"
echo "Sequence type: ${10}"
echo "Sequence length: ${11}"
echo "Split length: ${12}"
echo "Number of tests: ${13}"
echo "Data flips rate: ${14}"
echo "Shuffle classes: ${15}"
echo "Block sizes: ${16}"
echo "Resume simulations: ${17:-'Original simulation'}"
echo "Verbose level: ${18}"

#Command to execute Python program
python3 ${1} --dataset ${2} --data_tree_depth ${3} -T ${4} --nnarch ${5} --hidden_sizes ${6} \
--optimizer ${7} --nonlin ${8} --lr ${9} --seqtype ${10} --seqlength ${11} --split_length ${12} \
--nbrtest ${13} --data_flips_rate ${14} --shuffle_classes ${15} --blocksz ${16} --use_orig ${17:-""} --verbose {18} \

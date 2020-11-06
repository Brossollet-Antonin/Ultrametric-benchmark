#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
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
echo "Batch size: ${10}"
echo "Sequence type: ${11}"
echo "Sequence length: ${12}"
echo "Split length: ${13}"
echo "Number of tests: ${14}"
echo "Data flips rate: ${15}"
echo "Shuffle classes: ${16}"
echo "Block sizes: ${17}"
echo "Resume simulations: ${18:-'Original simulation'}"
echo "Verbose level: ${19}"

#Command to execute Python program
python3 ${1} --dataset ${2} --data_tree_depth ${3} -T ${4} --nnarch ${5} --hidden_sizes ${6} \
--optimizer ${7} --nonlin ${8} --lr ${9} --batch_sz ${10} --seqtype ${11} --seqlength ${12} --split_length ${13} \
--nbrtest ${14} --data_flips_rate ${15} --shuffle_classes ${16} --blocksz ${17} --use_orig ${18:-""} --verbose ${19} \

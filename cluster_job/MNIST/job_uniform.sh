#!/bin/sh
#
# Script to execute parameter exploration

tree_depth="3"
hidden_size="256"
minibatcheslist="10"
seqtype="uniform"

for value in {1..15}
  do
  sbatch individualjob_uniform.sh ${tree_depth} ${hidden_size} ${seqtype}
  sleep 1
done







	#End of script

#!/bin/sh
#
# Script to execute parameter exploration

tree_depth="5"
hidden_size="20"
minibatcheslist="10"
flip_rates="0.13"
seqtype="uniform"

for flip_rate in $flip_rates
  do  
  for value in {1..15}
    do
    sbatch individualjob_uniform.sh ${tree_depth} ${hidden_size} ${seqtype} ${flip_rate}
    sleep 1
  done
done







	#End of script

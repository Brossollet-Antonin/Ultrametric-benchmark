#!/bin/sh
#
# Script to execute parameter exploration
# Example use:
# bash job_MNIST T_list="0.2 0.225 0.25" n_reps=5 hidden_size=256 block_sizes="1 100 200 500 1000"
# seq_types="ultrametric random_blocks2 uniform" split_length=1000

for ARGUMENT in "$@"
do
  KEY=$(echo $ARGUMENT | cut -f1 -d=)
  VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
  case "$KEY" in
    T_list)               T_list=${VALUE} ;;
    seq_length)           seq_length=${VALUE} ;;
    n_reps)               n_reps=${VALUE} ;;
    hidden_size)          hidden_size=${VALUE} ;;
    block_sizes)          block_sizes=${VALUE} ;;
    seq_types)            seq_types=${VALUE} ;;
    shuffle_labels)       shuffle_labels=${VALUE} ;;
    split_length)         split_length=${VALUE} ;;
    *)   
  esac    
done

tree_depth=3

# IFS=';' read -r -a T_list <<< "$temperatures"
# IFS=';' read -r -a block_sizes <<< "$blocksizes"

for temperature in $T_list
do
  for seqtype in $seq_types
  do
    for (( value = 1; value <= $n_reps; value++ ))
    do
      sbatch individualjobMNIST.sh ${hidden_size} ${seq_length} ${split_length} ${temperature} ${seqtype} ${sl} ${block_sizes[*]}
      sleep 1
    done
  done
done







	#End of script

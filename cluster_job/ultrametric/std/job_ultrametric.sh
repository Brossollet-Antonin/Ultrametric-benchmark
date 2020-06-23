#!/bin/sh
#
# Script to execute parameter exploration
# Example use:
# bash job_ultrametric.sh tree_depth=5 T_list="0.4 0.5 0.6" n_reps=5 hidden_size=20 flip_rates="0.07 0.1 0.13" block_sizes="1 100 200 500 1000"
# seq_types="ultrametric random_blocks2 uniform" split_length=1000

for ARGUMENT in "$@"
do
  KEY=$(echo $ARGUMENT | cut -f1 -d=)
  VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
  case "$KEY" in
    tree_depth)           tree_depth=${VALUE} ;;
    T_list)               T_list=${VALUE} ;;
    n_reps)               n_reps=${VALUE} ;;
    hidden_size)          hidden_size=${VALUE} ;;
    flip_rates)           flip_rates=${VALUE} ;;
    block_sizes)          block_sizes=${VALUE} ;;
    seq_types)            seq_types=${VALUE} ;;
    shuffle_labels)       shuffle_labels=${VALUE} ;;
    seq_length)           seq_length={VALUE} ;;
    nbr_tests)            nbr_tests={VALUE} ;;
    split_length)         split_length=${VALUE} ;;
    *)   
  esac    
done

# IFS=';' read -r -a T_list <<< "$temperatures"
# IFS=';' read -r -a block_sizes <<< "$blocksizes"

for temperature in $T_list
do
  for flip_rate in $flip_rates
  do  
    for seqtype in $seq_types
    do
      for sl in $shuffle_labels
      do
        for (( value = 1; value <= $n_reps; value++ ))
        	do
        	  sbatch individualjob_ultrametric.sh ${tree_depth} ${hidden_size} ${seq_length} ${split_length} ${nbr_tests} ${temperature} ${seqtype} ${sl} ${flip_rate} "${block_sizes[*]}" 
        	  sleep 1
        	done
      done
    done
  done
done

	#End of script

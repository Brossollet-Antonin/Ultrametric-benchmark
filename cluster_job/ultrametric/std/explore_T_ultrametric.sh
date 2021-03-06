#!/bin/sh
#
# Script to execute parameter exploration

for ARGUMENT in "$@"
do
  KEY=$(echo $ARGUMENT | cut -f1 -d=)
  VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
  case "$KEY" in
    tree_depth)           tree_depth=${VALUE} ;;
    T_list)               T_list=${VALUE} ;;
    hidden_size)          hidden_size=${VALUE} ;;
    flip_rate)            flip_rate=${VALUE} ;;
    shuffle_labels)       shuffle_labels=${VALUE} ;;
    *)   
  esac    
done

minibatcheslist="10"
block_size=1
seq_type="ultrametric"
split_length=1000

for temperature in $T_list
do
  for sl in $shuffle_labels
  do
    for value in {1..3}
	  do
	    sbatch individualjob_ultrametric.sh ${tree_depth} ${hidden_size} ${split_length} ${temperature} ${seq_type} ${sl} ${flip_rate} ${block_size} 
	    sleep 1
	  done
  done
done

	#End of script

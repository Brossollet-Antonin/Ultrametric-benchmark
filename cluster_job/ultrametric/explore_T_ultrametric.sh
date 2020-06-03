#!/bin/sh
#
# Script to execute parameter exploration

tree_depth=$1
hidden_size=$2
flip_rate=$3
T_str=$4
IFS=';' read -r -a T_list <<< "$T_str"

minibatcheslist="10"
blocksizearr=(1)
seqtypelist="ultrametric"
shuffle_labels="1"
split_length="328"

for temperature in "${T_list[@]}"
do
  for seqtype in $seqtypelist
  do
    for sl in $shuffle_labels
    do
      for value in {1..3}
	  do
	    sbatch individualjob_ultrametric.sh ${tree_depth} ${hidden_size} ${split_length} ${temperature} ${seqtype} ${sl} ${flip_rate} ${blocksizearr[*]} 
	    sleep 1
	  done
    done
  done
done

	#End of script

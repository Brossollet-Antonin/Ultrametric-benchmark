#!/bin/sh
#
# Script to execute parameter exploration

tree_depth="7"
hidden_size="50"
minibatcheslist="10"
blocksizearr=(1 82 164 328 1312 5248 20992 41984 83968)
temperaturelist="0.6"
flip_rates="0.04 0.07 0.1 0.13"
seqtypelist="random_blocks2 ultrametric"
shuffle_labels="0 1"
split_length="328"

for temperature in $temperaturelist
do
  for flip_rate in $flip_rates
  do
    for seqtype in $seqtypelist
    do
      for sl in $shuffle_labels
      do
        for value in {1..10}
	do
	  sbatch individualjob_ultrametric.sh ${tree_depth} ${hidden_size} ${split_length} ${temperature} ${seqtype} ${sl} ${flip_rate} ${blocksizearr[*]} 
	  sleep 1
	done
      done
    done
  done
done

	#End of script

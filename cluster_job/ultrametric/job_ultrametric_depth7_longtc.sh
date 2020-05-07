#!/bin/sh
#
# Script to execute parameter exploration

tree_depth="7"
hidden_size="50"
minibatcheslist="10"
blocksizearr=(1 20992 41984 83968 167936)
temperaturelist="0.6"
seqtypelist="ultrametric"
shuffle_labels="1 0"
split_length="328"

for temperature in $temperaturelist
do
  for seqtype in $seqtypelist
  do
  	for sl in $shuffle_labels
  	do
	  for value in {1..5}
	  do
	    sbatch individualjob_ultrametric.sh ${tree_depth} ${hidden_size} ${split_length} ${temperature} ${seqtype} ${sl} ${blocksizearr[*]} 
	    sleep 1
	  done
	done
  done
done







	#End of script

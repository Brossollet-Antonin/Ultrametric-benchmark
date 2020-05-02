#!/bin/sh
#
# Script to execute parameter exploration

tree_depth="5"
hidden_size="20"
minibatcheslist="10"
blocksizearr=(1 100 200 500 1000 2000 4000 6000 8000 10000 20000)
temperaturelist="0.4"
seqtypelist="random_blocks2"
shuffle_labels="0"
split_length="1000"

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

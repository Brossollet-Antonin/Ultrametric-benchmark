#!/bin/sh
#
# Script to execute parameter exploration

minibatcheslist="10"
blocksizearr=(1 82 164 328 1312 5248 20992)
temperaturelist="0.6"
seqtypelist="ultrametric random_blocks2"
shuffle_labels="1 0"


for temperature in $temperaturelist
do
  for seqtype in $seqtypelist
  do
  	for sl in shuffle_labels
  	do
	  for value in {1..5}
	  do
	    sbatch individualjob_Antonin.sh ${temperature} ${seqtype} ${sl} ${blocksizearr[*]} 
	    sleep 1
	  done
	done
  done
done







	#End of script
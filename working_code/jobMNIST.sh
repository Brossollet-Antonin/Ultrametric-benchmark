#!/bin/sh
#
# Script to execute parameter exploration



minibatcheslist="10 50 100"
blocksizelist="1 10 50 100 1000 10000 100000"

for minibatchsize in $minibatcheslist
do
	for blocksize in $blocksizelist
	do
		sbatch individualjob.sh ${minibatchsize} ${blocksize}
		sleep 1
		
	 done
done

	#End of script
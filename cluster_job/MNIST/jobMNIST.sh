#!/bin/sh
#
# Script to execute parameter exploration

minibatcheslist="10"
blocksizearr=(1 100 500 1000 10000 100000)
temperaturelist="0.225"
seqtypelist="ultrametric random_blocks2"



for temperature in $temperaturelist
do
  for seqtype in $seqtypelist
  do
    for value in {1..10}
    do
      sbatch individualjobMNIST.sh ${temperature} ${seqtype} ${blocksizearr[*]}
      sleep 1
    done
  done
done







	#End of script

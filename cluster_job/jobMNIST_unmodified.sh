#!/bin/sh
#
# Script to execute parameter exploration

minibatcheslist="10"
blocksizelist="100"
temperaturelist="0.6"


for minibatchsize in $minibatcheslist
do
        for blocksize in $blocksizelist
        do
                for temperature in $temperaturelist
                do
                        for value in {1..5}
                        do
                                sbatch individualjobMNIST_unmodified.sh ${minibatchsize} ${blocksize} ${temperature}
                                sleep 1
                        done
                done

         done
done






	#End of script

#!/bin/sh
#
# Script to execute parameter exploration

seqtypel="temporal_correlation twofold_split onefold_split"



for seqtype in $seqtypel
do
  sbatch individualjobartificial7.sh ${seqtype}
  sleep 1
done

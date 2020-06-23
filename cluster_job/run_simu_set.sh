#!/bin/sh
#
# Script to execute parameter exploration
# Example use:
# bash jobMNIST.sh T_list="0.2 0.225 0.25" n_reps=5 hidden_size=256 block_sizes="1 100 200 500 1000"
# seq_types="ultrametric random_blocks2 uniform" split_length=1000

for ARGUMENT in "$@"
do
  KEY=$(echo $ARGUMENT | cut -f1 -d=)
  VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
  case "$KEY" in
    path)                 path=${VALUE};; # path of main.py 
    T_list)               T_list=${VALUE} ;;
    seq_length)           seq_length=${VALUE} ;;
    n_reps)               n_reps=${VALUE} ;;
    hidden_size)          hidden_size=${VALUE} ;;
    block_sizes)          block_sizes=${VALUE} ;;
    seq_types)            seq_types=${VALUE} ;;
    split_length)         split_length=${VALUE} ;;
    optimizer)            optimizer=${VALUE} ;;
    dataset)              dataset=${VALUE};;
    nbr_test)             nbr_test=${VALUE};;
    time)                 time=${VALUE};; # requiered time to run simulation
    nbr_cpu)              nbr_cpu=${VALUE};; # number of cpu to request on cluster
    mem_per_cpu)          mem_per_cpu=${VALUE};; # memory per cpu
    mail)                 mail=${VALUE};; # email adress to receive notifications
    *)   
  esac    
done

for temperature in $T_list
do
  for seqtype in $seq_types
  do
    for (( value = 1; value <= $n_reps; value++ ))
    do
      sbatch individual_simu.sh -t ${time} -c ${nbr_cpu} --mail-user ${mail} --mem-per-cpu ${mem_per_cpu} \
      ${hidden_size} ${seq_length} ${split_length} ${temperature} ${seqtype} ${optimizer} ${dataset} ${nbr_test} "${block_sizes[*]}" \
      ${path}
      sleep 1
    done
  done
done

	#End of script

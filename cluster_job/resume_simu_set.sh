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
    ## General params
    path)                 path=${VALUE};; # path of main.py
    resume_subfolders)    resume_subfolders=${VALUE};; # path the subfolders from which we will resume simulations

    ## Data + tree params
    dataset)              dataset=${VALUE};;
    tree_depth)           tree_depth=${VALUE};;
    temperature)          temperature=${VALUE} ;; # list of temperatures that will be used in the ultrametric scenario
    flip_rate)            flip_rate=${VALUE} ;; # when learning on artificial dataset, ratio of binary bits flipped at each tree node to generate binary patterns at the leaves
    shuffle_labels)       shuffle_labels=${VALUE} ;; # for artificial dataset, whether or not to shuffle leaves of the tree once patterns are generated

    ## Exemplar params
    seqtype)              seq_type=${VALUE} ;;
    seq_length)           seq_length=${VALUE} ;;
    split_length)         split_length=${VALUE} ;; # used in the random_blocks2 scenario
    nbr_tests)            nbr_tests=${VALUE};; # number of classification accuracy evaluations that will take place when learning on the sequence

    ## Model params
    nnarchi)              nnarchi=${VALUE} ;;
    hidden_sizes)         hidden_sizes=${VALUE} ;;
    optimizer)            optimizer=${VALUE} ;;
    lr)                   lr=${VALUE} ;;
    nonlin)               nonlin=${VALUE} ;;

    ## Shuffle params
    block_sizes)          block_sizes=${VALUE} ;;
    
    ## Slurm job params
    time)                 time=${VALUE};; # requiered time to run simulation
    nbr_cpu)              nbr_cpu=${VALUE};; # number of cpu to request on cluster
    mem)                  mem=${VALUE};; # total memory allocated to requested node
    mail)                 mail=${VALUE};; # email adress to receive notifications
    gpu)                  gpu=1;;  # to run simulations on GPU (can help avoid coredump error) 
    *)   
  esac    
done

# Some parameters MUST be provided, throw an error if not
if [ -z ${path+x} ]; then
  echo "Aborting: path to main.py was not provided"
  exit 1
fi
if [ -z ${resume_subfolders+x} ]; then
  echo "Aborting: path to main.py were not provided"
  exit 1
fi
if [ -z ${hidden_sizes+x} ]; then
  echo "Aborting: hidden size parameter for NN was not provided"
  exit 1
fi
if [ -z ${seq_length+x} ]; then
  echo "Aborting: sequence length was not provided"
  exit 1
fi
if [ -z ${dataset+x} ]; then
  echo "Aborting: dataset was not specified"
  exit 1
fi
if [ -z ${nnarchi+x} ]; then
  echo "Aborting: model architecture was not specified"
  exit 1
fi


# Deal with loop arguments when not provided as kwarg
if [ -z ${tree_depth+x} ]; then
  echo "No tree_depth provided. Using default: 3"
  tree_depth=3;
fi
if [ -z ${optimizer+x} ]; then
  echo "No optimizer specified. Will use adam"
  optimizer="adam";
fi
if [ -z ${nbr_tests+x} ]; then
  echo "No nbr_test specified. Will perform 300 evaluations on test set"
  nbr_tests=300;
fi
if [ -z ${split_length+x} ]; then
  echo "No split_depth provided. For random_blocks2 examplar generation will use blocks of size 1000"
  split_length=1000;
fi
if [ -z ${temperature+x} ]; then
  echo "No temperature provided. Using default: 0.4"
  temperature="0.4";
fi
if [ -z ${seqtype+x} ]; then
  echo "No sequence type provided. Using default: ultrametric"
  seqtype="ultrametric";
fi
if [ -z ${flip_rate+x} ]; then
  echo "No flipping ratio provided. If artificial dataset, using default: 0.1"
  flip_rate="0.1";
fi
if [ -z ${shuffle_labels+x} ]; then
  echo "No shuffle_labels provided. If artificial dataset: will shuffle labels"
  shuffle_labels="1";
fi
if [ -z ${block_sizes+x} ]; then
  echo "No block_sizes provided. Will not perform any shuffle"
  block_sizes="0";
fi
if [ -z ${lr+x} ]; then
  echo "No learning rate provided. Will use LR=0.01 by default"
  lr=0.01;
fi
if [ -z ${nonlin+x} ]; then
  echo "Model will use no nonlinearity"
  nonlin="none";
fi

for subfolder in ${resume_subfolders}
do
  # Resume simulation set using subfolder
  sbatch --time=${time:-"40:00:00"} --cpus-per-task=${nbr_cpu:-2} --mail-user=${mail:-""} --mem=${mem:-"4gb"} ${gpu:+--gres=gpu} \
  individual_simu.sh \
  ${path} ${dataset} ${tree_depth} ${temperature} ${nnarchi} "${hidden_sizes[*]}" ${optimizer} ${nonlin} ${lr} ${seqtype} ${seq_length} ${split_length} ${nbr_tests} ${flip_rate} ${shuffle_labels} "${block_sizes[*]}" ${subfolder}
  sleep 1
done

#End of script

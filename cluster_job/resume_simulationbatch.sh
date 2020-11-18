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
    jobname)              jobname=${VALUE};; # name of job (useful for following jobs in Habanero Slurm queue)
    block_sizes)          block_sizes=${VALUE};; # a space separated list of bloc sizes. Ex: "1 25 75 150 300"
    rb2_root)             rb2_root=${VALUE};; # path to the root of the random_blocks simulations you wish to resume from. Ex: "/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_64/FCL40/random_blocks2_length4000000_batches10_optimsgd_seqlen200_ratio20_splitlength2500"
    ultra_root)           ultra_root=${VALUE};; # path to the root of the ultrametric simulations you wish to resume from. Ex: "/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_64/FCL40/ultrametric_length4000000_batches10_optimsgd_seqlen200_ratio20"
    path)                 path=${VALUE};; # path of main.py
    verbose)              verbose=${VALUE};; # 0 for no simu verbose, 1 for synthetic verbose, 2 for extensive

    ## Data + tree params
    dataset)              dataset=${VALUE};;
    tree_depth)           tree_depth=${VALUE};;
    temperature)          temperature=${VALUE};; # list of temperatures that will be used in the ultrametric scenario
    flip_rate)            flip_rate=${VALUE};; # when learning on artificial dataset, ratio of binary bits flipped at each tree node to generate binary patterns at the leaves
    shuffle_labels)       shuffle_labels=${VALUE} ;; # for artificial dataset, whether or not to shuffle leaves of the tree once patterns are generated

    ## Exemplar params
    seq_length)           seq_length=${VALUE} ;;
    split_length)         split_length=${VALUE} ;; # used in the random_blocks2 scenario
    nbr_tests)            nbr_tests=${VALUE};; # number of classification accuracy evaluations that will take place when learning on the sequence

    ## Model params
    nnarchi)              nnarchi=${VALUE} ;;
    hidden_sizes)         hidden_sizes=${VALUE} ;;
    optimizer)            optimizer=${VALUE} ;;
    lr)                   lr=${VALUE} ;;
    batch_sz)             batch_sz=${VALUE} ;;
    nonlin)               nonlin=${VALUE} ;;
    
    ## Slurm job params
    time)                 time=${VALUE};; # requiered time to run simulation
    nbr_cpu)              nbr_cpu=${VALUE};; # number of cpu to request on cluster
    mem)                  mem=${VALUE};; # total memory allocated to requested node
    mail)                 mail=${VALUE};; # email adress to receive notifications
    gpu)                  gpu=1;;  # to run simulations on GPU (can help avoid coredump error)  
  esac    
done

# Some parameters MUST be provided, throw an error if not
if [ -z ${block_sizes+x} ]; then
  echo "Aborting: block_sizes was not provided"
  exit 1
fi
if [ -z ${path+x} ]; then
  echo "Aborting: path to main.py was not provided"
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
if [ -z ${jobname+x} ]; then
  echo "No job name provided. Using default: CL_benchmark"
  jobname="CL_benchmark";
fi
if [ -z ${verbose+x} ]; then
  echo "No verbose provided. Using default: 1"
  verbose=1;
fi
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
if [ -z ${flip_rate+x} ]; then
  echo "No flipping ratio provided. If artificial dataset, using default: 0.1"
  flip_rate="0.1";
fi
if [ -z ${shuffle_labels+x} ]; then
  echo "No shuffle_labels provided. If artificial dataset: will shuffle labels"
  shuffle_labels="1";
fi
if [ -z ${lr+x} ]; then
  echo "No learning rate provided. Will use LR=0.01 by default"
  lr=0.01;
fi
if [ -z ${batch_sz+x} ]; then
  echo "No natch size provided. Will use batch_sz=10 by default"
  batch_sz=10;
fi
if [ -z ${nonlin+x} ]; then
  echo "Model will use no nonlinearity"
  nonlin="none";
fi
if [ -z ${rb2_root+x} ]; then
  echo "Argument rb2_root was not provided. Will not resume simulations for random blocks simulations."
fi
if [ -z ${ultra_root+x} ]; then
  echo "Argument ultra_root was not provided. Will not resume simulations for ultrametric simulations."
fi

if [ ! -z ${ultra_root+x} ] ; then
  readarray ultra_subfolders < <(find ${ultra_root} -maxdepth 1 -mindepth 1 -type d)
fi
if [ ! -z ${rb2_root+x} ] ; then
  readarray rb2_subfolders < <(find ${rb2_root} -maxdepth 1 -mindepth 1 -type d)
fi

for sim_id in "${!ultra_subfolders[@]}"
do
	for bs in ${block_sizes}
	do
    if [ ! -z ${rb2_root+x} ] ; then
      bash resume_simu_set.sh jobname="${jobname}_s${bs}_rb2" block_sizes=${bs} seqtype="random_blocks2" path=${path} resume_subfolders=${rb2_subfolders[$sim_id]} dataset=${dataset} tree_depth=${tree_depth} temperature=${temperature} flip_rate=${flip_rate} shuffle_labels=${shuffle_labels} seq_length=${seq_length} split_length=${split_length} nbr_tests=${nbr_tests} nnarchi=${nnarchi} hidden_sizes=${hidden_sizes} optimizer=${optimizer} lr=${lr} batch_sz=${batch_sz} nonlin=${nonlin} time=${time} nbr_cpu=${nbr_cpu} mem=${mem} mail=${mail}
    fi
    if [ ! -z ${ultra_root+x} ] ; then
      bash resume_simu_set.sh jobname="${jobname}_s${bs}_ultra" block_sizes=${bs} seqtype="ultrametric" path=${path} resume_subfolders=${ultra_subfolders[$sim_id]} dataset=${dataset} tree_depth=${tree_depth} temperature=${temperature} flip_rate=${flip_rate} shuffle_labels=${shuffle_labels} seq_length=${seq_length} split_length=${split_length} nbr_tests=${nbr_tests} nnarchi=${nnarchi} hidden_sizes=${hidden_sizes} optimizer=${optimizer} lr=${lr} batch_sz=${batch_sz} nonlin=${nonlin} time=${time} nbr_cpu=${nbr_cpu} mem=${mem} mail=${mail}
    fi
	done
done

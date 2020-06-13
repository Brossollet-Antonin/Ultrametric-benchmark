#!/bin/sh

#SBATCH --job-name=run_multithread_ultrametric               # Job Name
#SBATCH --ntasks=3                       # Number of Tasks : 3
#SBATCH --cpus-per-task=4                # 4 CPU allocation per Task
#SBATCH --partition=8CPUNodes            # Name of the Slurm partition used

#SBATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

tree_depth=5
T=$1
hidden_width=$2
bs=$3
lr=$4
split_sizes=$5
nruns_per_splitsize=3

IFS=';' read -r -a splsz_list <<< "$split_sizes"

for split in $splsz_list
do
	for (( run_id=1; run_id <= $nruns_per_splitsize; run_id++ ))
	do
		srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 200000 --seqtype random_blocks2 --split_length ${split} --nbrtest 200 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} --blocksz 1" &
	done
done

wait

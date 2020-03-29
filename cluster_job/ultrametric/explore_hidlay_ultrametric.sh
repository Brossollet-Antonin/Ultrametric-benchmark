#!/bin/sh

#SBATCH --job-name=run_multithread_ultrametric               # Job Name
#SBATCH --nodes=2                        # Number of Nodes : 2
#SBATCH --cpus-per-task=1                # 4 CPU allocation per Task
#SBATCH --mem-per-cpu=8g

#SBATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

tree_depth=5
T=$1
hidden_widths=$2
batch_sizes=$3
learning_rates=$4
nruns_perwidth=2

IFS=';' read -r -a hw_list <<< "$hidden_widths"
IFS=';' read -r -a bs_list <<< "$batch_sizes"
IFS=';' read -r -a lr_list <<< "$learning_rates"

for w in "${hw_list[@]}"
do
	for bs in "${bs_list[@]}"
	do
		for lr in "${lr_list[@]}"
		do
			for ((run_id=1; run_id<=$nruns_perwidth; run_id++))
			do
			srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 200000 --seqtype ultrametric --nbrtest 200 --nnarchi FCL --hidden_sizes ${w} -T ${T} --lr ${lr} --minibatch ${bs}" &
			done
		done
	done
done
wait

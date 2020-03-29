#!/bin/sh

#SBATCH --job-name=explore_T_ultrametric # Job Name
#SBATCH --nodes=2                        # Number of Nodes : 2
#SBATCH --cpus-per-task=1                # 1 CPU allocation per Task
#SBATCH --mem-per-cpu=8g

#SBATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

tree_depth=5
T_str=$1
hidden_width=20
nruns=4

IFS=';' read -r -a T_list <<< "$T_str"

for T in "${T_list[@]}"
do
	srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 200000 --seqtype ultrametric --nbrtest 200 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} " &
done

wait

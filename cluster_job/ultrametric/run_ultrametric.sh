#!/bin/sh

#SBATCH --job-name=run_multithread_ultrametric               # Job Name
#SBATCH --ntasks=8                       # Number of Tasks : 2*4=8
#SBATCH --cpus-per-task=1                # 1 CPU allocation per Task
#SBATCH --mem_per-cpu=8g

#SBATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

tree_depth=5
T=0.4
split_length=1000
hidden_width=20
nruns_rb=4
nruns_ultra=4
blocksize_list="1 100 200 500 1000 2000 4000 6000 8000 10000 20000"

for ((run_id=1; run_id<=$nruns_rb; run_id++))
do
	srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 200000 --seqtype random_blocks2 --split_length ${split_length} --nbrtest 200 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} --blocksz ${blocksize_list}" &
done

for ((run_id=1; run_id<=$nruns_ultra; run_id++))
do
	srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 200000 --seqtype ultrametric --nbrtest 200 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} --blocksz ${blocksize_list}" &
done

wait

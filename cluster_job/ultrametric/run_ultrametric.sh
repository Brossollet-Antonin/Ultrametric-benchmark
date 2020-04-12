#!/bin/sh

#SBATCH --job-name=run_multithread_ultrametric               # Job Name
#SBATCH --ntasks=8                       # Number of Tasks : 2*4=8
#SBATCH --cpus-per-task=2                # 2 CPUs allocation per Task
#SBATCH --mem_per-cpu=8g

#SBATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,…
#SBATCH --mail-user=lebastardsimon@gmail.com

tree_depth=7
T=0.6
split_length=328
hidden_width=50
nruns_rb=4
nruns_ultra=4
blocksize_list="1 82 164 328 1312 5248 20992"

for ((run_id=1; run_id<=$nruns_rb; run_id++))
do
	srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 300000 --seqtype random_blocks2 --split_length ${split_length} --nbrtest 300 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} --blocksz ${blocksize_list}" &
done

for ((run_id=1; run_id<=$nruns_ultra; run_id++))
do
	srun -n1 -N1 "python3 main.py -v --dataset artificial --data_tree_depth ${tree_depth} --data_seq_size 200 --seqlength 300000 --seqtype ultrametric --nbrtest 300 --nnarchi FCL --hidden_sizes ${hidden_width} -T ${T} --blocksz ${blocksize_list}" &
done

wait

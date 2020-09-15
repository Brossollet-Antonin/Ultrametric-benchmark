block_sizes="1 500 1000 2500 5000 10000 20000 40000 80000 160000 320000 640000 1280000"
rb2_root="/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_64/FCL40/random_blocks2_length4000000_batches10_optimsgd_seqlen200_ratio20_splitlength2500"
ultra_root="/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_64/FCL40/ultrametric_length4000000_batches10_optimsgd_seqlen200_ratio20"

for bs in $block_sizes
do
	for resume_subdir in $(find ${rb2_root} -maxdepth 1 -mindepth 1 -type d)
	do
		bash resume_simu_set.sh jobname="CLUM_artifd6_s${bs}_rb2" block_sizes=${bs} seqtype="random_blocks2" path=/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py resume_subfolders=${resume_subdir} dataset=artificial tree_depth=6 temperature="0.4" flip_rate="0.1" shuffle_labels="1" seq_length=4000000 split_length=2500 nbr_tests=800 nnarchi=FCL hidden_sizes=40 optimizer=sgd lr=0.001 nonlin=celu time="5-0" nbr_cpu=2 mem=32g mail=lebastardsimon@gmail.com
	done
	for resume_subdir in $(find ${ultra_root} -maxdepth 1 -mindepth 1 -type d)
	do
		bash resume_simu_set.sh jobname="CLUM_artifd6_s${bs}_ultra" block_sizes=${bs} seqtype="ultrametric" path=/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py resume_subfolders=${resume_subdir} dataset=artificial tree_depth=6 temperature="0.4" flip_rate="0.1" shuffle_labels="1" seq_length=4000000 split_length=2500 nbr_tests=800 nnarchi=FCL hidden_sizes=40 optimizer=sgd lr=0.001 nonlin=celu time="5-0" nbr_cpu=2 mem=32g mail=lebastardsimon@gmail.com
	done
done
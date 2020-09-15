block_sizes="1 25 75 150 300 600 1200 2400 4800 9600 19200 38400"
rb2_root="/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_16/FCL10/random_blocks2_length60000_batches10_optimsgd_seqlen200_ratio20_splitlength150"
ultra_root="/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/Results/1toM/artificial_16/FCL10/ultrametric_length60000_batches10_optimsgd_seqlen200_ratio20"

for bs in $block_sizes
do
	for resume_subdir in $(find ${rb2_root} -maxdepth 1 -mindepth 1 -type d)
	do
		bash resume_simu_set.sh jobname="CLUM_artifd4_s${bs}_rb2" block_sizes=${bs} seqtype="random_blocks2" path=/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py resume_subfolders=${resume_subdir} dataset=artificial tree_depth=4 temperature="0.4" flip_rate="0.1" shuffle_labels="1" seq_length=60000 split_length=150 nbr_tests=240 nnarchi=FCL hidden_sizes=10 optimizer=sgd lr=0.001 nonlin=celu time="28:00:00" nbr_cpu=2 mem=32g mail=lebastardsimon@gmail.com
	done
	for resume_subdir in $(find ${ultra_root} -maxdepth 1 -mindepth 1 -type d)
	do
		bash resume_simu_set.sh jobname="CLUM_artifd4_s${bs}_ultra" block_sizes=${bs} seqtype="ultrametric" path=/rigel/theory/users/sl4744/projects/Ultrametric-benchmark/main.py resume_subfolders=${resume_subdir} dataset=artificial tree_depth=4 temperature="0.4" flip_rate="0.1" shuffle_labels="1" seq_length=60000 split_length=150 nbr_tests=240 nnarchi=FCL hidden_sizes=10 optimizer=sgd lr=0.001 nonlin=celu time="28:00:00" nbr_cpu=2 mem=32g mail=lebastardsimon@gmail.com
	done
done
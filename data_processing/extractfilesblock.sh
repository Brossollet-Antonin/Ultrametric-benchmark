#! /bin/bash


date=2019-07-02
dataset=CIFAR10
test=length2000000_batches10
ind=0
temp='0.150 0.250'
blocks='1  10 100 1000 10000 100000'

for T in $temp
do
	for block in $blocks
	do 
		ind=0
		for folder in ./$date/Blocks/$dataset/$test/T$T*block$block\ *
		do
			echo $folder
			savefolder="./Blocks/$dataset/block$block T$T"
			echo $savefolder
			mkdir -p "$savefolder/$ind"
			cp "$folder/var_original_accuracy.npy" "$savefolder/$ind/"
			cp "$folder/var_shuffle_accuracy.npy" "$savefolder/$ind/"
			cp "$folder/var_original_classes_prediction.npy" "$savefolder/$ind/"
			cp "$folder/var_shuffle_classes_prediction.npy" "$savefolder/$ind/"
			cp "$folder/parameters.npy" "$savefolder/$ind/"
			cp "$folder/original" "$savefolder/$ind"
			cp "$folder/shuffle" "$savefolder/$ind"
			((ind++))
		done
	done
done

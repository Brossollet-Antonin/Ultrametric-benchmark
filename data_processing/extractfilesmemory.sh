#! /bin/bash


date=2019-07-16
dataset=MNIST
test=length10000000_batches10_lr0.0100
ind=0
temp='0.150 0.130 0.140'
memory='0 8 80 400 800 4000 8000'

for T in $temp
do
	for mem in $memory
	do 
		ind=0
		for folder in ./$date/Memory/$dataset/$test/T$T\ Memory${mem}\ *
		do
			echo $folder
			savefolder="./Memory/$dataset/Memory$mem T$T"
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

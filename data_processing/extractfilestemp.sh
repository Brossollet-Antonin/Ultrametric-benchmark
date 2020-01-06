#! /bin/bash


date=2019-07-10
dataset=MNIST
test=length10000000_batches10
ind=0
temp='0.110 0.130 0.900'

for T in $temp
do
	ind=0
	for folder in ./$date/Temperature/$dataset/$test/T$T*
	do
		echo $folder
		savefolder=./Temperature/$dataset/$date/T$T/$ind
		mkdir -p $savefolder
		cp "$folder/var_original_accuracy.npy" $savefolder
		cp "$folder/var_shuffle_accuracy.npy" $savefolder
		cp "$folder/var_original_classes_prediction.npy" $savefolder
		cp "$folder/var_shuffle_classes_prediction.npy" $savefolder
		cp "$folder/parameters.npy" $savefolder
		cp "$folder/original" $savefolder
		cp "$folder/shuffle" $savefolder
		((ind++))
	done
done

#! /bin/bash


date=2019-07-15
dataset=MNIST
test=length10000000_batches
ind=0
temp='0.150 0.130'
learningrate='0.0010 0.0100 0.0500 0.1000 0.5000'

for lr in $learningrate
do
	for T in $temp
	do
		ind=0
		for folder in ./$date/LR/$dataset/${test}*${lr}/T$T*
		do
			echo $folder
			savefolder=./LR/$dataset/$date/T$T/LR$lr/$ind
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
done

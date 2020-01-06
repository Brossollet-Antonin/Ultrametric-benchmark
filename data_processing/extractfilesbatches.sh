#! /bin/bash


date=2019-07-11
dataset=MNIST
test=length10000000_batches
ind=0
temp='0.150 0.130'
minibatches='10 50 100 1000 10000'

for batch in $minibatches
do
	for T in $temp
	do
		ind=0
		for folder in ./$date/Minibatches/$dataset/${test}${batch}/T$T*
		do
			echo $folder
			savefolder=./Minibatches/$dataset/$date/T$T/minibatch$batch/$ind
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

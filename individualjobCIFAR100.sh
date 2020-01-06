#!/bin/sh

# To run each individual job inside the loop

#SBATCH --account=theory # The account name for the job.
#SBATCH --job-name="MNIST $minibatchsize $blocksize" # The job name.
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=8:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.
 
module load anaconda
 
#Command to execute Python program
python main.py --dataset CIFAR100 --nnarch ResNet --resnettype 50 --minibatch ${1} --blocksz ${2} --T ${3} 
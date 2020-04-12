# Ultrametric-benchmark
This project contain the code for exploring the ultrametric benchmark for continual analysis. 

## Getting started 
The main.py file is used to run the tests. Numerous options can be passed as parameters (see main.py --help for more informations) such as the architecture of the networks, the dataset used, the ultrametric sequence parameters (lenght, temperature of the random walk...), the shuffle parameters... The code produce files containing the accuracy tested at different points of the training process to monitor the learning process. 

## Results analysis
The results analysis is not automatically done for the moment. Some codes to do some of the plots already produced are located in the [Data analysis](data_analysis) folder. Several notebooks for data analisys can be found in the [notebooks](notebooks) folder.
## Run jobs on Habanero
The [cluster job](cluster_job) folder contain scripts to run batch jobs on the theory centre cluster Habanero to train batches on neural networks using different parameters to produce more statistically significant data and explore different parameters. 
Infos on how to use Habanero can be found [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+HPC+Cluster+User+Documentation).

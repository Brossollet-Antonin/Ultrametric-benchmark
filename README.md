# A task-agnostic benchmark for realistic continual learning
Here we introduce a new task-agnostic benchmark for continual learning that uses an ultrametric space to generate training data with a much richer and more realistic temporal profile. We argue that the multi-scale temporal structure of the learning sequence can have a dramatic influence on catastrophic forgetting, and propose a methodology to measure the influence of each timescale on known metrics for catastrophic interference.

## Concepts & implementation
![Conceptual model for continual learning selected in our framework](./resources/online_summary/UM_ConceptualModel)
![Our current implementation of the conceptual framework](./resources/online_summary/UM_Implementation) 

## Getting started
The main.py file is used to run a set of simulations. Each simulation consist of:
- a task/dataset
- a neural network model

Simulation consists in:
- generating an *original* learning sequence consisting of label examples of a given length, corresponding to the task at hand. Let's say we train a CNN model on MNIST data, on a sequence of 1M examples, defined as: 0110101233233222301011110101477555566656565774... this sequence goes on until the one million'th digit.
- training the neural network, from scratch, on the original sequence, and testing the classification accuracy on a test set every *k* training iterations. Note that training time is linear.
- training the neural network, from scratch, on a set of shuffled sequences, each corresponding to deleting temporal correlations past a given threshold. Note that training time is quadratic, or to be exact linear in the sequence length and linear in the number of evaluations (so quadratic at constant evaluation resolution).
To run a set of simulation, use the following command:
``python3 <PATH_TO_main.py> --dataset <artificial/MNIST/CIFAR10/CIFAR100> --data_tree_depth <arg1> -T <arg2> --nnarchi <FCL/CNN/ResNet> --hidden_sizes <arg3> \
--optimizer <sgd/adam/adagrad> --nonlin <none/relu> --lr <arg4> --seqtype <uniform/random_blocks2/ultrametric> --seqlength <arg5> --split_length <arg6> \
--nbrtest <arg7> --data_flips_rate <arg8> --shuffle_classes <0/1> --blocksz <arg9> --verbose 1``
Numerous options can be passed as parameters such as the architecture of the networks, the dataset used, the ultrametric sequence parameters (lenght, temperature of the random walk...), the shuffle parameters...

The code produces files containing:
- train_labels_orig.pickle the original sequence that our model was trained on.
- for each shuffle size that was specified, train_labels_shfl.pickle contains an example of shuffled sequence generated from the original sequence. In practice, this is the last shuffle that was applied to the original sequence in order to train the model, at the very end of training.
- var_original_accuracy.npy contains a list of test scores, one per evaluation during learning.
- for each shuffle size that was specified, var_shuffle_accuracy.npy contains the contains the list of test scores, one per evaluation.
- classes_template.npy provides, in the case of the artificial dataset, a template of each class that was generated to construct the original dataset. Each example is then a noisy version of the template for the corresponding class
- distribution.json contains a comprehensive list of the parameters used for the simulations.

## Results analysis
We are currently working at automating the visual analysis of the results.
You should soon be able to simply run a command that will generate the main plots from our article, on your own dataset.
Some codes to do some of the plots already produced are located in the [Data analysis](data_analysis) folder. Several notebooks for data analisys can be found in the [notebooks](notebooks) folder.

## Run jobs on Habanero
The [cluster job](cluster_job) folder contain scripts to run batch jobs on the Habanero cluster at Columbia University.
Infos on how to use Habanero can be found [here](https://confluence.columbia.edu/confluence/display/rcs/Habanero+HPC+Cluster+User+Documentation).

# GSE
This repository contains the code for the paper "Group-wise Sparse and Explainable Adversarial Attacks" (https://arxiv.org/abs/2311.17434).

## Setup
Dependencies: `numpy`, `torch`, `torchvision`, `natsort`, `pandas`, `matplotlib`, `skimage`
  
The NIPS2017 data set can be found at https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/data.
Since the .pt files containing the weights for the ResNet50 CAM and the WideResNet are over the limit of 100MB, they are not included in this repository.

## Run main
The `main.py` file contains the code for the main experiments. It can split the data set into chunks for 'embarrassingly parallel' execution.
For example to run a targeted test for GSE and a ResNet20 on images 1000-1999 of the CIFAR10 test set with a batch size of 500, execute
  `python main.py --dataset 'CIFAR10' --model 'ResNet20' --numchunks 10 --chunk 1 --batchsize 500 --attack 'GSE' --targeted 1`

When all experiments are finished, execute `process_results.py` to combine the results corresponding to the same experiment.

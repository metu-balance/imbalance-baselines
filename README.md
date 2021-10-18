# imbalance-baselines

This is a Python library providing:
* Tools for training ResNets on imbalanced datasets
* Sampling methods to be used in imbalanced learning
* Loss functions to be used during training and utilities for the calculation of the required hyperparameters

The functions use 64-bit floating number (double) precision.

## Table of Contents
1. [Library Contents](#library-contents)
   1. [sampling.py](#samplingpy)
   2. [datasets.py](#datasetspy)
   3. [models.py](#modelspy)
   4. [loss_functions.py](#loss_functionspy)
   5. [training.py](#trainingpy)
   6. [utils.py](#utilspy)
2. [Example Usage](#example-usage)
3. [Our Results](#our-results)

## Library Contents 

### sampling.py

This file contains sampling mehods to be used in imbalanced training, in order to overcome the disadvantages of
class imbalance. 

The methods are divided into two as "offline" and "online" methods.

#### Offline methods

These methods apply the re-balancing once when the dataset object is being created.

##### OverSampler

This method eliminates the imbalance by random over-sampling: Images from under-reprsented classes are randomly
replicated until the classes are balanced. 

##### UnderSampler

This method eliminates the imbalance by random under-sampling: Images from over-reprsented classes are randomly removed
until the classes are balanced.

#### Online methods

These methods eliminate the imbalance over time, as the images are being picked from the dataset.

##### ClassBalancedSampling

In this sampling method, images from each class are selected with equal probabilities. The sampling is done according to
the method given in the paper [Decoupling Representation and Classifier for Long-Tailed Recognition](https://www.semanticscholar.org/paper/Decoupling-Representation-and-Classifier-for-Kang-Xie/c6ecdf34ab566efb06bd05c4f1bc9bda218f7dc9).
First, a class is uniformly sampled, and then, an instance from the chosen class is uniformly sampled.

##### ProgressivelyBalancedSampling

In this sampling method, the sampling probabilities progressively change from imbalanced to balanced, according to the
method given in the paper [Decoupling Representation and Classifier for Long-Tailed Recognition](https://www.semanticscholar.org/paper/Decoupling-Representation-and-Classifier-for-Kang-Xie/c6ecdf34ab566efb06bd05c4f1bc9bda218f7dc9).

### datasets.py

This file contains the available imbalanced datasets to work on:

#### CIFAR10_LT
This class provides a customized version of the CIFAR-10 dataset to be long-tailed. The desired offline sampler can be passed as an
argument.

If no sampler is specified, the sampling method specified in the paper [TODO](TODO) is applied:
The training samples are reduced according to the exponential function TODO.

No modifications are done on the test set.

#### INaturalist

This class provides the iNaturalist 2017 and 2018 datasets. The desired year's dataset can be chosen with an argument.

These datasets are imbalanced by their own, so the class does not support applying additional offline sampling methods. 

#### generate_data

This function ... TODO ...

### models.py

This file contains the custom ResNet definitions to be used in training.

#### ResNet32

This class derives from `torch.nn.Module` and implements a custom ResNet with 32 layers. 

### training.py

This file contains the tools and functions to be used during the training of the models.

#### train_models

TODO

### loss_functions.py

TODO

### utils.py

This file contains the various utilites to be used in testing or hyperparameter calculation.

#### get_weights

TODO

#### get_accuracy

This function takes a trained model and a test dataset as input and outputs the resulting accuracy. The "top-n accuracy"
statistic can also be specified.

---

## Example Usage

TODO

---

## Our Results

TODO

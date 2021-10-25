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

---

### datasets.py

This file contains the available imbalanced datasets to work on:

#### CIFAR10_LT
This class provides a customized version of the CIFAR-10 dataset to be long-tailed. The desired offline sampler can be passed as an
argument.

If no sampler is specified, the sampling method specified in the paper [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) is applied:
The training samples are reduced according to an exponential function.

No modifications are done on the test set.

#### INaturalist

This class provides the iNaturalist 2017 and 2018 datasets. The desired year's dataset can be chosen with an argument.

These datasets are imbalanced by their own, so the class does not support applying additional offline sampling methods. 

#### generate_data

This function prepares the desired dataset's objects along with variables holding related information to be used in training and
testing.

---

### models.py

This file contains the custom ResNet definitions to be used in training. Currently, only a custom ResNet-32 model is
defined.

#### ResNet32

This class derives from `torch.nn.Module` and implements a custom ResNet with 32 layers. 

---

### training.py

This file contains the tools and functions to be used during the training of the models.

#### train_models

This function defines and trains the desired ResNet models with the desired loss functions, using a dataset's `DataLoader` object. 

---

### loss_functions.py

This function contains loss function implementations to be used during training.

---

### utils.py

This file contains the various utilites to be used in testing or hyperparameter calculation.

#### get_weights

This function returns the list of weights per each class of a dataset to be used in class-balanced loss functions. The
implementation is as specified in the paper [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555).

#### get_accuracy

This function takes a trained model and a test dataset as input and outputs the resulting accuracy. The "top-n accuracy"
statistic can also be specified.

---

## Example Usage

Below is an example that trains two models for 200 epochs using the imbalanced version of the CIFAR-10 dataset,
class-balanced and non-class-balanced focal losses, and the custom ResNet-32 implementation. The training progress is
set to be printed at every 10 batches. Later, the models' accuracies are tested.

```python
from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import utils

beta = 0.9999

train_dl, test_dl, class_cnt, train_class_sizes, test_class_sizes = datasets.generate_data(
    batch_size=128,
    dataset="IMB_CIFAR10",
    datasets_path="test_files/datasets/",
    cifar_imb_factor=100
)

weights = utils.get_weights(train_class_sizes, beta)
weights.requires_grad = False

print("Got weights:", weights)

models = training.train_models("IMB_CIFAR10", train_dl, class_cnt, weights,
                               epoch_cnt=200, print_freq=10, train_focal=True, train_cb_focal=True)

model_focal = models[0]
model_cb_focal = models[3]

focal_avg_acc, focal_per_class_acc = utils.get_accuracy(test_dl, model_focal, test_class_sizes)
cb_focal_avg_acc, cb_focal_per_class_acc = utils.get_accuracy(test_dl, model_cb_focal, test_class_sizes)

print("Focal Loss:")
print("Average accuracy:", focal_avg_acc)
print("Accuracy per class:", focal_per_class_acc)

print("Class-Balanced Focal Loss:")
print("Average accuracy:", cb_focal_avg_acc)
print("Accuracy per class:", cb_focal_per_class_acc)

print("Done!")
```

Output:

```
TODO
```
---

TODO: Add a sampling method usage example

## Our Results

TODO

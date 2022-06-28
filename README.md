---
# Readme TODO
This readme needs rework!
* Explain setup & requirements
* Explain directory structure & how to make additions to each file or method
    * The directory structure itself may need some organization - should we use subfolders?
    * Each added dataset/loss/model/... must have ... fields...
      * e.g. Each model needs num_classes field to determine output size
    * For each added dataset/loss/model/..., ... must be added in config.py, dataset.py/loss.py/(...).py,
  \_\_init__.py... 
* Implement config. class & explain usage (list available loss, model... choices etc., explain config writing guide for
new datasets & other components)
---

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
import torch.cuda
from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import utils

beta = 0.9999
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dl, test_dl, class_cnt, train_class_sizes, test_class_sizes = datasets.generate_data(
    batch_size=128,
    dataset="IMB_CIFAR10",
    datasets_path="datasets/",
    cifar_imb_factor=100
)

weights = utils.get_weights(train_class_sizes, beta, device=device)
weights.requires_grad = False

print("Got weights:", weights)

models = training.train_models("IMB_CIFAR10", train_dl, class_cnt, weights, device=device, epoch_cnt=1,
                               print_freq=10, train_focal=True, train_cb_focal=True)

model_focal = models[0]
model_cb_focal = models[3]

focal_avg_acc, focal_per_class_acc = utils.get_accuracy(test_dl, model_focal, test_class_sizes, device=device)
cb_focal_avg_acc, cb_focal_per_class_acc = utils.get_accuracy(test_dl, model_cb_focal, test_class_sizes, device=device)

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
Files already downloaded and verified
Files already downloaded and verified
Number of training samples:
[5000 2997 1796 1077  645  387  232  139   83   50]
Got weights: tensor([0.0506, 0.0769, 0.1211, 0.1950, 0.3188, 0.5246, 0.8684, 1.4426, 2.4092,
        3.9927], device='cuda:0', dtype=torch.float64)
Starting training with Long-Tailed CIFAR10 dataset, ResNet-32 models.
Epoch: 0 | Batch: 1
     Focal: 2.581519189487201
  CB Focal: 0.7345769421845707

Epoch: 0 | Batch: 50
     Focal: 1.6703934794157314
  CB Focal: 0.3952958685285224

Epoch: 0 | Batch: 97
     Focal: 1.5652494204323226
  CB Focal: 0.38759862578652887

Epoch: 1 | Batch: 50
     Focal: 1.4629814085467638
  CB Focal: 0.3902064873347718
```
...
```
Epoch: 198 | Batch: 97
     Focal: 0.3481788270618167
  CB Focal: 0.13591878579199754

Epoch: 199 | Batch: 50
     Focal: 0.3476705417281534
  CB Focal: 0.13620742757213233

Epoch: 199 | Batch: 97
     Focal: 0.34149305391253787
  CB Focal: 0.13366841483947772

Saved model (ResNet-32 focal, Long-Tailed CIFAR10): ./output/models/rn32_focal_IMB_CIFAR10.pth
Saved model (ResNet-32 cb. focal, Long-Tailed CIFAR10): ./output/models/rn32_cb_focal_IMB_CIFAR10.pth
Focal Loss:
Average accuracy: tensor(0.5217, device='cuda:0')
Accuracy per class: [0.9860000610351562, 0.9520000219345093, 0.76500004529953, 0.6260000467300415, 0.6110000014305115, 0.34800001978874207, 0.43700000643730164, 0.26600000262260437, 0.15600000321865082, 0.07000000029802322]
Class-Balanced Focal Loss:
Average accuracy: tensor(0.5882, device='cuda:0')
Accuracy per class: [0.956000030040741, 0.9000000357627869, 0.6340000033378601, 0.5120000243186951, 0.609000027179718, 0.4540000259876251, 0.550000011920929, 0.45500001311302185, 0.4660000205039978, 0.3460000157356262]
Done!
```
---

TODO: Add a sampling method usage example

## Our Results

TODO

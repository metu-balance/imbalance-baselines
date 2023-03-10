# TODO: OUTDATED README, TO BE REWRITTEN!

# imbalance-baselines

**imbalance-baselines** is a modular Python deep learning library, providing a common measure for the performance of
different imbalance mitigation methods.  

## Table of Contents
1. [About imbalance-baselines](#about-imbalance-baselines)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [Configuration Files](#configuration-files)
   2. [Examples](#examples)
5. [Directory Structure](#directory-structure)
6. [Implemented Components & Imbalance Mitigation Methods](#implemented-components--imbalance-mitigation-methods)
7. [Adding New Components](#adding-new-components)
   1. [Adding a new dataset](#adding-a-new-dataset)
   2. [Adding a new model (backbone)](#adding-a-new-model--backbone-)
   3. [Adding a new optimizer](#adding-a-new-optimizer)
   4. [Adding a new evaluation method](#adding-a-new-evaluation-method)
   5. [Adding a new loss function](#adding-a-new-loss-function)
   6. [Adding a new sampling method](#adding-a-new-sampling-method)
8. [Features Considered for Addition](#features-considered-for-addition)
9. [License](#license)
10. [Credits & Contact](#credits--contact)

## About imbalance-baselines

Class imbalance problem is a profound and active topic in deep learning-based image
recognition. As new methods are developed to overcome this, a need emerges to efficiently
implement and compare them against each other.

`imbalance_baselines` aims to implement various data imbalance mitigation methods in a modular library to
allow efficient usage and additions, and to provide a common ground for evaluation and comparison of such methods.

The project began development as a workbench for comparing the results of class-balancing methods. Later, the
code was repurposed into a more general library to make adaptations to other methods easier, finally resulting in
the current structure.

Though it was not an initial aim, the library also provides code for training models. The training code is designed
so that multiple models are trained together in a single iteration loop over the dataset.

## Requirements

The library is developed and tested using Python 3.10.

Used third-party libraries are:
* matplotlib (3.5.1)
* numpy (1.22)
* omegaconf (2.3.0)
* pillow (9.2.0)
* torch (1.12)
* yaml (0.2.5)

Our Conda environment is also available under the file `environment.yaml`.

## Installation

First, install the appropriate version of PyTorch, preferably using a command from
[their website](https://pytorch.org/).

Then, install the libraries that were not already installed as a dependency of PyTorch.
To install with Conda:

`conda install -c conda-forge yaml matplotlib omegaconf`

## Usage

The components of the library representing different parts of the deep learning pipeline can be imported separately as
needed. The preferences and specifications for the pipeline are supplied through a configuration file.

### Configuration files
The configuration system uses `omegaconf`, taking advantage of the flexibility of YAML format. 
The options are listed separately for different components of the library, and the expected format can be observed under
`imbalance_baselines/default_config`. However, it should be kept in mind that custom fields can be added or some unused 
fields may be discarded according to desired usage, as in the case with
the example under `examples/mixup/ex-input-manifold-mixup.yaml`.

`imbalance_baselines/default_config` is used in an attempt to obtain values for missing fields that the program
tries to access.

In the configurations, a "global seed" throughout the library components is defined by default for PyTorch, Numpy and
Python's random components. The provided training code defines the model objects so that every model of the same type
are initialized with the same weights. Note that to ensure true reproducibility, environment variable modifications may
be required since indeterministic algorithms can be used automatically. See
[PyTorch's guide](https://pytorch.org/docs/stable/notes/randomness.html) for details.

### Examples

Currently, two examples are provided in the library. They can be executed by placing the `.py` scripts in an environment
that can access the `imbalance_baselines` repository, and passing the desired configuration file as a command line
argument.

#### Training Pipeline Example
[This example](examples/ex-training-pipeline.py) contains the base code for training and evaluating multiple
models using different loss functions on a dataset.

All properties of training, dataset, model, etc. are specified through the configuration files as usual. The aim of the
script completely depends on the configuration, as long as it is supported by the library's training pipeline
(`imbalance_baselines/training.py`). The script simply imports and runs library components, showing that the dataset
preparation, training and evaluation steps can be arbitrarily imported and used in projects.

Currently, class-balancing loss methods are completely supported by the training pipeline, and can be run using an
example configuration from `examples/class_balance/`. Mix-up methods are not yet completely implemented in the pipeline
and with sampling methods, the pipeline is yet to be tested.

Assuming that the script can import the library from the working directory:

This example trains two models of the same type, using the regular and class balancing versions of the focal loss, and
evaluates them:

```$ python3 ex_training_pipeline.py examples/class_balance/ex-cbfocal-pipeline.yaml```

The output of the example above is:

<details>
<summary>Output</summary>

```
[2023-01-28 11:27:58,633 - WARNING] Value "global_seed" is incomplete in configuration. Trying to use the default value...
[2023-01-28 11:27:58,634 - WARNING] Got default value for global_seed: 42
[2023-01-28 11:27:58,639 - WARNING] Value "logging_level" is incomplete in configuration. Trying to use the default value...
[2023-01-28 11:27:58,639 - WARNING] Got default value for logging_level: INFO
Files already downloaded and verified
Files already downloaded and verified
[2023-01-28 11:28:01,591 - INFO] Number of training samples:
[2023-01-28 11:28:01,591 - INFO] [5000 2997 1796 1077  645  387  232  139   83   50]
Formed dataset and dataloaders with 10 classes.
[2023-01-28 11:28:10,251 - INFO] Got class-balancing weights: tensor([0.1812, 0.1894, 0.2157, 0.2729, 0.3785, 0.5606, 0.8688, 1.3862, 2.2584,
        3.6883], device='cuda:0')
[2023-01-28 11:28:10,415 - INFO] Starting training.
[2023-01-28 11:28:10,415 - INFO] Dataset: Long-Tailed CIFAR10
[2023-01-28 11:28:10,415 - INFO] Optimizer: Stochastic Gradient Descent with Momentum and Linear Warm-up
Epoch: 1 | Batch: 97
                                                      Focal Loss 1.548289788614618
                                       Class Balanced Focal Loss 0.5544903718892684

Epoch: 2 | Batch: 97
                                                      Focal Loss 1.439890690463336
                                       Class Balanced Focal Loss 0.5314467319360663

Epoch: 3 | Batch: 97
                                                      Focal Loss 1.3629619118282807
                                       Class Balanced Focal Loss 0.5140372297435765

Epoch: 4 | Batch: 97
                                                      Focal Loss 1.2773334160941177
                                       Class Balanced Focal Loss 0.5188421745513672

Epoch: 5 | Batch: 97
                                                      Focal Loss 1.2597391394370483
                                       Class Balanced Focal Loss 0.49396369254714956

Epoch: 6 | Batch: 97
                                                      Focal Loss 1.189922014947332
                                       Class Balanced Focal Loss 0.48668119555473155

Epoch: 7 | Batch: 97
                                                      Focal Loss 1.1496834715110908
                                       Class Balanced Focal Loss 0.46902456270925086

Epoch: 8 | Batch: 97
                                                      Focal Loss 1.1250419417170443
                                       Class Balanced Focal Loss 0.46395138398687014

Epoch: 9 | Batch: 97
                                                      Focal Loss 1.0923474024288584
                                       Class Balanced Focal Loss 0.4538307299376475

Epoch: 10 | Batch: 97
                                                      Focal Loss 1.0536850345060318
                                       Class Balanced Focal Loss 0.44666730904746144
[2023-01-28 11:40:38,307 - INFO] Saved model ResNet-32, Focal Loss on Long-Tailed CIFAR10 to output/models/resnet32_focal_epoch5_2023-01-28-11:40:38.pth
[2023-01-28 11:40:38,358 - INFO] Saved model ResNet-32, Class Balanced Focal Loss on Long-Tailed CIFAR10 to output/models/resnet32_cb_focal_epoch5_2023-01-28-11:40:38.pth
[2023-01-28 11:52:55,585 - INFO] Saved model ResNet-32, Focal Loss on Long-Tailed CIFAR10 to output/models/resnet32_focal_epoch10_2023-01-28-11:52:55.pth
[2023-01-28 11:52:55,638 - INFO] Saved model ResNet-32, Class Balanced Focal Loss on Long-Tailed CIFAR10 to output/models/resnet32_cb_focal_epoch10_2023-01-28-11:52:55.pth
Training completed, evaluating models...
[2023-01-28 11:52:55,639 - INFO] Starting evaluation with method: Prediction Accuracy Calculation
ResNet-32 trained with Focal Loss:
  Average top-1 accuracy: 0.2519%
Top-1 accuracy per class: [0.971000075340271, 0.8670000433921814, 0.38600000739097595, 0.026000000536441803, 0.2680000066757202, 0.0, 0.0, 0.0010000000474974513, 0.0, 0.0]

ResNet-32 trained with Class Balanced Focal Loss:
  Average top-1 accuracy: 0.286%
Top-1 accuracy per class: [0.8300000429153442, 0.9190000295639038, 0.29200002551078796, 0.05400000140070915, 0.04800000041723251, 0.013000000268220901, 0.41100001335144043, 0.1210000067949295, 0.0, 0.1720000058412552]
```
</details>

```$ python3 ex_training_pipeline.py examples/class_balance/ex-all-losses-pipeline.yaml```

This example, on the other hand, trains more models with more loss function configurations for a longer time. At the end
of training, the progress of the training loss of each model is plotted. 

#### Input and Manifold Mix-Up Example
This example simply trains and evaluates a model using Manifold Mix-up and Input Mix-up together. Notably, this example
uses a custom field to store the model's options.


# Directory Structure
All of the components are organized into files and simply are stored in a single directory named `imbalance_baselines`.
Any Python script that can access the library directory can use the desired classes as well as the training pipeline as
in the given examples.

Additionally, `__init__.py`  helps Python recognize the directory as a library.
It holds variables and functions that concern the whole library, as well as dictionaries that match
internal configuration names with more descriptive strings.

# Implemented Components & Imbalance Mitigation Methods
## Focal Loss
Implemented under: `imbalance_baselines/loss_functions.py`, in `FocalLoss` class.

This class features our own implementation of Focal Loss.

## Class-balancing Weights Based on Effective Number of Samples
Implemented under: `imbalance_baselines/loss_functions.py` in `FocalLoss` class and
`imbalance_baselines/datasets.py` in `get_cb_weights` function.

This imbalance mitigation method is developed as described in the original paper[^1]. 

## Input Mix-up
Implemented under: `imbalance_baselines/loss_functions.py`, in `InputMixup` class.

Input Mix-up works by creating a new data by interpolating two randomly chosen dataset samples.
This method was adapted from the Bag of Tricks code repository[^2].

## Manifold Mix-up with ResNet-32 
Implemented under: `imbalance_baselines/models.py`, in `ResNet32ManifoldMixup` class.

Similar to Input Mix-up, Manifold Mix-up works by creating new examples by interpolating the output of an intermediate
layer of the model on two random dataset samples.
This method was also adapted from the Bag of Tricks code repository[^2].

## Fine-tuning with Mix-up
Implemented under: `imbalance_baselines/training.py`, in `finetune_mixup` function.

He et al.[^3] shows that for models trained using mix-up methods, finetuning without mix-up for a number of epochs can
increase performance. This fine-tuning step is implemented as a continuation of the mix-up method adapted from Bag of
Tricks[^2].

## Sampling methods
These dataset augmentation methods were adapted from the Bag of Tricks code repository[^2].

### Under-sampling
Implemented under: `imbalance_baselines/sampling.py`, in `UnderSampler` class.

Tries balancing an imbalanced dataset by dropping examples from over-represented classes.

### Over-sampling
Implemented under: `imbalance_baselines/sampling.py`, in `OverSampler` class.

Tries balancing an imbalanced dataset by copying examples in under-represented classes.

### Class-balanced sampling
Implemented under: `imbalance_baselines/sampling.py`, in `ClassBalancedSampling` class.

Samples from the imbalanced dataset such that each class has an equal probablility of being chosen.

### Progressively-balanced sampling
Implemented under: `imbalance_baselines/sampling.py`, in `ProgressivelyBalancedSampling` class.

Applies class-balanced sampling gradually, changing linearly from totally imbalanced sampling at the start to
class-balanced sampling. Similar to the learning rate warm-up policy that can be applied at the start of training. 

# Adding New Components
Adding new components may require some additional steps other than defining the desired class in the corresponding
`.py` file:

## Adding a new dataset
* Any dataset class deriving `torch.utils.data.Datasets` is supported. If offline sampler support is desired,
the class should also receive the sampler object as a parameter and use it to modify the set.
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`DSET_NAMES` dictionary in `imbalance_baselines/__init__.py`.
* The number of classes in the dataset should be provided to `DSET_CLASS_CNTS` dictionary in
`imbalance_baselines/__init__.py` along with the chosen internal string representation of the dataset.

## Adding a new model (backbone)
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`MODEL_NAMES` dictionary in `imbalance_baselines/__init__.py`.
* If using the training pipeline is desired, the model object has to be defined in the `TrainTask` class'
`__init__` method. If needed, correct initialization must be done while training tasks are being initialized at the
start of the `train_models` function.

## Adding a new optimizer
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`OPT_NAMES` dictionary in `imbalance_baselines/__init__.py`.
* If using the training pipeline is desired, the internal name of the optimizer has to be correctly recognized
in the `train_models` function, after the initialization of the training tasks are completed.

## Adding a new evaluation method
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`EVAL_NAMES` dictionary in `imbalance_baselines/__init__.py`.

## Adding a new loss function
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`LOSS_NAMES` dictionary in `imbalance_baselines/__init__.py`.
* If using the training pipeline is desired, the loss object has to be defined in the `TrainTask` class'
`__init__` method. If needed, correct initialization must be done while training tasks are being initialized at the
start of the `train_models` function.

## Adding a new sampling method
Currently, no extra steps are required other than making any necessary modifications in the existing dataset classes.

# Features Considered for Addition
* wandb support may be implemented.
* Epoch times should be measured.
* Evaluation on test or validation sets should be able to be run every few epochs.
* `tqdm` module may be utilized to track the progress better.

# License
This program is licensed under Apache License 2.0. See the `LICENSE` file for details.

# Credits & Contact
This project was developed by:
* Berkin Kerim Konar (berkinkerimkonar@gmail.com) 
* Kıvanç Tezören (kivanctezoren@gmail.com)

(Equal contributors in the library's architecture, method implementations and testing.) 

The project was advised by:
* [Emre Akbaş, Ph.D.](https://user.ceng.metu.edu.tr/~emre/)
* [Sinan Kalkan, Ph.D.](https://user.ceng.metu.edu.tr/~skalkan/)

We also gratefully acknowledge the computational resources kindly provided by [METU-ROMER (Center for Robotics and
Artificial Intelligence)](http://romer.metu.edu.tr/) and [METU ImageLab](https://image.ceng.metu.edu.tr/).


## References:
[^1]: Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss
Based on Effective Number of Samples. 2019 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2019.00949

[^2]: Zhang, Y., Wei, X. S., Zhou, B., & Wu, J. (2021). Bag of Tricks for Long-Tailed Visual
Recognition with Deep Convolutional Neural Networks. In Proceedings of the AAAI Conference
on Artificial Intelligence (Vol. 35, No. 4, pp. 3447-3455). Repository: https://github.com/zhangyongshun/BagofTricks-LT

[^3]: He, Z.; Xie, L.; Chen, X.; Zhang, Y.; Wang, Y.; and Tian, Q. 2019b. Data augmentation revisited: Rethinking the
distribution gap between clean and augmented data. arXiv preprint arXiv:1909.09148.

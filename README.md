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

#### Class-Balancing Loss Training Pipeline Example
...

...
(If any optional fields are omitted, they are completed as in
`imbalance_baselines/default_config.yaml`)


[The script](examples/class-balance/ex-cb-pipeline.py) under `examples/class_balance/ex_cb_pipeline.py` can be run with the command:

```$ ...```

producing the output:

<details>
<summary>Output</summary>

```
... TODO
```
</details>

#### Input and Manifold Mix-Up Example
This example simply trains and evaluates a model using Manifold Mix-up and Input Mix-up together. Notably, this example
uses a custom field to store the model's options.

...

# Directory Structure
All of the components are organized into files and simply are stored in a single directory named `imbalance_baselines`.
Any Python script that can access the library directory can use the desired classes as well as the training pipeline as
in the given examples.

# Implemented Components & Imbalance Mitigation Methods
## Focal Loss
Implemented under: `imbalance_baselines/loss_functions.py`, in `FocalLoss` class.
...
TODO should also provide pytorch version

## Class-balancing Weights Based on Effective Number of Samples
Implemented under: `imbalance_baselines/loss_functions.py` in `FocalLoss` class and
`imbalance_baselines/utils.py` in `get_cb_weights` function.
...

## Input Mix-up
Implemented under: `imbalance_baselines/loss_functions.py`, in `InputMixup` class.
... Adapted from bag-of-tricks...

## Manifold Mix-up with ResNet-32 
Implemented under: `imbalance_baselines/models.py`, in `ResNet32ManifoldMixup` class.
... Adapted from Bag of Tricks...

## Fine-tuning with Mix-up
Implemented under: `imbalance_baselines/training.py`, in `finetune_mixup` function.
... Continuation of Bag of Tricks...

## Under-sampling
Implemented under: `imbalance_baselines/sampling.py`, in `UnderSampler` class.
...

## Over-sampling
Implemented under: `imbalance_baselines/sampling.py`, in `OverSampler` class.
...

## Class-balanced sampling
Implemented under: `imbalance_baselines/sampling.py`, in `ClassBalancedSampling` class.
...

## Progressively-balanced sampling
Implemented under: `imbalance_baselines/sampling.py`, in `ProgressivelyBalancedSampling` class.
...


# Adding New Components
...
* Each added dataset/loss/model/... must have ... fields...
  * e.g. Each model needs num_classes field to determine output size
  * each added dataset should consider offline sampling support
* For each added dataset/loss/model/..., ... must be added in config.py, dataset.py/loss.py/(...).py,
\_\_init__.py...
  * Task field initializations are done in training.py -- may need to add string checks for field names

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

## Adding a new optimizer
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`OPT_NAMES` dictionary in `imbalance_baselines/__init__.py`.

## Adding a new evaluation method
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`EVAL_NAMES` dictionary in `imbalance_baselines/__init__.py`.

## Adding a new loss function
* The string chosen to be used in configuration and a more descriptive name pair must be added to
`LOSS_NAMES` dictionary in `imbalance_baselines/__init__.py`.

## Adding a new sampling method
...

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
  * Implementatation of the class-balancing loss, sampling and manifold mix-up methods 
* Kıvanç Tezören (kivanctezoren@gmail.com)
  * Implementation of class-balancing loss methods, the training pipeline and the library structure 

The project was advised by:
* [Emre Akbaş, Ph.D.](https://user.ceng.metu.edu.tr/~emre/)
* [Sinan Kalkan, Ph.D.](https://user.ceng.metu.edu.tr/~skalkan/)

We also gratefully acknowledge the computational resources kindly provided by [METU-ROMER (Center for Robotics and
Artificial Intelligence)](http://romer.metu.edu.tr/) and [METU ImageLab](https://image.ceng.metu.edu.tr/).

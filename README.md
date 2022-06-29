# imbalance-baselines

**imbalance-baselines** is a modular Python deep learning library that provides a common measure for the performance of
different imbalance mitigation methods.  

## Table of Contents
1. [Installation](#installation)
   1. [Requirements](#requirements)
2. [Usage](#usage)
   1. [Configuration Files](#configuration-files)
   2. [Example](#example)
3. [Directory Structure](#directory-structure)
4. [Implemented Components & Imbalance Mitigation Methods](#implemented-components--imbalance-mitigation-methods)
5. [Adding New Components](#adding-new-components)
   1. [Adding a new dataset](#adding-a-new-dataset)
   2. [Adding a new model (backbone)](#adding-a-new-model-backbone)
   3. [Adding a new optimizer](#adding-a-new-optimizer)
   4. [Adding a new evaluation method](#adding-a-new-evaluation-method)
   5. [Adding a new loss function](#adding-a-new-loss-function)
   6. [Adding a new sampling method](#adding-a-new-sampling-method)
6. [Features Considered for Addition](#features-considered-for-addition)
7. [License](#license)

## Installation
...

### Requirements

The experiments were run using the Conda environment specified in `environment.yaml`, using Python 3.8. Used libraries
are:
* torch
* numpy
* pilllow
* datetime
* json
* matplotlib
* typing
* pathlib
* pprint
* yaml
* sys
* os

(TODO: Specify each required library with version)

## Usage

The components of the library representing different parts of the deep learning pipeline can be imported separately as
needed. The preferences and specifications for the pipeline are supplied through a configuration file.

### Configuration files
...

### Example

In this example, each component of the library is imported and to create the pipeline from scratch. A set of models
using different loss functions and backbones are trained for 200 epochs on the imbalanced version of the CIFAR-10
dataset. The imbalance of the dataset is visualized with a plot when it is initialized. Class-balancing weights on the
loss functions are used as the imbalance mitigation method. When the training completes, convergence of the loss is
visualized for every model and loss configuration. Later, the models' accuracies are tested and compared.

This example can be found in `examples/class-balance`. The configuration file is as follows (If any optional fields are omitted, they are completed as in
`imbalance_baselines/default_config.yaml`):
```yaml
# TODO: Copy the yaml
```

The training script is:
```python
# TODO: Copy from cb. example
```

Output:
```
# TODO
```

# Directory Structure
...

# Implemented Components & Imbalance Mitigation Methods
## Focal Loss
Implemented under: `imbalance_baselines/loss_functions.py`, in `FocalLoss` class.
...

## Class-balancing Weights Based on Effective Number of Samples
Implemented under: `imbalance_baselines/loss_functions.py` in `FocalLoss` class and
`imbalance_baselines/utils.py` in `get_weights` function.
...

## Input Mix-up
Implemented under: `imbalance_baselines/loss_functions.py`, in `InputMixup` class.
...

## Manifold Mix-up with ResNet-32 
Implemented under: `imbalance_baselines/models.py`, in `ResNet32ManifoldMixup` class.
...

## Fine-tuning with Mix-up
Implemented under: `imbalance_baselines/training.py`, in `finetune_mixup` function.
...

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
* Print messages should be timestamped and logged in a file.
* Epoch times should be measured.
* Evaluation on test or validation sets should be able to be run every few epochs.

# License
This program is licensed under Apache License 2.0. See the `LICENSE` file for details.

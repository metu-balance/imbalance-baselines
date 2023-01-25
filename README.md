# imbalance-baselines

**imbalance-baselines** is a modular Python deep learning library, providing a common measure for the performance of
different imbalance mitigation methods.  

## Table of Contents
1. [About imbalance-baselines](#about-imbalance-baselines)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [Configuration Files](#configuration-files)
   2. [Example](#example)
5. [Directory Structure](#directory-structure)
6. [Implemented Components & Imbalance Mitigation Methods](#implemented-components--imbalance-mitigation-methods)
7. [Adding New Components](#adding-new-components)
   1. [Adding a new dataset](#adding-a-new-dataset)
   2. [Adding a new model (backbone)](#adding-a-new-model-backbone)
   3. [Adding a new optimizer](#adding-a-new-optimizer)
   4. [Adding a new evaluation method](#adding-a-new-evaluation-method)
   5. [Adding a new loss function](#adding-a-new-loss-function)
   6. [Adding a new sampling method](#adding-a-new-sampling-method)
8. [Features Considered for Addition](#features-considered-for-addition)
9. [License](#license)

## About imbalance-baselines

TODO

developed for... as...

aims to fill the gap / need for... quickly changing methods, new ones emerge. need a common evaluation basis

not initial aim, but also provides a sample training code. adopts a "single-loop -- multiple models" approach

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
takes advantage of the flexibility of YAML format and `omegaconf`...
expected format:
... TODO

TODO custom params for different losses, optimizers, datasets, eval. methods

### Example

In this example, each component of the library is imported and to create the pipeline from scratch. A set of models
using different loss functions and backbones are trained for 200 epochs on the imbalanced version of the CIFAR-10
dataset. The imbalance of the dataset is visualized with a plot when it is initialized. Class-balancing weights on the
loss functions are used as the imbalance mitigation method. When the training completes, convergence of the loss is
visualized for every model and loss configuration. Later, the models' accuracies are tested and compared.

The code and output below illustrates the example in `examples/class-balance`. Click to expand the relevant sections:
<details>
<summary>The configuration file</summary>

TODO: update config. Use better config & output when backup is implemented?
```yaml
Dataset:
  datasets_path: "datasets/"
  dataset_name: "IMB_CIFAR10"
  dataset_params:
    imb_factor: 100
    normalize_mu:
      val1: 0.4914
      val2: 0.4822
      val3: 0.4465
    normalize_std:
      val1: 0.2023
      val2: 0.1994
      val3: 0.2010
    jitter_brightness: 0.4
    jitter_contrast: 0.4
    jitter_saturation: 0.4
    jitter_hue: 0.25
DataGeneration:
  batch_size: 256
  num_workers: 2
  pad: 4
  image_size:
    width: 32
    height: 32
  train_shuffle: True
  sampler: None
  plotting:
    draw_dataset_plots: True
    plot_size:
      width: 16
      height: 12
    plot_path: "output/dataset_plots/"
  train_transform:
    -
      transform_name: "pad"
      transform_params:
        fill: 0
        mode: "constant"
    -
      transform_name: "random_resized_crop"
      transform_params: None
    -
      transform_name: "random_horizontal_flip"
      transform_params: None
  test_transform:
    -
      transform_name: "center_crop"
      transform_params: None
Training:
  epoch_count: 5
  multi_gpu: True
  backup:
    save_models: False
    #models_path: ""
    #load_models: False
  tasks:
    -
      model: "resnet32"
      loss: "focal"
      task_options:
        init_fc_bias: True
        focal_loss_gamma: 0.5
    -
      model: "resnet32"
      loss: "cb_focal"
      task_options:
        init_fc_bias: True
        focal_loss_gamma: 0.5
    -
      model: "resnet32"
      loss: "ce_sigmoid"
      task_options:
        init_fc_bias: True
    -
      model: "resnet32"
      loss: "cb_ce_sigmoid"
      task_options:
        init_fc_bias: True
    -
      model: "resnet32"
      loss: "ce_softmax"
      task_options:
        init_fc_bias: True
    -
      model: "resnet50"
      loss: "ce_softmax"
      task_options:
        init_fc_bias: True
    -
      model: "resnet32"
      loss: "cb_ce_softmax"
      task_options:
        init_fc_bias: True
  optimizer:
    name: "sgd_linwarmup"
    params:
      lr: 0.1
      lr_decay_epochs: [160, 180]
      lr_decay_rate: 0.1
      momentum: 0.9
      weight_decay: 2e-4
      warmup_epochs: 5
  printing:
    print_training: True
    print_epoch_frequency: 1
    print_batch_frequency: 300
  plotting:
    draw_loss_plots: True
    plot_size:
      width: 16
      height: 12
    plot_path: "output/loss_plots/"
Evaluation:
  -
    method_name: "get_accuracy"
    method_params:
      calc_avg: True
      calc_perclass: True
      top: 1
      print_task_options: True

```
</details>

(If any optional fields are omitted, they are completed as in
`imbalance_baselines/default_config.yaml`)

<details>
<summary>The training script</summary>

```python
# TODO: Copy from cb. example
```
</details>

The script can be run with the command:

```$ ...```

producing the output: 

<details>
<summary>Output</summary>
TODO update output

```
Got configuration:
{'DataGeneration': {'batch_size': 256,
                    'image_size': {'height': 32, 'width': 32},
                    'num_workers': 2,
                    'pad': 4,
                    'plotting': {'draw_dataset_plots': True,
                                 'plot_path': 'output/dataset_plots/',
                                 'plot_size': {'height': 12, 'width': 16}},
                    'sampler': 'None',
                    'test_transform': [{'transform_name': 'center_crop',
                                        'transform_params': 'None'}],
                    'train_shuffle': True,
                    'train_transform': [{'transform_name': 'pad',
                                         'transform_params': {'fill': 0,
                                                              'mode': 'constant'}},
                                        {'transform_name': 'random_resized_crop',
                                         'transform_params': 'None'},
                                        {'transform_name': 'random_horizontal_flip',
                                         'transform_params': 'None'}]},
 'Dataset': {'dataset_name': 'IMB_CIFAR10',
             'dataset_params': {'imb_factor': 100,
                                'jitter_brightness': 0.4,
                                'jitter_contrast': 0.4,
                                'jitter_hue': 0.25,
                                'jitter_saturation': 0.4,
                                'normalize_mu': {'val1': 0.4914,
                                                 'val2': 0.4822,
                                                 'val3': 0.4465},
                                'normalize_std': {'val1': 0.2023,
                                                  'val2': 0.1994,
                                                  'val3': 0.201}},
             'datasets_path': 'datasets/'},
 'Evaluation': [{'method_name': 'get_accuracy',
                 'method_params': {'calc_avg': True,
                                   'calc_perclass': True,
                                   'print_task_options': True,
                                   'top': 1}}],
 'Training': {'backup': {'load_models': False, 'save_models': False},
              'epoch_count': 5,
              'multi_gpu': True,
              'optimizer': {'name': 'sgd_linwarmup',
                            'params': {'lr': 0.1,
                                       'lr_decay_epochs': [160, 180],
                                       'lr_decay_rate': 0.1,
                                       'momentum': 0.9,
                                       'warmup_epochs': 5,
                                       'weight_decay': '2e-4'}},
              'plotting': {'draw_loss_plots': True,
                           'plot_path': 'output/loss_plots/',
                           'plot_size': {'height': 12, 'width': 16}},
              'printing': {'print_batch_frequency': 300,
                           'print_epoch_frequency': 1,
                           'print_training': True},
              'tasks': [{'loss': 'focal',
                         'model': 'resnet32',
                         'task_options': {'focal_loss_gamma': 0.5,
                                          'init_fc_bias': True}},
                        {'loss': 'cb_focal',
                         'model': 'resnet32',
                         'task_options': {'focal_loss_gamma': 0.5,
                                          'init_fc_bias': True}},
                        {'loss': 'ce_sigmoid',
                         'model': 'resnet32',
                         'task_options': {'init_fc_bias': True}},
                        {'loss': 'cb_ce_sigmoid',
                         'model': 'resnet32',
                         'task_options': {'init_fc_bias': True}},
                        {'loss': 'ce_softmax',
                         'model': 'resnet32',
                         'task_options': {'init_fc_bias': True}},
                        {'loss': 'ce_softmax',
                         'model': 'resnet50',
                         'task_options': {'init_fc_bias': True}},
                        {'loss': 'cb_ce_softmax',
                         'model': 'resnet32',
                         'task_options': {'init_fc_bias': True}}]}}
Files already downloaded and verified
Files already downloaded and verified
Number of training samples:
[5000 2997 1796 1077  645  387  232  139   83   50]
Got weights: tensor([0.0506, 0.0769, 0.1211, 0.1950, 0.3188, 0.5246, 0.8684, 1.4426, 2.4092,
        3.9927], device='cuda:0', dtype=torch.float64)
Starting training.
Dataset: Long-Tailed CIFAR10
Optimizer: Stochastic Gradient Descent with Momentum and Linear Warm-up
Epoch: 1 | Batch: 1
                                                      Focal Loss 2.3113355857652347
                                       Class Balanced Focal Loss 0.4671028448659578
                                   Cross Entropy Loss w/ Sigmoid 3.082459610677735
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.5946722811120968
                                   Cross Entropy Loss w/ Softmax 2.163457413842085
                                   Cross Entropy Loss w/ Softmax 2.72016097913876
                    Class Balanced Cross Entropy Loss w/ Softmax 0.44268706556144893

Epoch: 1 | Batch: 49
                                                      Focal Loss 1.5623736105620858
                                       Class Balanced Focal Loss 0.3822701608893069
                                   Cross Entropy Loss w/ Sigmoid 2.2002946079963044
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.5288878151847881
                                   Cross Entropy Loss w/ Softmax 1.4560797548582782
                                   Cross Entropy Loss w/ Softmax 4.886092021109583
                    Class Balanced Cross Entropy Loss w/ Softmax 0.3730006351314039

Epoch: 2 | Batch: 49
                                                      Focal Loss 1.4157535189099393
                                       Class Balanced Focal Loss 0.37433895514999044
                                   Cross Entropy Loss w/ Sigmoid 1.986499288786823
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.5150584659442548
                                   Cross Entropy Loss w/ Softmax 1.2737744495704644
                                   Cross Entropy Loss w/ Softmax 3.7335128418035715
                    Class Balanced Cross Entropy Loss w/ Softmax 0.35791634301457115

Epoch: 3 | Batch: 49
                                                      Focal Loss 1.3530345578654663
                                       Class Balanced Focal Loss 0.3553615769456919
                                   Cross Entropy Loss w/ Sigmoid 1.9187631333336044
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.503695429950026
                                   Cross Entropy Loss w/ Softmax 1.213122842871003
                                   Cross Entropy Loss w/ Softmax 4.335269324469429
                    Class Balanced Cross Entropy Loss w/ Softmax 0.3534365151387008

Epoch: 4 | Batch: 49
                                                      Focal Loss 1.2932492419758952
                                       Class Balanced Focal Loss 0.36705452816033096
                                   Cross Entropy Loss w/ Sigmoid 1.8737296923770488
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.5159317209712332
                                   Cross Entropy Loss w/ Softmax 1.1834843200245313
                                   Cross Entropy Loss w/ Softmax 3.1986316846881415
                    Class Balanced Cross Entropy Loss w/ Softmax 0.36797500484988704

Epoch: 5 | Batch: 49
                                                      Focal Loss 1.2631974480049102
                                       Class Balanced Focal Loss 0.35209558654834927
                                   Cross Entropy Loss w/ Sigmoid 1.8097543102845988
                    Class Balanced Cross Entropy Loss w/ Sigmoid 0.49639566158677323
                                   Cross Entropy Loss w/ Softmax 1.1165861874115173
                                   Cross Entropy Loss w/ Softmax 3.030411464050197
                    Class Balanced Cross Entropy Loss w/ Softmax 0.33854315569092813

Starting evaluation with method: Prediction Accuracy Calculation
ResNet-32 trained with Focal Loss using training parameters {'init_fc_bias': True, 'focal_loss_gamma': 0.5}:
  Average top-1 accuracy: 0.221%
Top-1 accuracy per class: [0.9380000233650208, 0.7570000290870667, 0.18800000846385956, 0.32600000500679016, 0.0010000000474974513, 0.0, 0.0, 0.0, 0.0, 0.0]

ResNet-32 trained with Class Balanced Focal Loss using training parameters {'init_fc_bias': True, 'focal_loss_gamma': 0.5}:
  Average top-1 accuracy: 0.2966%
Top-1 accuracy per class: [0.33800002932548523, 0.4620000123977661, 0.09600000083446503, 0.3710000216960907, 0.4050000309944153, 0.1770000010728836, 0.02200000174343586, 0.38600000739097595, 0.7090000510215759, 0.0]

ResNet-32 trained with Cross Entropy Loss w/ Sigmoid using training parameters {'init_fc_bias': True}:
  Average top-1 accuracy: 0.2166%
Top-1 accuracy per class: [0.9440000653266907, 0.7400000095367432, 0.13100001215934753, 0.3410000205039978, 0.010000000707805157, 0.0, 0.0, 0.0, 0.0, 0.0]

ResNet-32 trained with Class Balanced Cross Entropy Loss w/ Sigmoid using training parameters {'init_fc_bias': True}:
  Average top-1 accuracy: 0.2811%
Top-1 accuracy per class: [0.32500001788139343, 0.49500003457069397, 0.1210000067949295, 0.28200000524520874, 0.2770000100135803, 0.26900002360343933, 0.0, 0.3060000240802765, 0.7360000610351562, 0.0]

ResNet-32 trained with Cross Entropy Loss w/ Softmax using training parameters {'init_fc_bias': True}:
  Average top-1 accuracy: 0.2214%
Top-1 accuracy per class: [0.9440000653266907, 0.6580000519752502, 0.18800000846385956, 0.35500001907348633, 0.0690000057220459, 0.0, 0.0, 0.0, 0.0, 0.0]

ResNet-50 trained with Cross Entropy Loss w/ Softmax using training parameters {'init_fc_bias': True}:
  Average top-1 accuracy: 0.1172%
Top-1 accuracy per class: [0.8910000324249268, 0.24500000476837158, 0.01600000075995922, 0.01900000125169754, 0.0010000000474974513, 0.0, 0.0, 0.0, 0.0, 0.0]

ResNet-32 trained with Class Balanced Cross Entropy Loss w/ Softmax using training parameters {'init_fc_bias': True}:
  Average top-1 accuracy: 0.3084%
Top-1 accuracy per class: [0.5070000290870667, 0.7600000500679016, 0.00800000037997961, 0.11800000816583633, 0.40400001406669617, 0.4700000286102295, 0.003000000026077032, 0.31200000643730164, 0.5020000338554382, 0.0]


Done!
```
</details>

# Directory Structure
...

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
* `tqdm` module may be utilized to track the progress better.

# License
This program is licensed under Apache License 2.0. See the `LICENSE` file for details.

TODO credits - authors and advisors 

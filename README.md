---
# Readme TODO
This readme needs to be rewritten!
* Explain setup & requirements
* Explain directory structure & how to make additions to each file or method
    * The directory structure itself may need some organization - should we use subfolders?
    * Each added dataset/loss/model/... must have ... fields...
      * e.g. Each model needs num_classes field to determine output size
      * each added dataset should consider offline sampling support
    * For each added dataset/loss/model/..., ... must be added in config.py, dataset.py/loss.py/(...).py,
  \_\_init__.py...
      * Task field initializations are done in training.py -- may need to add string checks for field names
* Implement config. class & explain usage (list available loss, model... choices etc., explain config writing guide for
new datasets & other components)
* Examples should only contain configs. The example script should be a runner instead, capable of running any
configuraiton.

# Feature TODO
* May integrate wandb support
* Should log print messages with dates and additionally in a log file
* Should measure epoch times
* Should be able to run a test on a test/valid set every few epochs
---

# imbalance-baselines

**imbalance-baselines** is a modular Python deep learning library that provides a common measure for the performance of
different imbalance mitigation methods.  

## Table of Contents
1. [Installation](#installation)
2. [Implemented Methods](#implemented-methods)
3. [Example Usage](#example-usage)
4. [Adding New Components]
   1. [Adding a new dataset]
   2. [Adding a new model ("backbone")]
   3. [Adding a new loss function]
   4. [Adding a new sampling method]
5. [Features Considered for Addition]
6. [License]

## Installation

## Implemented Methods

## Example Usage

The components of the library representing different parts of the deep learning pipeline can be imported separately as
needed. In the example below, each component of the library is imported and to create the pipeline from scratch. The
preferences and specifications for the pipeline are supplied through a configuration file.

In this example, a set of models using different loss functions and backbones are trained for 200 epochs on the
imbalanced version of the CIFAR-10 dataset. The imbalance of the dataset is visualized with a plot when it is
initialized. Class-balancing weights on the loss functions are used as the imbalance mitigation method. When the
training completes, convergence of the loss is visualized for every model and loss configuration. Later, the models'
accuracies are tested and compared.

The configuration file is as follows (If any optional fields are omitted, they are completed as in
`imbalance_baselines/default_config.yaml`):
```yaml

```

The training script is:
```python
# Copy from cb. example
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

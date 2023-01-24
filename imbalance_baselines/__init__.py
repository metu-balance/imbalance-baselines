# TODO: May just use transformation classes themselves rather than their names in TRANSFORM_NAMES
# from torchvision import transforms as tr


# Global variables for names & abbreviations
DSET_NAMES = {
    "CIFAR10": "CIFAR10",
    "IMB_CIFAR10": "Long-Tailed CIFAR10",
    "INATURALIST_2017": "iNaturalist 2017",
    "INATURALIST_2018": "iNaturalist 2018"
}

DSET_CLASS_CNTS = {
    "CIFAR10": 10,
    "IMB_CIFAR10": 10,
    "INATURALIST_2017": 5089,
    "INATURALIST_2018": 8142
}

LOSS_NAMES = {
    "focal": "Focal Loss",
    "ce_sigmoid": "Cross Entropy Loss w/ Sigmoid",
    "ce_softmax": "Cross Entropy Loss w/ Softmax",
    "cb_focal": "Class Balanced Focal Loss",
    "cb_ce_sigmoid": "Class Balanced Cross Entropy Loss w/ Sigmoid",
    "cb_ce_softmax": "Class Balanced Cross Entropy Loss w/ Softmax",
}

MODEL_NAMES = {
    "resnet32": "ResNet-32",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "resnet32-manif-mu": "Resnet-32 w/ Manifold Mix-up"
}

OPTIMIZER_NAMES = {  # Names for optimizers
    "sgd": "Stochastic Gradient Descent with Momentum",
    "sgd_linwarmup": "Stochastic Gradient Descent with Momentum and Linear Warm-up"
}

EVAL_NAMES = {  # Names for evaluation methods
    "get_accuracy": "Prediction Accuracy Calculation"
}

TRANSFORM_NAMES = {  # Names for data transformations / augmentations
    "pad": "Padding",
    "random_resized_crop": "RandomResizedCrop",
    "random_horizontal_crop": "RandomHorizontalCrop",
    "center_crop": "CenterCrop"
}

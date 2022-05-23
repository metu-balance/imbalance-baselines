import sys
import torch.cuda

from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import utils
from imbalance_baselines.config import Config

beta = 0.9999
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML

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
                               print_freq=10, models_path="", train_focal=True, train_cb_focal=True)

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

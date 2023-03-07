import datetime as dt
from os import listdir
from os.path import isfile, join
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as torchmodels

from .loss_functions import FocalLoss, MixupLoss
from .utils import parse_cfg_str
from .datasets import get_cb_weights
from . import models

from .dataset import DSET_NAMES
from .loss import LOSS_NAMES
from .model import MODEL_NAMES
from .optimizer import OPTIMIZER_NAMES
from . import get_global_seed, logger

from .config_class_pairing import find_module_component

TIMESTAMP_FORMAT = "%Y-%m-%d-%H.%M.%S"


class TrainTask:
    def __init__(self, task_cfg, dataset_info: dict, device: torch.device = torch.device("cpu")):
        # Additional task-specific configurations: None if not specified; else, a dict of config.s.
        self.options = task_cfg.task_options
        
        self.model_name = task_cfg.model
        self.loss_name = task_cfg.loss

        # Set common seed for the model and loss objects
        self.seed = get_global_seed()

        self.model_obj = None
        self.loss_obj = None

        self.loss_history = []
        self.epoch_total_loss = 0

        # TODO: Generalize
        if self.model_name == "resnet32":
            self.model = models.ResNet32
        elif self.model_name == "resnet50":
            self.model = torchmodels.resnet50
        elif self.model_name == "resnet101":
            self.model = torchmodels.resnet101
        elif self.model_name == "resnet152":
            self.model = torchmodels.resnet152
        elif self.model_name == "resnet32_manif_mu":
            self.model = models.ResNet32ManifoldMixup(seed=self.seed, alpha=self.options.beta_dist_alpha)
        else:
            raise ValueError("Invalid model name received in TrainTask object: " + self.model_name)

        # TODO: Generalize
        if self.loss_name in ["focal", "ce_sigmoid", "cb_focal", "cb_ce_sigmoid"]:
            self.loss = FocalLoss
        elif self.loss_name in ["ce_softmax", "cb_ce_softmax"]:
            self.loss = nn.CrossEntropyLoss
        else:
            raise ValueError("Invalid loss function name received in TrainTask object: " + self.loss_name)

        # Get class balancing weights & define attribute if loss is a class balancing loss
        # TODO: Generalize
        if self.loss_name in ["cb_focal", "cb_ce_sigmoid", "cb_ce_softmax"]:
            self.cb_weights = get_cb_weights(dataset_info["train_class_sizes"], self.options.cb_beta, device=device)
            self.cb_weights.requires_grad = False

            logger.info(f"Got class-balancing weights: {self.cb_weights}")
    
    def __getitem__(self, item):
        """Get an option of the task. If it does not exist, simply return False."""
        # TODO: Should use attributes or the task config. directly instead
        if item in self.options.keys():
            return self.options[item]
        else:
            return False


def print_progress(task_list, epoch, batch, print_padding=64):
    """
    :param epoch: Training epoch count, assuming epochs are counted starting from 1 rather than 0.
    :param batch: Training batch count, assuming batches are counted starting from 1 rather than 0.
    """

    print("Epoch:", epoch, "| Batch:", batch)
    
    for t in task_list:
        print(
            LOSS_NAMES[t.loss_name].rjust(print_padding),
            t.epoch_total_loss / batch
        )

    # TODO: Add option for evaluating the model on train / validation / test sets & printing the accuracy
    #   May use the first metric given in the configuration (or a "default metric") and inform the user about the
    #   used metric.

    print()  # Print empty line


def save_all_models(training_tasks: List[TrainTask], models_path, dataset_name, epoch):
    """
    :param epoch: Training epoch count, assuming epochs are counted starting from 1 rather than 0.
    """

    for t in training_tasks:
        tstamp = dt.datetime.now().strftime(TIMESTAMP_FORMAT)

        if epoch > 0:
            file_name = t.model_name + "_" + t.loss_name + f"_epoch{epoch}_" + tstamp + ".pth"
        else:
            raise ValueError(f"Invalid epoch count (< 0): {epoch}")

        save_path = join(models_path, file_name)
        torch.save(t.model_obj.state_dict(), save_path)
        logger.info(f"Saved model {MODEL_NAMES[t.model_name]}, {LOSS_NAMES[t.loss_name]}"
                    f" on {DSET_NAMES[dataset_name]} to {save_path}")


def finetune_mixup(model: models.ResNet32ManifoldMixup, dataloader, optim, loss_fn: MixupLoss, epoch_cnt=10,
                   device: torch.device = torch.device("cpu")):
    model.close_mixup()
    loss_fn.close_mixup()

    for epoch in range(epoch_cnt):
        for i, (inp, label) in enumerate(dataloader):
            inp = inp.to(device)
            label = label.to(device)

            optim.zero_grad()

            loss = loss_fn(model(inp), label)
            loss.backward()
            optim.step()

    return model


def train_models(cfg, train_dl: DataLoader, dataset_info: dict, device: torch.device = torch.device("cpu")):
    # Parse configuration
    dataset_name = cfg.Dataset.dataset_name
    train_cfg = cfg.Training
    epoch_cnt = parse_cfg_str(train_cfg.epoch_count, int)

    opt_name = train_cfg.optimizer.name
    opt_params = train_cfg.optimizer.params
    weight_decay_value = parse_cfg_str(opt_params.weight_decay, float)

    multi_gpu = train_cfg.multi_gpu

    print_training = train_cfg.printing.print_training
    if print_training:
        print_batch_freq = parse_cfg_str(train_cfg.printing.print_batch_frequency, int)
        print_epoch_freq = parse_cfg_str(train_cfg.printing.print_epoch_frequency, int)

    draw_loss_plots = train_cfg.plotting.draw_loss_plots
    if draw_loss_plots:
        plot_size = (
            parse_cfg_str(train_cfg.plotting.plot_size.width, int),
            parse_cfg_str(train_cfg.plotting.plot_size.height, int)
        )
        plot_path = train_cfg.plotting.plot_path

    save_models = train_cfg.backup.save_models
    load_models = train_cfg.backup.load_models
    if save_models or load_models:
        models_path = train_cfg.backup.models_path
    else:
        models_path = ""
    
    # Sanitize configuration
    if print_training and (print_epoch_freq <= 0 or print_batch_freq <= 0):
        raise ValueError("Printing frequencies must be positive integers.")
    else:
        print_epoch_freq = int(print_epoch_freq)
        print_batch_freq = int(print_batch_freq)

    if save_models:
        save_epoch_interval = parse_cfg_str(train_cfg.backup.save_epoch_interval, int)

        if not models_path.endswith("/"):
            models_path += "/"

    if epoch_cnt < 0:
        epoch_cnt = 0

    # Define some training variables
    states = dict()  # Will provide the same initial state for every model of same type
    for model_name in MODEL_NAMES.keys():
        states[model_name] = None
    param_list = []
    
    # Create training tasks
    training_tasks = []
    for task_cfg in train_cfg["tasks"]:
        training_tasks.append(TrainTask(task_cfg, dataset_info, device=device))
    
    # Initialize models & losses of the tasks
    for t in training_tasks:
        # Create models, will be initialized later
        model = t.model(num_classes=dataset_info["class_count"]).to(device)
        if multi_gpu:
            model = nn.DataParallel(model)

        # Initialize the list of model parameters to be optimized
        if False and opt_params.disable_bias_weight_decay:  # FIXME: temporarily disabled, remove False in condition
            nonbias_params = list()
            bias_params = list()

            for np in model.named_parameters():
                if np[0].endswith("bias"):
                    bias_params.append(np[1])
                else:
                    nonbias_params.append(np[1])

            param_list.append({'params': nonbias_params, 'weight_decay': weight_decay_value})
            param_list.append({'params': bias_params, 'weight_decay': 0.0})
        else:
            param_list.append({'params': model.parameters(), 'weight_decay': weight_decay_value})

        # Initialize losses
        # TODO: Generalize
        if t.loss == FocalLoss:
            t.loss_obj = t.loss(device=device)
        elif t.loss == nn.CrossEntropyLoss:
            if t.loss_name == "ce_softmax":
                t.loss_obj = t.loss()
            elif t.loss_name == "cb_ce_softmax":
                t.loss_obj = t.loss(weight=t.cb_weights, reduction="sum")
        else:
            raise Exception("Unhandled loss type in task: " + str(t.loss))

        # Initialize models
        if load_models:  # Initialize model using existing state dict
            model_state_prefix = t.model_name + "_" + t.loss_name

            # Find latest saved model, according to the timestamp
            state_file_list = [f for f in listdir(models_path) if (
                isfile(join(models_path, f)) and f.startswith(model_state_prefix)
            )]

            if not state_file_list:  # No mathces
                raise FileNotFoundError(f"No matching model state has been found in {models_path} for"
                                        f" {MODEL_NAMES[t.model_name]}, {LOSS_NAMES[t.loss_name]}.")

            suffix_len = len(TIMESTAMP_FORMAT) + 4  # Timestamp string len. to sort files, +4 for .pth file extension
            state_file = sorted(state_file_list, key=lambda x: x[-suffix_len:])[-1]
            model.load_state_dict(torch.load(join(models_path, state_file), map_location=device))

            logger.info(f"Loaded model state for {MODEL_NAMES[t.model_name]}, {LOSS_NAMES[t.loss_name]}: {state_file}")
        else:  # Train the models from scratch
            if states[t.model_name]:  # Load existing model state if it is defined for the model type
                model.load_state_dict(states[t.model_name])
            else:  # Create new state for the model type
                if t["init_fc_bias"]:
                    # Initialize FC biases of models using sigmoid CE and focal losses
                    #   to avoid instability at the beginning of the training
                    b_pi = torch.tensor(0.1)
                    b = -torch.log((1 - b_pi) / b_pi)

                    if t.loss == FocalLoss:  # Used for focal loss and CE with sigmoid
                        if multi_gpu:
                            model.module.fc.bias.data.fill_(b)
                        else:
                            model.fc.bias.data.fill_(b)
                    # Initialize cross entropy loss models' FC biases with 0
                    elif t.loss == nn.CrossEntropyLoss:
                        if multi_gpu:
                            model.module.fc.bias.data.fill_(0)
                        else:
                            model.fc.bias.data.fill_(0)

                states[t.model_name] = model.state_dict()
        
        t.model_obj = model

        # TODO: Generalize
        if t.model_name == "resnet32_manif_mu":
            # Pass loss through MixupLoss
            t.loss_obj = MixupLoss(t.loss_obj, alpha=t["beta_dist_alpha"], seed=t.seed)

    # Initialize optimizer
    # TODO & FIXME: Linear warm-up choice should be a configuration rather than an optimizer type
    """
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_list,
            lr=parse_cfg_str(opt_params.lr, casttype=float),
            momentum=parse_cfg_str(opt_params.momentum, casttype=float)
        )

        lr_decay_epochs = opt_params.lr_decay_epochs
        lr_decay_rate = parse_cfg_str(opt_params.lr_decay_rate, casttype=float)

        # Unused param.s. Initialize nonetheless
        warmup_epochs = 0
        lr_warmup_step = 0
    elif opt_name == "sgd_linear_warmup":
        optimizer = torch.optim.SGD(
            param_list,
            lr=0,  # Will be graudally increased during training
            momentum=parse_cfg_str(opt_params.momentum, casttype=float)
        )

        warmup_epochs = parse_cfg_str(opt_params.warmup_epochs, int)
        lr_warmup_step = parse_cfg_str(opt_params.lr, casttype=float) / warmup_epochs

        lr_decay_epochs = opt_params.lr_decay_epochs
        lr_decay_rate = opt_params.lr_decay_rate
    else:
        raise ValueError("Optimizer name is not recognized: " + opt_name)
    """
    optimizer_class = find_module_component("optimizer", opt_name)
    optimizer = optimizer_class(params=param_list, **opt_params)

    logger.info("Starting training.")
    logger.info(f"Dataset: {DSET_NAMES[dataset_name]}")
    logger.info(f"Optimizer: {OPTIMIZER_NAMES[opt_name]}")

    for epoch in range(epoch_cnt):  # NOTE: epoch ranges from 0 to (epoch_cnt - 1). n in code refers to epoch no. n+1.
        for t in training_tasks:
            t.epoch_total_loss = 0

        if opt_name == "sgd_linear_warmup":
            # Linear warm-up of learning rate from 0 to given LR in first warmup_epochs epochs
            """
            if epoch < warmup_epochs:
                for g in optimizer.param_groups:
                    g["lr"] += lr_warmup_step
            """
            pass

        # Decay learning rate at certain epochs
        # FIXME: re-enable warm-up
        """
        if epoch + 1 in lr_decay_epochs:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay_rate
        """

        for i, (inp, target) in enumerate(train_dl):
            inp = inp.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            for t in training_tasks:
                # NOTE: Using startswith & endswith only once the loss name is detected. Should never assume
                #   the name of a loss, model, etc. otherwise.
                # TODO: Generalize
                if t.loss_name in ["focal", "ce_sigmoid", "cb_focal", "cb_ce_sigmoid"]:
                    if t.loss_name.endswith("focal"):
                        g = parse_cfg_str(t["focal_loss_gamma"], float)
                    else:
                        g = 0

                    loss = t.loss_obj(
                        t.model_obj(inp),
                        target,
                        alpha=t.cb_weights if t.loss_name.startswith("cb") else None,
                        gamma=g
                    )
                elif t.loss_name in ["ce_softmax", "cb_ce_softmax"]:
                    loss = t.loss_obj(
                        t.model_obj(inp),
                        target
                    ) / (target.shape[0] if t.loss_name.startswith("cb") else 1)
                    # If class-balancing, only normalization is needed since cb_weights are passed in loss
                    #   initialization.
                else:
                    raise Exception("Invalid loss name encountered during training: " + t.loss_name)

                loss.backward()
                t.epoch_total_loss += loss.item()

            optimizer.step()

            if print_training:
                #first_batch = (epoch == 0 and i == 0)
                first_batch = False  # Enable the comment above to print the first batch of training
                print_freq = (
                        (i % print_batch_freq == (print_batch_freq - 1))
                        and (epoch % print_epoch_freq == (print_epoch_freq - 1))
                )
                if first_batch or print_freq:
                    print_progress(task_list=training_tasks, epoch=epoch+1, batch=i+1)
        else:  # At the end of epochs
            if save_models and save_epoch_interval > 0 and (epoch + 1) % save_epoch_interval == 0:
                save_all_models(training_tasks, models_path, dataset_name, epoch=epoch+1)

            if draw_loss_plots:
                for t in training_tasks:
                    t.loss_history.append(t.epoch_total_loss / (i + 1))

            if print_training:
                print_progress(task_list=training_tasks, epoch=epoch+1, batch=i+1)

    # Mix-up fine-tuning phase, executed in separate loops for each model:
    for t in training_tasks:
        if t.model_name in ["resnet32_manif_mu"]:
            t.model_obj.close_mixup()
            t.loss_obj.close_mixup()

            for epoch in range(t["finetune_mixup_epochs"]):
                # NOTE: epoch ranges from 0 to (epoch_cnt - 1). Use n-1 for the nth epoch.

                logger.info(
                    "Starting finetuning without mix-up for "
                    + MODEL_NAMES[t.model_name]
                    + " trained with "
                    + LOSS_NAMES[t.loss_name]
                    + (" using training parameters " + str(t.options) if t.options else "")
                    + ":"
                )

                t.epoch_total_loss = 0

                # TODO: Adapt from above, consider the plots and print logs
                # TODO: Input mix-up (thru. MixupLoss) should be able to used independently of manifold mix-up
                #  Adjust config.s, loss definitions & training code accordingly.
                ...

    # Training is now done for all training tasks.

    # Save the trained models
    if save_models and (save_epoch_interval < 0 or epoch_cnt % save_epoch_interval != 0):  # Ensure models were not saved at the end of last epoch
        save_all_models(training_tasks, models_path, dataset_name, epoch=epoch_cnt)

    if draw_loss_plots:
        legend = []
        plt.figure(figsize=plot_size)

        for t in training_tasks:
            plt.plot(t.loss_history, "-")  # TODO: Plot with a sampled color
            legend.append(LOSS_NAMES[t.loss_name] + " with " + MODEL_NAMES[t.model_name])

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(legend)
        plt.title(
            f"Loss vs. Epochs on {DSET_NAMES[dataset_name]}"
        )

        tstamp = dt.datetime.now().strftime(TIMESTAMP_FORMAT)
        plt.savefig(plot_path + f"{dataset_name.lower()}-losses-{tstamp}" + ".png")

        # plt.show()
   
    # Return trained models along with loss and model names
    return [{"model": t.model_obj, "loss_name": t.loss_name, "model_name": t.model_name, "options": t.options}
            for t in training_tasks]

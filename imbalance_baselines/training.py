import datetime as dt
import logging

import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models as torchmodels
from .loss_functions import FocalLoss, MixupLoss
from .utils import parse_cfg_str, get_cb_weights
from . import models
from . import DSET_NAMES, LOSS_NAMES, MODEL_NAMES, OPTIMIZER_NAMES, get_global_seed, logger


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
        self.cb_weights = None

        self.loss_history = []
        self.epoch_total_loss = 0
        
        if self.model_name == "resnet32":
            self.model = models.ResNet32
        elif self.model_name == "resnet50":
            self.model = torchmodels.resnet50
        elif self.model_name == "resnet101":
            self.model = torchmodels.resnet101
        elif self.model_name == "resnet152":
            self.model = torchmodels.resnet152
        elif self.model_name == "resnet32_manif_mu":
            self.model = models.ResNet32ManifoldMixup(alpha=self.options.beta_dist_alpha, seed=self.seed)
        else:
            raise ValueError("Invalid model name received in TrainTask object: " + self.model_name)

        if self.loss_name in ["focal", "ce_sigmoid", "cb_focal", "cb_ce_sigmoid"]:
            self.loss = FocalLoss
        elif self.loss_name in ["ce_softmax", "cb_ce_softmax"]:
            self.loss = nn.CrossEntropyLoss
        else:
            raise ValueError("Invalid loss function name received in TrainTask object: " + self.loss_name)

        # Get class balancing weights if loss is a class balancing loss
        if self.cb_weights is None and self.loss_name in ["cb_focal", "cb_ce_sigmoid", "cb_ce_softmax"]:
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
    dataset = cfg.Dataset.dataset_name
    train_cfg = cfg.Training
    epoch_cnt = parse_cfg_str(train_cfg.epoch_count, int)

    multi_gpu = train_cfg.multi_gpu

    print_training = train_cfg.printing.print_training
    print_batch_freq = parse_cfg_str(train_cfg.printing.print_batch_frequency, int)
    print_epoch_freq = parse_cfg_str(train_cfg.printing.print_epoch_frequency, int)

    draw_loss_plots = train_cfg.plotting.draw_loss_plots
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
    
    # Sanitize print frequencies
    if print_training and (print_epoch_freq <= 0 or print_batch_freq <= 0):
        raise ValueError("Printing frequencies must be positive integers.")
    else:
        print_epoch_freq = int(print_epoch_freq)
        print_batch_freq = int(print_batch_freq)

    # Sanitize paths
    if save_models:
        if not models_path.endswith("/"):
            models_path += "/"
    
        # Create temporary dir. under model_path if it does not exist
        os.makedirs("temp/interrupted/", mode=611, exist_ok=True)
        os.makedirs("temp/epoch_end/", mode=611, exist_ok=True)
    
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
        # Initialize models
        model = t.model(num_classes=dataset_info["class_count"]).to(device)
        if multi_gpu:
            model = nn.DataParallel(model)
        
        param_list.append({'params': model.parameters()})
        
        if states[t.model_name]:
            model.load_state_dict(states[t.model_name])
        else:
            states[t.model_name] = model.state_dict()
        
        t.model_obj = model

        # Initialize losses
        if t.loss == FocalLoss:
            t.loss_obj = t.loss(device=device)
        elif t.loss == nn.CrossEntropyLoss:
            if t.loss_name == "ce_softmax":
                t.loss_obj = t.loss()
            elif t.loss_name == "cb_ce_softmax":
                t.loss_obj = t.loss(weight=t.cb_weights, reduction="sum")
        else:
            raise Exception("Unhandled loss type in task: " + str(t.loss))

        if t.model_name == "resnet32_manif_mu":
            # Pass loss through MixupLoss
            t.loss_obj = MixupLoss(t.loss_obj, alpha=t["beta_dist_alpha"], seed=t.seed)
    
    # TODO [1]: Loading models may be handled by a different func. or with different parameters
    if load_models:
        # Assuming the file exists for each model that will be tested:
        # TODO [1]: Catch loading errors in try-except blocks
        # TODO [1]: Reach saved models using task objects' loss_name, model_name, ... fields
        """
        if train_focal:
            rn_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_focal_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_focal_{dataset}.pth")
        
        if train_sigmoid_ce:
            rn_sigmoid_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth")
        
        if train_softmax_ce:
            rn_softmax_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth")
        
        if train_cb_focal:
            rn_cb_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth")
        
        if train_cb_sigmoid_ce:
            rn_cb_sigmoid_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth")
        
        if train_cb_softmax_ce:
            rn_cb_softmax_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth",
                           map_location=device)
            )
            logger.info(f"Loaded model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth")
        """
        pass
    else:  # Train the models from scratch
        # Initialize FC biases
        b_pi = torch.tensor(0.1)
        b = -torch.log((1 - b_pi) / b_pi)
        for t in training_tasks:
            if t["init_fc_bias"]:
                # Initialize FC biases of models using sigmoid CE and focal losses
                #   to avoid instability at the beginning of the training
                if t.loss == FocalLoss:  # Used for focal loss and CE with sigmoid
                    if multi_gpu:
                        t.model_obj.module.fc.bias.data.fill_(b)
                    else:
                        t.model_obj.fc.bias.data.fill_(b)
                # Initialize cross entropy loss models' FC biases with 0
                elif t.loss == nn.CrossEntropyLoss:
                    if multi_gpu:
                        t.model_obj.module.fc.bias.data.fill_(0)
                    else:
                        t.model_obj.fc.bias.data.fill_(0)

        # TODO [3]: Add option to disable optimizer's weight decay for the biases.
        #  (is simply turning grad. off correct?)
        #rn_focal.fc.bias.requires_grad_(False)
        #rn_sigmoid_ce.fc.bias.requires_grad_(False)
        #rn_cb_focal.fc.bias.requires_grad_(False)
        #rn_cb_sigmoid_ce.fc.bias.requires_grad_(False)
        
        # Initialize optimizer
        opt_name = train_cfg.optimizer.name
        opt_params = train_cfg.optimizer.params
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_list,
                lr=parse_cfg_str(opt_params.lr, casttype=float),
                momentum=parse_cfg_str(opt_params.momentum, casttype=float),
                weight_decay=parse_cfg_str(opt_params.weight_decay, casttype=float)
            )
            
            lr_decay_epochs = opt_params.lr_decay_epochs
            lr_decay_rate = parse_cfg_str(opt_params.lr_decay_rate, casttype=float)
            
            # Unused param.s. Initialize nonetheless
            warmup_epochs = 0
            lr_warmup_step = 0
        elif opt_name == "sgd_linwarmup":
            optimizer = torch.optim.SGD(
                param_list,
                lr=0,  # Will be graudally increased during training
                momentum=parse_cfg_str(opt_params.momentum, casttype=float),
                weight_decay=parse_cfg_str(opt_params.weight_decay, casttype=float)
            )
            
            warmup_epochs = parse_cfg_str(opt_params.warmup_epochs, int)
            lr_warmup_step = parse_cfg_str(opt_params.lr, casttype=float) / warmup_epochs

            lr_decay_epochs = opt_params.lr_decay_epochs
            lr_decay_rate = opt_params.lr_decay_rate
        else:
            raise Exception("Optimizer name is not recognized: " + opt_name)
        
        logger.info("Starting training.")
        logger.info(f"Dataset: {DSET_NAMES[dataset]}")
        logger.info(f"Optimizer: {OPTIMIZER_NAMES[opt_name]}")
        
        try:
            for epoch in range(epoch_cnt):  # NOTE: epoch ranges from 0 to (epoch_cnt - 1). Use n-1 for the nth epoch.
                for t in training_tasks:
                    t.epoch_total_loss = 0
                
                if opt_name == "sgd_linwarmup":
                    # Linear warm-up of learning rate from 0 to given LR in first warmup_epochs epochs
                    if epoch < warmup_epochs:
                        for g in optimizer.param_groups:
                            g["lr"] += lr_warmup_step

                # Decay learning rate at certain epochs
                if epoch + 1 in lr_decay_epochs:
                    for g in optimizer.param_groups:
                        g["lr"] *= lr_decay_rate
                
                for i, (inp, target) in enumerate(train_dl):
                    inp = inp.to(device)
                    target = target.to(device)
                    
                    optimizer.zero_grad()
                    
                    for t in training_tasks:
                        # NOTE: Using startswith & endswith only once the loss name is detected. Should never assume
                        #   the name of a loss, model, etc. otherwise.
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
                        first_batch = (epoch == 0 and i == 0)
                        print_freq = (
                                (i % print_batch_freq == (print_batch_freq - 1))
                                and (epoch % print_epoch_freq == (print_epoch_freq - 1))
                        )
                        if first_batch or print_freq:
                            print_progress(task_list=training_tasks, epoch=epoch+1, batch=i+1)
                else:  # The end of each epoch
                    if save_models:  # Temporary backup for each epoch
                        # Delete all temporary files under temp/epoch_end/
                        tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                        
                        for f in os.listdir(models_path + "temp/epoch_end/"):
                            fpath = models_path + "temp/epoch_end/" + f
                            os.remove(fpath)
                            #print("Removed:", fpath)
                        
                        """
                        if train_focal:
                            torch.save(
                                rn_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_sigmoid_ce:
                            torch.save(
                                rn_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_softmax_ce:
                            torch.save(
                                rn_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_focal:
                            torch.save(
                                rn_cb_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_sigmoid_ce:
                            torch.save(
                                rn_cb_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_softmax_ce:
                            torch.save(
                                rn_cb_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #logger.info(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        """
                        
                    if draw_loss_plots:
                        for t in training_tasks:
                            t.loss_history.append(t.epoch_total_loss / (i + 1))
                    
                    if print_training:
                        print_progress(task_list=training_tasks, epoch=epoch+1, batch=i+1)

            # Mix-up fine-tuning phase, executed in separate loops for each model:
            for t in training_tasks:
                if t.model_name in ["resnet32_manif_mu"]:
                    for epoch in range(t["finetune_mixup_epochs"]):
                        # NOTE: epoch ranges from 0 to (epoch_cnt - 1). Use n-1 for the nth epoch.

                        logger.info(
                            "Starting mix-up finetuning for "
                            + MODEL_NAMES[t.model_name]
                            + " trained with "
                            + LOSS_NAMES[t.loss_name]
                            + (" using training parameters " + str(t.options) if t.options else "")
                            + ":"
                        )

                        t.model_obj.close_mixup()
                        t.loss_obj.close_mixup()

                        # TODO [5]: Adapt from above, consider the plots and print logs
                        ...

        except KeyboardInterrupt:
            logger.info("Got KeyboardInterrupt.")
            
            if save_models:
                logger.info("Backing up the models.")

                tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                
                """
                if train_focal:
                    torch.save(
                        rn_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_sigmoid_ce:
                    torch.save(
                        rn_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_softmax_ce:
                    torch.save(
                        rn_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_focal:
                    torch.save(
                        rn_cb_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_sigmoid_ce:
                    torch.save(
                        rn_cb_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_softmax_ce:
                    torch.save(
                        rn_cb_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    logger.info(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                """

            logger.info("Deleting previous backups.")
            # Delete all OLD temporary files under temp/interrupted/
            # FIXME [1]: delete only old files! create a list of old files before backing up.
            for f in os.listdir(models_path + "temp/interrupted/"):
                fpath = models_path + "temp/interrupted/" + f
                os.remove(fpath)
                logger.info("Removed:", fpath)

            logger.info("Terminating.")
            sys.exit(1)

        # Save the trained models
        if save_models:
            tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            """
            if train_focal:
                torch.save(
                    rn_focal.state_dict(),
                    models_path + f"rn{resnet_type}_focal_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_focal_{dataset}_{tstamp}.pth")
            
            if train_sigmoid_ce:
                torch.save(
                    rn_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}_{tstamp}.pth")
            
            if train_softmax_ce:
                torch.save(
                    rn_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_softmax_ce_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_softmax_ce_{dataset}_{tstamp}.pth")
            
            if train_cb_focal:
                torch.save(
                    rn_cb_focal.state_dict(),
                    models_path + f"rn{resnet_type}_cb_focal_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_focal_{dataset}_{tstamp}.pth")
            
            if train_cb_sigmoid_ce:
                torch.save(
                    rn_cb_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}_{tstamp}.pth")
            
            if train_cb_softmax_ce:
                torch.save(
                    rn_cb_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}_{tstamp}.pth"
                )
                logger.info(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}_{tstamp}.pth")
            """
        
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
                f"Loss vs. Epochs on {DSET_NAMES[dataset]}"
            )
            
            tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            plt.savefig(plot_path + f"{dataset.lower()}-losses-" + tstamp + ".png")
            
            # plt.show()
   
    # Return trained models along with loss and model names
    return [{"model": t.model_obj, "loss_name": t.loss_name, "model_name": t.model_name, "options": t.options}
            for t in training_tasks]

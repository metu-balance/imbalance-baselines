# TODO: Check out wandb
import datetime as dt
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models as torchmodels
from .loss_functions import FocalLoss
from . import models
from . import DSET_NAMES


class TrainTask:
    def __init__(self, task_cfg):
        # Additional task-specific configurations: None if not specified; else, a dict of config.s.
        self.options = task_cfg["task_options"]
        
        self.model_name = task_cfg["model"]
        self.loss_name = task_cfg["loss"]
        #self.eval_name = task_cfg["eval"]  # TODO [5]: Implement eval config
        
        self.model_obj = None
        self.loss_obj = None
        
        self.loss_history = []
        
        # TODO: Add support for more models
        # TODO: Map model names & objects elsewhere in a dict, preferably in models.py
        if self.model_name == "resnet32":
            self.model = models.ResNet32
        elif self.model_name == "resnet50":
            self.model = torchmodels.resnet50
        elif self.model_name == "resnet101":
            self.model = torchmodels.resnet101
        elif self.model_name == "resnet152":
            self.model = torchmodels.resnet152
        else:
            # TODO [1]: Do this in config.py
            raise ValueError("Invalid model name given: " + self.model_name)

        # TODO: Map model names & objects elsewhere in a dict, preferably in loss_functions.py
        # TODO: Do not initialize the losses here, do not pass weights to class
        if self.loss_name in ["focal", "ce_sigmoid", "cb_focal", "cb_ce_sigmoid"]:
            self.loss = FocalLoss
        elif self.loss_name in ["ce_softmax", "cb_ce_softmax"]:
            self.loss = nn.CrossEntropyLoss
        else:
            # TODO [1]: Do this in config.py
            raise ValueError("Invalid loss function name given: " + self.loss_name)
    
    def __getitem__(self, item):
        """Get an option of the task. If it does not exist, simply return False."""
        if item in self.options.keys():
            return self.options[item]
        else:
            return False
    

# TODO [2]: Pass train & test preferences toghether in lists instead of separate param.s (See: TODO in config.s).
#   Iterate over the lists (for eg. loss, model choices) later to avoid code repetition.
#   This should also satisfy the main intention of the project, which is to provide a baseline for
#   experiments described using preference combinations. Just choose model architecture, dataset,
#   loss fn., sampling method... etc. and then program should take care of the rest. The current
#   structure may need to be modified (e.g it might not be suitable for multiple data transformation
#   methods, since it would require different iterations over the dataset for different experiments).
# TODO [4]: Pass double precision preference through cfg
# TODO [3]: Should not pass weights, should call utils.get_weights when necessary
# TODO: Throughout the function, check whether iterations over the tasks can be merged.
def train_models(cfg, train_dl: DataLoader, class_cnt: int, weights: [float] = None,
                 device: torch.device = torch.device("cpu")):
    # Parse configuration
    # TODO: Check these config variable usages since they were converted from func. param.s, may omit some.
    dataset = cfg["Dataset"]["name"]
    train_cfg = cfg["Training"]
    epoch_cnt = train_cfg["epoch_count"]
    multi_gpu = train_cfg["multi_gpu"]
    resnet_type = train_cfg["model"]
    print_training = train_cfg["printing"]["print_training"]
    print_freq = train_cfg["printing"]["print_frequency"]
    draw_plots = train_cfg["draw_plots"]
    save_models = train_cfg["backup"]["save_models"]
    load_models = train_cfg["backup"]["load_models"]
    if save_models or load_models:
        models_path = train_cfg["backup"]["models_path"]
    else:
        models_path = ""
    
    # Sanitize print_freq
    # TODO [1]: Do this in config.py
    if print_training and print_freq <= 0:
        raise ValueError("Printing frequency must be a positive integer.")
    else:
        print_freq = int(print_freq)

    # Sanitize models_path
    # TODO [1]: Do this in config.py
    if save_models:
        if not models_path.endswith("/"):
            models_path += "/"
    
        # Create temporary dir. under model_path if it does not exist
        os.makedirs("temp/interrupted/", mode=611, exist_ok=True)
        os.makedirs("temp/epoch_end/", mode=611, exist_ok=True)
    
    state = None  # Will provide the same initial state for every model
    param_list = []
    
    # Create training tasks
    training_tasks = []
    for task_cfg in train_cfg["tasks"]:
        training_tasks.append(TrainTask(task_cfg))
    
    # Initialize models & losses of the tasks
    for t in training_tasks:
        # Initialize models
        # TODO [4]: Cast to double according to the config
        model = t.model(num_classes=class_cnt).double()
        if multi_gpu:
            model = nn.DataParallel(model)
        
        param_list.append({'params': model.parameters()})
        
        if state:
            model.load_state_dict(state)
        else:
            state = model.state_dict()
        
        t.model_obj = model
        
        # Initialize losses
        if t.loss == FocalLoss:
            t.loss_obj = t.loss(device)
        elif t.loss == nn.CrossEntropyLoss:
            if t.loss_name == "ce_softmax":
                t.loss_obj = t.loss()
            elif t.loss_name == "cb_ce_softmax":
                t.loss_obj = t.loss(weight=weights, reduction="sum")
    
    # TODO: Loading models may be handled by a different func. or with different parameters
    if load_models:
        # Assuming the file exists for each model that will be tested:
        # TODO: Catch loading errors in try-except blocks
        # TODO: Reach saved models using task objects' loss_name, model_name, ... fields
        """
        if train_focal:
            rn_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_focal_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_focal_{dataset}.pth")
        
        if train_sigmoid_ce:
            rn_sigmoid_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth")
        
        if train_softmax_ce:
            rn_softmax_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth")
        
        if train_cb_focal:
            rn_cb_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth")
        
        if train_cb_sigmoid_ce:
            rn_cb_sigmoid_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth")
        
        if train_cb_softmax_ce:
            rn_cb_softmax_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth")
        """
        pass
    else:  # Train the models from scratch
        # Initialize FC biases
        b_pi = torch.tensor(0.1, dtype=torch.double)
        b = -torch.log((1 - b_pi) / b_pi)
        for t in training_tasks:
            if t["init_fc_bias"]:
                # Initialize FC biases of models using sigmoid_ce and focal losses
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

        # TODO: Add option to disable optimizer's weight decay for the biases.
        #  (is simply turning grad. off correct?)
        #rn_focal.fc.bias.requires_grad_(False)
        #rn_sigmoid_ce.fc.bias.requires_grad_(False)
        #rn_cb_focal.fc.bias.requires_grad_(False)
        #rn_cb_sigmoid_ce.fc.bias.requires_grad_(False)
        
        # Initialize optimizer
        opt_name = train_cfg["optimizer"]["name"]
        opt_params = train_cfg["optimizer"]["params"]
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_list,
                lr=opt_params["lr"],
                momentum=opt_params["momentum"],
                weight_decay=opt_params["weight_decay"]
            )
        elif opt_name == "sgd_linwarmup":
            optimizer = torch.optim.SGD(
                param_list,
                lr=0,  # Will be graudally increased during training
                momentum=opt_params["momentum"],
                weight_decay=opt_params["weight_decay"]
            )
        else:
            raise Exception("Optimizer name is not recognized: " + opt_name)
            
        # TODO: Continue conversion...
        #   Add new field holding the loss history to each task object if draw_plots is true
        #   Restrain maximum task count if draw_plots is enabled.
        if draw_plots:
            
            if train_focal: history_loss_focal = []
            if train_sigmoid_ce: history_loss_sigmoid_ce = []
            if train_softmax_ce: history_loss_softmax_ce = []
            if train_cb_focal: history_loss_cb_focal = []
            if train_cb_sigmoid_ce: history_loss_cb_sigmoid_ce = []
            if train_cb_softmax_ce: history_loss_cb_softmax_ce = []
        
        print(f"Starting training with {DSET_NAMES[dataset]} dataset,",
              f"ResNet-{resnet_type} models.")
        try:
            for epoch in range(epoch_cnt):
                if train_focal: total_loss_focal = 0
                if train_sigmoid_ce: total_loss_sigmoid_ce = 0
                if train_softmax_ce: total_loss_softmax_ce = 0
                if train_cb_focal: total_loss_cb_focal = 0
                if train_cb_sigmoid_ce: total_loss_cb_sigmoid_ce = 0
                if train_cb_softmax_ce: total_loss_cb_softmax_ce = 0
                
                if epoch < 5:
                    # Linear warm-up of learning rate from 0 to 0.1 in the first 5 epochs
                    for g in optimizer.param_groups:
                        g["lr"] += 0.02
                elif epoch in [159, 179]:
                    # Decay learning rate by 0.01 at 160th and 180th epochs
                    for g in optimizer.param_groups:
                        g["lr"] *= 0.01
                
                for i, (inp, target) in enumerate(train_dl):
                    inp = inp.double().to(device)
                    target = target.to(device)
                    
                    optimizer.zero_grad()
                    
                    if train_focal:
                        loss_focal = focal_loss(
                            rn_focal(inp),
                            target,
                            gamma=0.5,
                            device=device
                        )
                        
                        loss_focal.backward()
                        total_loss_focal += loss_focal.item()
                    
                    if train_sigmoid_ce:
                        loss_sigmoid_ce = focal_loss(
                            rn_sigmoid_ce(inp),
                            target,
                            device=device
                        )
                        
                        loss_sigmoid_ce.backward()
                        total_loss_sigmoid_ce += loss_sigmoid_ce.item()
                    
                    if train_softmax_ce:
                        loss_softmax_ce = cel(
                            rn_softmax_ce(inp),
                            target
                        )
                        
                        loss_softmax_ce.backward()
                        total_loss_softmax_ce += loss_softmax_ce.item()
                    
                    if train_cb_focal:
                        loss_cb_focal = focal_loss(
                            rn_cb_focal(inp),
                            target,
                            alpha=weights,
                            gamma=0.5,
                            device=device
                        )
                        
                        loss_cb_focal.backward()
                        total_loss_cb_focal += loss_cb_focal.item()
                    
                    if train_cb_sigmoid_ce:
                        loss_cb_sigmoid_ce = focal_loss(
                            rn_cb_sigmoid_ce(inp),
                            target,
                            alpha=weights,
                            device=device
                        )
                        
                        loss_cb_sigmoid_ce.backward()
                        total_loss_cb_sigmoid_ce += loss_cb_sigmoid_ce.item()
                    
                    if train_cb_softmax_ce:
                        loss_cb_softmax_ce = cb_softmax_cel(
                            rn_cb_softmax_ce(inp),
                            target
                        ) / target.shape[0]
                        
                        loss_cb_softmax_ce.backward()
                        total_loss_cb_softmax_ce += loss_cb_softmax_ce.item()
                    
                    optimizer.step()
                    
                    if print_training and \
                            ((epoch == 0 and i == 0) or (i % print_freq == (print_freq - 1))):
                        print("Epoch:", epoch, "| Batch:", str(i + 1))
                        
                        if train_focal:
                            print("Focal:".rjust(11), total_loss_focal/(i+1))
                        if train_sigmoid_ce:
                            print("sigmoid_ce:".rjust(11), total_loss_sigmoid_ce/(i+1))
                        if train_softmax_ce:
                            print("Cross Entropy:".rjust(11), total_loss_softmax_ce/(i+1))
                        if train_cb_focal:
                            print("CB Focal:".rjust(11), total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid_ce:
                            print("CB sigmoid_ce:", total_loss_cb_sigmoid_ce/(i+1))
                        if train_cb_softmax_ce:
                            print("CB Cross Entropy:", total_loss_cb_softmax_ce/(i+1))
                        
                        print()  # Print empty line
                else:  # The end of each epoch
                    if save_models:  # Temporary backup for each epoch
                        # Delete all temporary files under temp/epoch_end/
                        tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                        
                        for f in os.listdir(models_path + "temp/epoch_end/"):
                            fpath = models_path + "temp/epoch_end/" + f
                            os.remove(fpath)
                            #print("Removed:", fpath)
                        
                        if train_focal:
                            torch.save(
                                rn_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_sigmoid_ce:
                            torch.save(
                                rn_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_softmax_ce:
                            torch.save(
                                rn_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_focal:
                            torch.save(
                                rn_cb_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_sigmoid_ce:
                            torch.save(
                                rn_cb_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                        
                        if train_cb_softmax_ce:
                            torch.save(
                                rn_cb_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                    
                    if draw_plots:
                        if train_focal:
                            history_loss_focal.append(total_loss_focal/(i+1))
                        if train_sigmoid_ce:
                            history_loss_sigmoid_ce.append(total_loss_sigmoid_ce/(i+1))
                        if train_softmax_ce:
                            history_loss_softmax_ce.append(total_loss_softmax_ce/(i+1))
                        if train_cb_focal:
                            history_loss_cb_focal.append(total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid_ce:
                            history_loss_cb_sigmoid_ce.append(total_loss_cb_sigmoid_ce/(i+1))
                        if train_cb_softmax_ce:
                            history_loss_cb_softmax_ce.append(total_loss_cb_softmax_ce/(i+1))
                    
                    if print_training:
                        print("Epoch:", epoch, "| Batch:", str(i + 1))
                        
                        if train_focal:
                            print("Focal:".rjust(11), total_loss_focal/(i+1))
                        if train_sigmoid_ce:
                            print("sigmoid_ce:".rjust(11), total_loss_sigmoid_ce/(i+1))
                        if train_softmax_ce:
                            print("Cross Entropy:".rjust(11), total_loss_softmax_ce/(i+1))
                        if train_cb_focal:
                            print("CB Focal:".rjust(11), total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid_ce:
                            print("CB sigmoid_ce:", total_loss_cb_sigmoid_ce/(i+1))
                        if train_cb_softmax_ce:
                            print("CB Cross Entropy:", total_loss_cb_softmax_ce/(i+1))
                        
                        print()  # Print empty line
        except KeyboardInterrupt:
            print("Got KeyboardInterrupt.")
            
            if save_models:
                print("Deleting previous backups.")
                # Delete all temporary files under temp/interrupted/
                for f in os.listdir(models_path + "temp/interrupted/"):
                    fpath = models_path + "temp/interrupted/" + f
                    os.remove(fpath)
                    print("Removed:", fpath)
                
                print("Backing up the models.")

                tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                
                if train_focal:
                    torch.save(
                        rn_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_sigmoid_ce:
                    torch.save(
                        rn_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_softmax_ce:
                    torch.save(
                        rn_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_focal:
                    torch.save(
                        rn_cb_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_sigmoid_ce:
                    torch.save(
                        rn_cb_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
                
                if train_cb_softmax_ce:
                    torch.save(
                        rn_cb_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}_{tstamp}.pth")
            
            print("Terminating.")
            sys.exit(1)
        
        
        # Save the trained models
        
        if save_models:
            tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            if train_focal:
                torch.save(
                    rn_focal.state_dict(),
                    models_path + f"rn{resnet_type}_focal_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_focal_{dataset}_{tstamp}.pth")
            
            if train_sigmoid_ce:
                torch.save(
                    rn_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}_{tstamp}.pth")
            
            if train_softmax_ce:
                torch.save(
                    rn_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_softmax_ce_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_softmax_ce_{dataset}_{tstamp}.pth")
            
            if train_cb_focal:
                torch.save(
                    rn_cb_focal.state_dict(),
                    models_path + f"rn{resnet_type}_cb_focal_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_focal_{dataset}_{tstamp}.pth")
            
            if train_cb_sigmoid_ce:
                torch.save(
                    rn_cb_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}_{tstamp}.pth")
            
            if train_cb_softmax_ce:
                torch.save(
                    rn_cb_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}_{tstamp}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}_{tstamp}.pth")
        
        if draw_plots:
            legend = []
            plt.figure(figsize=(16, 12))
            
            if train_focal:
                plt.plot(history_loss_focal, "-b")
                legend.append("Focal Loss")
            if train_sigmoid_ce:
                plt.plot(history_loss_sigmoid_ce, "-r")
                legend.append("sigmoid_ce CE Loss")
            if train_softmax_ce:
                plt.plot(history_loss_softmax_ce, "-g")
                legend.append("Softmax CE Loss")
            if train_cb_focal:
                plt.plot(history_loss_cb_focal, "-c")
                legend.append("Class-Balanced Focal Loss")
            if train_cb_sigmoid_ce:
                plt.plot(history_loss_cb_sigmoid_ce, "-m")
                legend.append("Class-Balanced sigmoid_ce CE Loss")
            if train_cb_softmax_ce:
                plt.plot(history_loss_cb_softmax_ce, "-y")
                legend.append("Class-Balanced Softmax CE Loss")
            
            if legend:
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(legend)
                plt.title(
                    f"Loss vs. Epochs on {DSET_NAMES[dataset]} with ResNet-{resnet_type}"
                )
                
                plt.savefig(
                    f"./plots/{dataset.lower()}_rn-{resnet_type}-losses.png"
                )
            
                plt.show()
    
    # TODO: Convert: Return the tasks? A separate model objects list, formed with list comprehension?
    return (rn_focal, rn_sigmoid_ce, rn_softmax_ce, rn_cb_focal, rn_cb_sigmoid_ce,
            rn_cb_softmax_ce)

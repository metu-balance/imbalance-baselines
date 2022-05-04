# TODO: Check out wandb

import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models as torchmodels
from . import models
from . import loss_functions
from . import DSET_NAMES


# TODO: Parameters look cluttered, need to simplify. Use a config. class?
#   Pass train & test preferences in a list instead of separate param.s.
#   Iterate over the list later to avoid code repetition.
# TODO: Instead of save_models param, assume false if models_path == ""
#   TODO: At start of func., if save_models, create temporary dir. under model_path if it does not exist
# TODO: Add timestamp to saved models
def train_models(dataset: str, train_dl: DataLoader, class_cnt: int, weights: [float],
                 epoch_cnt: int = 200, multi_gpu: bool = False, device: torch.device = torch.device("cpu"),
                 resnet_type: str = "32", print_training: bool = True, print_freq: int = 100,
                 draw_plots: bool = False, models_path: str = "./trained_models/",
                 save_models: bool = False, load_models: bool = False, train_focal: bool = False,
                 train_sigmoid: bool = False, train_ce: bool = False, train_cb_focal: bool = False,
                 train_cb_sigmoid: bool = False, train_cb_ce: bool = False):
    
    # Sanitize print_freq
    if print_training and print_freq <= 0:
        raise ValueError("Printing frequency must be a positive integer.")
    else:
        print_freq = int(print_freq)
    
    if resnet_type == "32":
        rn = models.ResNet32
    elif resnet_type == "50":
        rn = torchmodels.resnet50
    elif resnet_type == "101":
        rn = torchmodels.resnet101
    elif resnet_type == "152":
        rn = torchmodels.resnet152
    else:
        raise ValueError("Invalid resnet_type")
    
    param_list = []
    
    rn_focal = None
    rn_sigmoid = None
    rn_ce = None
    rn_cb_focal = None
    rn_cb_sigmoid = None
    rn_cb_ce = None
    
    # The "state" var. provides the same initial state for every model
    state = None
    
    if train_focal:
        rn_focal = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_focal = nn.DataParallel(rn_focal)
        rn_focal = rn_focal.to(device)
        
        param_list.append({'params': rn_focal.parameters()})
        
        state = rn_focal.state_dict()
    if train_sigmoid:
        rn_sigmoid = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_sigmoid = nn.DataParallel(rn_sigmoid)
        rn_sigmoid = rn_sigmoid.to(device)
        
        param_list.append({'params': rn_sigmoid.parameters()})
        
        if state:
            rn_sigmoid.load_state_dict(state)
        else:
            state = rn_sigmoid.state_dict()
    if train_ce:
        rn_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_ce = nn.DataParallel(rn_ce)
        rn_ce = rn_ce.to(device)
        
        param_list.append({'params': rn_ce.parameters()})
        
        if state:
            rn_ce.load_state_dict(state)
        else:
            state = rn_ce.state_dict()
    if train_cb_focal:
        rn_cb_focal = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_focal = nn.DataParallel(rn_cb_focal)
        rn_cb_focal = rn_cb_focal.to(device)
        
        param_list.append({'params': rn_cb_focal.parameters()})
        
        if state:
            rn_cb_focal.load_state_dict(state)
        else:
            state = rn_cb_focal.state_dict()
    if train_cb_sigmoid:
        rn_cb_sigmoid = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_sigmoid = nn.DataParallel(rn_cb_sigmoid)
        rn_cb_sigmoid = rn_cb_sigmoid.to(device)
        
        param_list.append({'params': rn_cb_sigmoid.parameters()})
        
        if state:
            rn_cb_sigmoid.load_state_dict(state)
        else:
            state = rn_cb_sigmoid.state_dict()
    if train_cb_ce:
        rn_cb_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_ce = nn.DataParallel(rn_cb_ce)
        rn_cb_ce = rn_cb_ce.to(device)
        
        param_list.append({'params': rn_cb_ce.parameters()})
        
        if state:
            rn_cb_ce.load_state_dict(state)
    
    # TODO: Loading models may be handled by a different func. or with different parameters
    if load_models:
        # Assuming the file exists for each model that will be tested:
        # TODO: Catch loading errors in try-except blocks
        
        if train_focal:
            rn_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_focal_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_focal_{dataset}.pth")
        
        if train_sigmoid:
            rn_sigmoid.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_sigmoid_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} sigmoid, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_sigmoid_{dataset}.pth")
        
        if train_ce:
            rn_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_ce_{dataset}.pth")
        
        if train_cb_focal:
            rn_cb_focal.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth")
        
        if train_cb_sigmoid:
            rn_cb_sigmoid.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_sigmoid_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. sigmoid, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_sigmoid_{dataset}.pth")
        
        if train_cb_ce:
            rn_cb_ce.load_state_dict(
                torch.load(models_path + f"rn{resnet_type}_cb_ce_{dataset}.pth",
                           map_location=device)
            )
            print(f"Loaded model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                  models_path + f"rn{resnet_type}_cb_ce_{dataset}.pth")
    else:
        # Initialize FC biases of models trained with sigmoid and focal losses
        #   to avoid instability at the beginning of the training
        # TODO: Is this block required?
        pi = torch.tensor(0.1, dtype=torch.double)
        b = -torch.log((1 - pi) / pi)
        
        if multi_gpu:
            if train_focal: rn_focal.module.fc.bias.data.fill_(b)
            if train_sigmoid: rn_sigmoid.module.fc.bias.data.fill_(b)
            if train_cb_focal: rn_cb_focal.module.fc.bias.data.fill_(b)
            if train_cb_sigmoid: rn_cb_sigmoid.module.fc.bias.data.fill_(b)
        else:
            if train_focal: rn_focal.fc.bias.data.fill_(b)
            if train_sigmoid: rn_sigmoid.fc.bias.data.fill_(b)
            if train_cb_focal: rn_cb_focal.fc.bias.data.fill_(b)
            if train_cb_sigmoid: rn_cb_sigmoid.fc.bias.data.fill_(b)
        
        # TODO: Disable optimizer's weight decay for the biases (is this required?)
        #rn_focal.fc.bias.requires_grad_(False)
        #rn_sigmoid.fc.bias.requires_grad_(False)
        #rn_cb_focal.fc.bias.requires_grad_(False)
        #rn_cb_sigmoid.fc.bias.requires_grad_(False)
        
        # Initialize cross entropy loss models' FC biases with 0
        # TODO: How about FC biases of other models?
        if multi_gpu:
            if train_ce: rn_ce.module.fc.bias.data.fill_(0)
            if train_cb_ce: rn_cb_ce.module.fc.bias.data.fill_(0)
        else:
            if train_ce: rn_ce.fc.bias.data.fill_(0)
            if train_cb_ce: rn_cb_ce.fc.bias.data.fill_(0)
        
        optimizer = torch.optim.SGD(
            param_list,
            lr=0,  # Will be graudally increased to 0.1 in 5 epochs
            momentum=0.9,
            weight_decay=2e-4
        )
        
        if train_ce: cel = nn.CrossEntropyLoss()
        if train_cb_ce:
            #print("Passing weights:", weights)
            cb_cel = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        
        if draw_plots:
            if train_focal: history_loss_focal = []
            if train_sigmoid: history_loss_sigmoid = []
            if train_ce: history_loss_ce = []
            if train_cb_focal: history_loss_cb_focal = []
            if train_cb_sigmoid: history_loss_cb_sigmoid = []
            if train_cb_ce: history_loss_cb_ce = []
        
        print(f"Starting training with {DSET_NAMES[dataset]} dataset,",
              f"ResNet-{resnet_type} models.")
        try:
            for epoch in range(epoch_cnt):
                if train_focal: total_loss_focal = 0
                if train_sigmoid: total_loss_sigmoid = 0
                if train_ce: total_loss_ce = 0
                if train_cb_focal: total_loss_cb_focal = 0
                if train_cb_sigmoid: total_loss_cb_sigmoid = 0
                if train_cb_ce: total_loss_cb_ce = 0
                
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
                        loss_focal = loss_functions.focal_loss(
                            rn_focal(inp),
                            target,
                            gamma=0.5,
                            device=device
                        )
                        
                        loss_focal.backward()
                        total_loss_focal += loss_focal.item()
                    
                    if train_sigmoid:
                        loss_sigmoid = loss_functions.focal_loss(
                            rn_sigmoid(inp),
                            target,
                            device=device
                        )
                        
                        loss_sigmoid.backward()
                        total_loss_sigmoid += loss_sigmoid.item()
                    
                    if train_ce:
                        loss_ce = cel(
                            rn_ce(inp),
                            target
                        )
                        
                        loss_ce.backward()
                        total_loss_ce += loss_ce.item()
                    
                    if train_cb_focal:
                        loss_cb_focal = loss_functions.focal_loss(
                            rn_cb_focal(inp),
                            target,
                            alpha=weights,
                            gamma=0.5,
                            device=device
                        )
                        
                        loss_cb_focal.backward()
                        total_loss_cb_focal += loss_cb_focal.item()
                    
                    if train_cb_sigmoid:
                        loss_cb_sigmoid = loss_functions.focal_loss(
                            rn_cb_sigmoid(inp),
                            target,
                            alpha=weights,
                            device=device
                        )
                        
                        loss_cb_sigmoid.backward()
                        total_loss_cb_sigmoid += loss_cb_sigmoid.item()
                    
                    if train_cb_ce:
                        loss_cb_ce = cb_cel(
                            rn_cb_ce(inp),
                            target
                        ) / target.shape[0]
                        
                        loss_cb_ce.backward()
                        total_loss_cb_ce += loss_cb_ce.item()
                    
                    optimizer.step()
                    
                    if print_training and \
                            ((epoch == 0 and i == 0) or (i % print_freq == (print_freq - 1))):
                        print("Epoch:", epoch, "| Batch:", str(i + 1))
                        
                        if train_focal:
                            print("Focal:".rjust(11), total_loss_focal/(i+1))
                        if train_sigmoid:
                            print("Sigmoid:".rjust(11), total_loss_sigmoid/(i+1))
                        if train_ce:
                            print("Cross Entropy:".rjust(11), total_loss_ce/(i+1))
                        if train_cb_focal:
                            print("CB Focal:".rjust(11), total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid:
                            print("CB Sigmoid:", total_loss_cb_sigmoid/(i+1))
                        if train_cb_ce:
                            print("CB Cross Entropy:", total_loss_cb_ce/(i+1))
                        
                        print()  # Print empty line
                else:  # The end of each epoch
                    if save_models:  # Temporary backup for each epoch
                        # Delete all temporary files under temp/epoch_end/
                        for f in os.listdir(models_path + "temp/epoch_end/"):
                            fpath = models_path + "temp/epoch_end/" + f
                            os.remove(fpath)
                            #print("Removed:", fpath)
                        
                        if train_focal:
                            torch.save(
                                rn_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_sigmoid:
                            torch.save(
                                rn_sigmoid.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} sigmoid, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_ce:
                            torch.save(
                                rn_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_focal:
                            torch.save(
                                rn_cb_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_sigmoid:
                            torch.save(
                                rn_cb_sigmoid.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. sigmoid, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_ce:
                            torch.save(
                                rn_cb_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                    
                    if draw_plots:
                        if train_focal:
                            history_loss_focal.append(total_loss_focal/(i+1))
                        if train_sigmoid:
                            history_loss_sigmoid.append(total_loss_sigmoid/(i+1))
                        if train_ce:
                            history_loss_ce.append(total_loss_ce/(i+1))
                        if train_cb_focal:
                            history_loss_cb_focal.append(total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid:
                            history_loss_cb_sigmoid.append(total_loss_cb_sigmoid/(i+1))
                        if train_cb_ce:
                            history_loss_cb_ce.append(total_loss_cb_ce/(i+1))
                    
                    if print_training:
                        print("Epoch:", epoch, "| Batch:", str(i + 1))
                        
                        if train_focal:
                            print("Focal:".rjust(11), total_loss_focal/(i+1))
                        if train_sigmoid:
                            print("Sigmoid:".rjust(11), total_loss_sigmoid/(i+1))
                        if train_ce:
                            print("Cross Entropy:".rjust(11), total_loss_ce/(i+1))
                        if train_cb_focal:
                            print("CB Focal:".rjust(11), total_loss_cb_focal/(i+1))
                        if train_cb_sigmoid:
                            print("CB Sigmoid:", total_loss_cb_sigmoid/(i+1))
                        if train_cb_ce:
                            print("CB Cross Entropy:", total_loss_cb_ce/(i+1))
                        
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
                
                if train_focal:
                    torch.save(
                        rn_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_sigmoid:
                    torch.save(
                        rn_sigmoid.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} sigmoid, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_ce:
                    torch.save(
                        rn_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_focal:
                    torch.save(
                        rn_cb_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_sigmoid:
                    torch.save(
                        rn_cb_sigmoid.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. sigmoid, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_ce:
                    torch.save(
                        rn_cb_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
            
            print("Terminating.")
            sys.exit(1)
        
        
        # Save the trained models
        
        if save_models:
            if train_focal:
                torch.save(
                    rn_focal.state_dict(),
                    models_path + f"rn{resnet_type}_focal_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_focal_{dataset}.pth")
            
            if train_sigmoid:
                torch.save(
                    rn_sigmoid.state_dict(),
                    models_path + f"rn{resnet_type}_sigmoid_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} sigmoid, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_sigmoid_{dataset}.pth")
            
            if train_ce:
                torch.save(
                    rn_ce.state_dict(),
                    models_path + f"rn{resnet_type}_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_ce_{dataset}.pth")
            
            if train_cb_focal:
                torch.save(
                    rn_cb_focal.state_dict(),
                    models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth")
            
            if train_cb_sigmoid:
                torch.save(
                    rn_cb_sigmoid.state_dict(),
                    models_path + f"rn{resnet_type}_cb_sigmoid_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. sigmoid, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_sigmoid_{dataset}.pth")
            
            if train_cb_ce:
                torch.save(
                    rn_cb_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_ce_{dataset}.pth")
        
        if draw_plots:
            legend = []
            plt.figure(figsize=(16, 12))
            
            if train_focal:
                plt.plot(history_loss_focal, "-b")
                legend.append("Focal Loss")
            if train_sigmoid:
                plt.plot(history_loss_sigmoid, "-r")
                legend.append("Sigmoid CE Loss")
            if train_ce:
                plt.plot(history_loss_ce, "-g")
                legend.append("CE Loss")
            if train_cb_focal:
                plt.plot(history_loss_cb_focal, "-c")
                legend.append("Class-Balanced Focal Loss")
            if train_cb_sigmoid:
                plt.plot(history_loss_cb_sigmoid, "-m")
                legend.append("Class-Balanced Sigmoid CE Loss")
            if train_cb_ce:
                plt.plot(history_loss_cb_ce, "-y")
                legend.append("Class-Balanced CE Loss")
            
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
    
    # TODO: Return in a dict instead of a tuple for easier access to desired models
    return (rn_focal, rn_sigmoid, rn_ce, rn_cb_focal, rn_cb_sigmoid,
            rn_cb_ce)

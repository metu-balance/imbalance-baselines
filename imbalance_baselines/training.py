# TODO: Check out wandb

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


# TODO: Parameters look cluttered, need to simplify. Use a config. class?
#   Pass train & test preferences in a list instead of separate param.s.
#   Iterate over the list later to avoid code repetition.
#   TODO: This should also satisfy the main intention of the project, which is to provide a baseline for
#     experiments described using preference combinations. Just choose model architecture, dataset,
#     loss fn., sampling method... etc. and then program should take care of the rest. The current
#     structure may need to be modified (e.g it might not be suitable for multiple data transformation
#     methods, since it would require different iterations over the dataset for different experiments).
# TODO: Instead of save_models param, assume false if models_path == ""
#   TODO: At start of func., if save_models, create temporary dir. under model_path if it does not exist
# TODO: Add timestamp to saved models
def train_models(dataset: str, train_dl: DataLoader, class_cnt: int, weights: [float],
                 epoch_cnt: int = 200, multi_gpu: bool = False, device: torch.device = torch.device("cpu"),
                 resnet_type: str = "32", print_training: bool = True, print_freq: int = 100,
                 draw_plots: bool = False, models_path: str = "./trained_models/",
                 save_models: bool = False, load_models: bool = False, train_focal: bool = False,
                 train_sigmoid_ce: bool = False, train_softmax_ce: bool = False, train_cb_focal: bool = False,
                 train_cb_sigmoid_ce: bool = False, train_cb_softmax_ce: bool = False):
    
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
    rn_sigmoid_ce = None
    rn_softmax_ce = None
    rn_cb_focal = None
    rn_cb_sigmoid_ce = None
    rn_cb_softmax_ce = None
    
    # The "state" var. provides the same initial state for every model
    state = None
    
    if train_focal:
        rn_focal = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_focal = nn.DataParallel(rn_focal)
        rn_focal = rn_focal.to(device)
        
        param_list.append({'params': rn_focal.parameters()})
        
        state = rn_focal.state_dict()
    if train_sigmoid_ce:
        rn_sigmoid_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_sigmoid_ce = nn.DataParallel(rn_sigmoid_ce)
        rn_sigmoid_ce = rn_sigmoid_ce.to(device)
        
        param_list.append({'params': rn_sigmoid_ce.parameters()})
        
        if state:
            rn_sigmoid_ce.load_state_dict(state)
        else:
            state = rn_sigmoid_ce.state_dict()
    if train_softmax_ce:
        rn_softmax_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_softmax_ce = nn.DataParallel(rn_softmax_ce)
        rn_softmax_ce = rn_softmax_ce.to(device)
        
        param_list.append({'params': rn_softmax_ce.parameters()})
        
        if state:
            rn_softmax_ce.load_state_dict(state)
        else:
            state = rn_softmax_ce.state_dict()
    if train_cb_focal:
        rn_cb_focal = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_focal = nn.DataParallel(rn_cb_focal)
        rn_cb_focal = rn_cb_focal.to(device)
        
        param_list.append({'params': rn_cb_focal.parameters()})
        
        if state:
            rn_cb_focal.load_state_dict(state)
        else:
            state = rn_cb_focal.state_dict()
    if train_cb_sigmoid_ce:
        rn_cb_sigmoid_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_sigmoid_ce = nn.DataParallel(rn_cb_sigmoid_ce)
        rn_cb_sigmoid_ce = rn_cb_sigmoid_ce.to(device)
        
        param_list.append({'params': rn_cb_sigmoid_ce.parameters()})
        
        if state:
            rn_cb_sigmoid_ce.load_state_dict(state)
        else:
            state = rn_cb_sigmoid_ce.state_dict()
    if train_cb_softmax_ce:
        rn_cb_softmax_ce = rn(num_classes=class_cnt).double()
        if multi_gpu: rn_cb_softmax_ce = nn.DataParallel(rn_cb_softmax_ce)
        rn_cb_softmax_ce = rn_cb_softmax_ce.to(device)
        
        param_list.append({'params': rn_cb_softmax_ce.parameters()})
        
        if state:
            rn_cb_softmax_ce.load_state_dict(state)
    
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
    else:
        # Initialize FC biases of models trained with sigmoid_ce and focal losses
        #   to avoid instability at the beginning of the training
        # TODO: Is this block required?
        pi = torch.tensor(0.1, dtype=torch.double)
        b = -torch.log((1 - pi) / pi)
        
        if multi_gpu:
            if train_focal: rn_focal.module.fc.bias.data.fill_(b)
            if train_sigmoid_ce: rn_sigmoid_ce.module.fc.bias.data.fill_(b)
            if train_cb_focal: rn_cb_focal.module.fc.bias.data.fill_(b)
            if train_cb_sigmoid_ce: rn_cb_sigmoid_ce.module.fc.bias.data.fill_(b)
        else:
            if train_focal: rn_focal.fc.bias.data.fill_(b)
            if train_sigmoid_ce: rn_sigmoid_ce.fc.bias.data.fill_(b)
            if train_cb_focal: rn_cb_focal.fc.bias.data.fill_(b)
            if train_cb_sigmoid_ce: rn_cb_sigmoid_ce.fc.bias.data.fill_(b)
        
        # TODO: Disable optimizer's weight decay for the biases (is this required?)
        #rn_focal.fc.bias.requires_grad_(False)
        #rn_sigmoid_ce.fc.bias.requires_grad_(False)
        #rn_cb_focal.fc.bias.requires_grad_(False)
        #rn_cb_sigmoid_ce.fc.bias.requires_grad_(False)
        
        # Initialize cross entropy loss models' FC biases with 0
        # TODO: How about FC biases of other models?
        if multi_gpu:
            if train_softmax_ce: rn_softmax_ce.module.fc.bias.data.fill_(0)
            if train_cb_softmax_ce: rn_cb_softmax_ce.module.fc.bias.data.fill_(0)
        else:
            if train_softmax_ce: rn_softmax_ce.fc.bias.data.fill_(0)
            if train_cb_softmax_ce: rn_cb_softmax_ce.fc.bias.data.fill_(0)
        
        optimizer = torch.optim.SGD(
            param_list,
            lr=0,  # Will be graudally increased to 0.1 in 5 epochs
            momentum=0.9,
            weight_decay=2e-4
        )
        
        if train_softmax_ce:
            cel = nn.CrossEntropyLoss()
        if train_cb_softmax_ce:
            cb_softmax_cel = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        if train_sigmoid_ce or train_focal or train_cb_sigmoid_ce or train_cb_focal:
            focal_loss = FocalLoss(device=device)
        
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
                        
                        if train_sigmoid_ce:
                            torch.save(
                                rn_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_softmax_ce:
                            torch.save(
                                rn_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_focal:
                            torch.save(
                                rn_cb_focal.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_sigmoid_ce:
                            torch.save(
                                rn_cb_sigmoid_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                        
                        if train_cb_softmax_ce:
                            torch.save(
                                rn_cb_softmax_ce.state_dict(),
                                models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                            )
                            #print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                            #      models_path + f"temp/epoch_end/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                    
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
                
                if train_focal:
                    torch.save(
                        rn_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_sigmoid_ce:
                    torch.save(
                        rn_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_softmax_ce:
                    torch.save(
                        rn_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_focal:
                    torch.save(
                        rn_cb_focal.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_focal_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_sigmoid_ce:
                    torch.save(
                        rn_cb_sigmoid_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_sigmoid_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
                
                if train_cb_softmax_ce:
                    torch.save(
                        rn_cb_softmax_ce.state_dict(),
                        models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth"
                    )
                    print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                          models_path + f"temp/interrupted/rn{resnet_type}_cb_softmax_ce_{dataset}_epoch{epoch}_batch{i+1}.pth")
            
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
            
            if train_sigmoid_ce:
                torch.save(
                    rn_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_sigmoid_ce_{dataset}.pth")
            
            if train_softmax_ce:
                torch.save(
                    rn_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_softmax_ce_{dataset}.pth")
            
            if train_cb_focal:
                torch.save(
                    rn_cb_focal.state_dict(),
                    models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. focal, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_focal_{dataset}.pth")
            
            if train_cb_sigmoid_ce:
                torch.save(
                    rn_cb_sigmoid_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. sigmoid_ce, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_sigmoid_ce_{dataset}.pth")
            
            if train_cb_softmax_ce:
                torch.save(
                    rn_cb_softmax_ce.state_dict(),
                    models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth"
                )
                print(f"Saved model (ResNet-{resnet_type} cb. cross entropy, {DSET_NAMES[dataset]}):",
                      models_path + f"rn{resnet_type}_cb_softmax_ce_{dataset}.pth")
        
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
    
    # TODO: Return in a dict instead of a tuple for easier access to desired models
    return (rn_focal, rn_sigmoid_ce, rn_softmax_ce, rn_cb_focal, rn_cb_sigmoid_ce,
            rn_cb_softmax_ce)

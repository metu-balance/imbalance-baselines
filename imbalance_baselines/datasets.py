import datetime as dt
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from numpy.random import Generator, PCG64
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Callable, Optional
from . import sampling
from . import DSET_NAMES, TRANSFORMATIONS


class CIFAR10LT(datasets.CIFAR10):
    cls_cnt = 10
    
    def __init__(
            self,
            root: str,
            imb_factor=100,  # largest cls. / smallest cls. sample counts
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            sampler=None
    ) -> None:
        
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            download
        )
        
        self.cnt_per_cls_dict = dict()
        self.rng = Generator(PCG64())
        self.sampler = sampler
        
        img_num_list = self.get_img_cnt_per_cls(self.cls_cnt, imb_factor)
        self.generate_imb_data(img_num_list)
    
    def get_img_cnt_per_cls(self, cls_cnt, imb_factor):
        """Return the image count per class required to create class imbalance.
    
        Args:
          cls_cnt: Number of classes
          imb_factor: Imbalance factor (Largest / Smallest class sizes)
        """
        
        img_max = len(self.data) / cls_cnt
        img_cnt_per_cls = []
        
        for cls_index in range(cls_cnt):
            img_cnt_per_cls.append(int(
                img_max * ((1/imb_factor)**(cls_index / (cls_cnt - 1)))
            ))
        
        return img_cnt_per_cls
    
    def generate_imb_data(self, img_cnt_per_cls):
        new_data = []
        new_targets = []
        
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        if self.sampler is None:
            for cls, img_cnt in zip(classes, img_cnt_per_cls):
                self.cnt_per_cls_dict[cls] = img_cnt
                i = np.where(targets_np == cls)[0]
                
                self.rng.shuffle(i)
                selected_i = i[:img_cnt]
                
                new_data.append(self.data[selected_i, ...])
                new_targets.extend([cls] * img_cnt)
        elif isinstance(self.sampler, sampling.OnlineSampler):
            raise ValueError(
                "Online sampling method supplied to CIFAR10LT, only offline methods are accepted. Use online "
                "methods with DataLoader using the sampler parameter."
            )
        else:
            # TODO: Ensure that sampler's argument is of form [(feature, label), (feature, label), ...]
            imb_data = self.sampler([(img, self.targets[i]) for i, img in enumerate(self.data)])
            
            for i in imb_data:
                # Note that samplers are expected to take and return data as a list of (feature, label) tuples
                new_data.append(i[0])
                new_targets.append(i[1])
        
        new_data = np.vstack(new_data)
        
        self.data = new_data
        self.targets = new_targets
    
    def get_cls_cnt_list(self):
        """Return the current (imbalanced) image count per class."""
        cls_cnt_list = []
        
        for i in range(self.cls_cnt):
            cls_cnt_list.append(self.cnt_per_cls_dict[i])
        
        return cls_cnt_list


# TODO: Add sampler support
class INaturalist(Dataset):
    def __init__(
            self,
            root: str,
            annotations: str,
            version: str = "2017",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if version not in ["2017", "2018"]:
            raise RuntimeError("version argument must be either '2017' or '2018'.")
        
        self.root = root
        self.version = version
        self.transform = transform
        self.target_transform = target_transform
        
        self.class_cnt = 5089 if version == "2017" else 8142
        
        self.to_tensor = transforms.ToTensor()
        
        with open(annotations) as f:
            ann_data = json.load(f)
        
        self.imgs = [a["file_name"] for a in ann_data["images"]]
        
        # self.classes holds the label of each image in the corresponding index
        if "annotations" in ann_data.keys():
            self.classes = [a["category_id"] for a in ann_data["annotations"]]
        else:
            # If not given, set class labels to 0
            self.classes = [0] * len(self.imgs)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if self.root.endswith("/"):
            path = self.root + self.imgs[idx]
        else:
            path = self.root + "/" + self.imgs[idx]
        
        img = Image.open(path).convert("RGB")
        
        target = self.classes[idx]  # label of the image (i.e. species ID)
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.to_tensor(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_cls_cnt_list(self):
        cls_cnt_list = [0] * self.class_cnt
        
        for i in self.classes:
            cls_cnt_list[i] += 1
        
        return cls_cnt_list


def generate_data(cfg, sampler=None):
    # Parse configuration
    dataset_cfg = cfg["Dataset"]
    datasets_path = dataset_cfg["datasets_path"]
    dataset_name = dataset_cfg["dataset_name"]
    dataset_params = dataset_cfg["dataset_params"]

    normalize_mu = dataset_params["normalize_mu"]
    normalize_std = dataset_params["normalize_std"]
    
    datagen_cfg = cfg["DataGeneration"]
    batch_size = datagen_cfg["batch_size"]
    pad = datagen_cfg["pad"]
    img_size = datagen_cfg["image_size"]
    train_shuffle = datagen_cfg["train_shuffle"]
    draw_dataset_plots = datagen_cfg["plotting"]["draw_dataset_plots"]
    plot_size = datagen_cfg["plotting"]["plot_size"]
    plot_path = datagen_cfg["plotting"]["plot_path"]
    
    # False if sampler is None or is offline
    sampler_is_online = isinstance(sampler, sampling.OnlineSampler) and (sampler is not None)
    
    if not datasets_path.endswith("/"):
        datasets_path += "/"
    
    train_transforms = []
    test_transforms = []
    
    # TODO: Avoid code repetition below
    for tr in datagen_cfg["train_transform"]:
        tr_name = tr["transform_name"]
        tr_params = tr["transform_params"]
        
        tr_class = TRANSFORMATIONS[tr_name]
        if tr_class == transforms.Pad:
            train_transforms.append(tr_class(padding=pad, fill=tr_params["fill"], mode=tr_params["mode"]))
        elif tr_class == transforms.ColorJitter:
            jtr_brightness = dataset_params["jitter_brightness"]
            jtr_contrast = dataset_params["jitter_contrast"]
            jtr_saturation = dataset_params["jitter_saturation"]
            jtr_hue = dataset_params["jitter_hue"]
            
            train_transforms.append(tr_class(jtr_brightness, jtr_contrast, jtr_saturation, jtr_hue))
        elif tr_class in [transforms.RandomResizedCrop, transforms.CenterCrop]:
            train_transforms.append(tr_class(img_size))
        else:
            train_transforms.append(tr_class())

    train_transforms.append(transforms.ToTensor())
    # TODO: Should test normalization be in place for train set?
    train_transforms.append(transforms.Normalize(mean=normalize_mu, std=normalize_std, inplace=True))
    
    for tr in datagen_cfg["test_transform"]:
        tr_name = tr["transform_name"]
        tr_params = tr["transform_params"]
    
        tr_class = TRANSFORMATIONS[tr_name]
        if tr_class == transforms.Pad:
            test_transforms.append(tr_class(padding=pad, fill=tr_params["fill"], mode=tr_params["mode"]))
        elif tr_class == transforms.ColorJitter:
            jtr_brightness = dataset_params["jitter_brightness"]
            jtr_contrast = dataset_params["jitter_contrast"]
            jtr_saturation = dataset_params["jitter_saturation"]
            jtr_hue = dataset_params["jitter_hue"]
        
            test_transforms.append(tr_class(jtr_brightness, jtr_contrast, jtr_saturation, jtr_hue))
        elif tr_class in [transforms.RandomResizedCrop, transforms.CenterCrop]:
            test_transforms.append(tr_class(img_size))
        else:
            test_transforms.append(tr_class())

    test_transforms.append(transforms.ToTensor())
    # TODO: Should test normalization be done for test set? Should it be in place?
    test_transforms.append(transforms.Normalize(mean=normalize_mu, std=normalize_std))
    
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)
    
    if dataset_name == "CIFAR10":
        train_ds = datasets.CIFAR10(
            datasets_path + "cifar10",
            train=True,
            download=True,
            transform=train_transforms
        )
        
        test_ds = datasets.CIFAR10(
            datasets_path + "cifar10",
            train=False,
            download=True,
            transform=test_transforms
        )
    elif dataset_name == "IMB_CIFAR10":  # Long-Tailed CIFAR10
        train_ds = CIFAR10LT(
            datasets_path + "cifar10",
            imb_factor=dataset_params["cifar10_imb_factor"],
            train=True,
            download=True,
            transform=train_transforms,
            sampler=None if sampler_is_online else sampler
        )
        
        test_ds = datasets.CIFAR10(  # Test set is not imbalanced
            datasets_path + "cifar10",
            train=False,
            download=True,
            transform=test_transforms
        )
    elif dataset_name == "INATURALIST_2017":
        train_ds = INaturalist(
            datasets_path + "inat2017",
            "datasets/inat2017/train2017.json",
            version="2017",
            transform=train_transforms
        )
        
        test_ds = INaturalist(
            datasets_path + "inat2017/test2017",
            "datasets/inat2017/test2017.json",
            version="2017",
            transform=test_transforms
        )
    elif dataset_name == "INATURALIST_2018":
        train_ds = INaturalist(
            datasets_path + "inat2018",
            "datasets/inat2018/train2018.json",
            version="2018",
            transform=train_transforms
        )
        
        test_ds = INaturalist(
            datasets_path + "inat2018",
            "datasets/inat2018/test2018.json",
            version="2018",
            transform=test_transforms
        )
    else:
        raise ValueError("The given dataset name is not recognized.")
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False if sampler_is_online or not train_shuffle else True,
        sampler=sampler if sampler_is_online else None
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=2
    )
    
    if dataset_name == "CIFAR10":
        class_cnt = 10
        train_class_sizes = [5000] * class_cnt
        test_class_sizes = [1000] * class_cnt
    elif dataset_name == "IMB_CIFAR10":
        class_cnt = 10
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = [1000] * class_cnt
    elif dataset_name == "INATURALIST_2017":
        class_cnt = 5089
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = test_ds.get_cls_cnt_list()
    elif dataset_name == "INATURALIST_2018":
        class_cnt = 8142
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = test_ds.get_cls_cnt_list()
    else:
        raise ValueError("The given dataset name is not recognized.")
    
    print("Number of training samples:")
    print(np.array(train_class_sizes))  # Numpy array for cleaner output
    
    if draw_dataset_plots:
        x = np.arange(class_cnt)
        fig, axs = plt.subplots(2, constrained_layout=True)

        fig.set_figwidth(plot_size[0])
        fig.set_figheight(plot_size[1])
        
        fig.suptitle('Class Sizes & Weights for ' + DSET_NAMES[dataset_name])
        
        axs[0].set_title('Size Per Class (Unsorted)')
        axs[0].plot(x, train_class_sizes)
        
        axs[1].set_title('Size Per Class (Sorted)')
        axs[1].plot(x, sorted(train_class_sizes, reverse=True))

        tstamp = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        plt.savefig(plot_path + f"{dataset_name.lower()}-size-per-class-" + tstamp + ".png")
        
        plt.show()
    
    return train_dl, test_dl, class_cnt, train_class_sizes, test_class_sizes



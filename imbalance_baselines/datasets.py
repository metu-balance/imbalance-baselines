import datetime as dt
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Callable, Optional
from . import sampling
from .dataset import DSET_NAMES, DSET_CLASS_CNTS
from . import logger, get_global_seed
from .utils import parse_cfg_str, seed_everything


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
        self.rng = np.random.default_rng(seed=get_global_seed())
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
        if imb_factor == 0:
            return img_max
        
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


# TODO: Add sampler support to INaturalist
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


def seed_worker(worker_id):
    seed_everything(get_global_seed())


def generate_data(cfg):
    # Parse configuration
    dataset_cfg = cfg.Dataset
    datasets_path = dataset_cfg.datasets_path
    dataset_name = dataset_cfg.dataset_name
    dataset_params = dataset_cfg.dataset_params

    normalize_mu = (
        parse_cfg_str(dataset_params.normalize_mu.val1, float),
        parse_cfg_str(dataset_params.normalize_mu.val2, float),
        parse_cfg_str(dataset_params.normalize_mu.val3, float),
    )
    normalize_std = (
        parse_cfg_str(dataset_params.normalize_std.val1, float),
        parse_cfg_str(dataset_params.normalize_std.val2, float),
        parse_cfg_str(dataset_params.normalize_std.val3, float),
    )
    
    datagen_cfg = cfg.DataGeneration
    batch_size = parse_cfg_str(datagen_cfg.batch_size, int)
    worker_count = parse_cfg_str(datagen_cfg.num_workers, int)
    pad = parse_cfg_str(datagen_cfg.pad, int)
    img_size = (
        parse_cfg_str(datagen_cfg.image_size.width, int),
        parse_cfg_str(datagen_cfg.image_size.height, int)
    )
    train_shuffle = datagen_cfg.train_shuffle
    
    plot_cfg = datagen_cfg.plotting
    draw_dataset_plots = plot_cfg.draw_dataset_plots
    if draw_dataset_plots:
        plot_size = (
            parse_cfg_str(plot_cfg.plot_size.width, int),
            parse_cfg_str(plot_cfg.plot_size.height, int)
        )
        plot_path = plot_cfg.plot_path
    
    class_count = DSET_CLASS_CNTS[dataset_name]
    sampler = parse_cfg_str(datagen_cfg.sampler, None)
    
    if sampler is not None:
        # Initialize sampler if offline
        sampler_name = datagen_cfg.sampler.sampler_name
        sampler_params = datagen_cfg.sampler.sampler_params

        # TODO: Generalize / deprecate
        if sampler_name == "oversampler":
            sampler = sampling.OverSampler(class_count, sampler_params.ratio)
            sampler_is_online = False
        elif sampler_name == "undersampler":
            sampler = sampling.UnderSampler(class_count, sampler_params.ratio)
            sampler_is_online = False
        else:  # It is known that a valid online sampler name is provided by the config.
            sampler_is_online = True
    else:
        sampler_is_online = False
    
    if not datasets_path.endswith("/"):
        datasets_path += "/"
    
    def form_transf(dg_cfg, is_train):
        """Form a sequence of data transformations to be applied.

        :param dg_cfg: Data generation configuration
        """

        transf_list = []

        transf_cfg_list = dg_cfg["train_transform" if is_train else "test_transform"]
        if transf_cfg_list is None:
            transf_cfg_list = []

        for transf in transf_cfg_list:
            tr_name = transf.transform_name
            tr_params = transf.transform_params

            # TODO: Generalize
            if tr_name == "pad":
                transf_list.append(
                    transforms.Pad(
                        padding=pad,
                        fill=parse_cfg_str(tr_params.fill, int),
                        padding_mode=tr_params.mode))
            elif tr_name == "color_jitter":
                jtr_brightness = parse_cfg_str(dataset_params.jitter_brightness, float)
                jtr_contrast = parse_cfg_str(dataset_params.jitter_contrast, float)
                jtr_saturation = parse_cfg_str(dataset_params.jitter_saturation, float)
                jtr_hue = parse_cfg_str(dataset_params.jitter_hue, float)
        
                transf_list.append(transforms.ColorJitter(jtr_brightness, jtr_contrast, jtr_saturation, jtr_hue))
            elif tr_name == "random_resized_crop":
                transf_list.append(transforms.RandomResizedCrop(img_size))
            elif tr_name == "center_crop":
                transf_list.append(transforms.CenterCrop(img_size))
            elif tr_name == "random_horizontal_flip":
                transf_list.append(transforms.RandomHorizontalFlip())
            else:
                raise ValueError("Unrecognized transformation name: " + tr_name)
    
        transf_list.append(transforms.ToTensor())
        transf_list.append(transforms.Normalize(mean=normalize_mu, std=normalize_std, inplace=is_train))

        return transforms.Compose(transf_list)

    # NOTE: Both lists contain normalization at the end, applying normalization both for train and test sets
    train_transforms = form_transf(datagen_cfg, is_train=True)
    test_transforms = form_transf(datagen_cfg, is_train=False)

    # TODO: Generalize
    if dataset_name == "CIFAR10":
        train_ds = CIFAR10LT(
            datasets_path + "cifar10",
            imb_factor=0,
            train=True,
            download=True,
            transform=train_transforms,
            sampler=None if sampler_is_online else sampler  # Offline samp. or None
        )
        
        test_ds = datasets.CIFAR10(
            datasets_path + "cifar10",
            train=False,
            download=True,
            transform=test_transforms
        )
    elif dataset_name == "CIFAR10LT":  # Long-Tailed CIFAR10
        train_ds = CIFAR10LT(
            datasets_path + "cifar10",
            imb_factor=parse_cfg_str(dataset_params.imb_factor, int),
            train=True,
            download=True,
            transform=train_transforms,
            sampler=None if sampler_is_online else sampler  # Offline samp. or None
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
    
    if sampler_is_online:  # It is known that sampler is not None and is online
        # Initialize online sampler
        sampler_name = datagen_cfg.sampler.sampler_name
        sampler_params = datagen_cfg.sampler.sampler_params

        # TODO: Generalize / deprecate
        if sampler_name == "cb_sampler":
            sampler = sampling.ClassBalancedSampling(train_ds, class_count, sampler_params.q_value)
        elif sampler_name == "progb_sampler":
            sampler = sampling.ProgressivelyBalancedSampling(train_ds, class_count, sampler_params.total_epochs)
        else:
            raise ValueError("Invalid sampler name encountered: " + sampler_name)

    generator = torch.Generator()
    generator.manual_seed(get_global_seed())

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=worker_count,
        shuffle=False if sampler_is_online or not train_shuffle else True,
        sampler=sampler if sampler_is_online else None,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=worker_count,
        worker_init_fn=seed_worker,
        generator=generator
    )

    # TODO: Generalize
    if dataset_name == "CIFAR10":
        train_class_sizes = [5000] * class_count
        test_class_sizes = [1000] * class_count
    elif dataset_name == "CIFAR10LT":
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = [1000] * class_count
    elif dataset_name == "INATURALIST_2017":
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = test_ds.get_cls_cnt_list()
    elif dataset_name == "INATURALIST_2018":
        train_class_sizes = train_ds.get_cls_cnt_list()
        test_class_sizes = test_ds.get_cls_cnt_list()
    else:
        raise ValueError("The given dataset name is not recognized.")
    
    logger.info("Number of training samples:")
    logger.info(np.array(train_class_sizes))  # Numpy array for cleaner output
    
    if draw_dataset_plots:
        x = np.arange(class_count)
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
        
        # plt.show()
    
    return train_dl, test_dl,\
        {"class_count": class_count, "train_class_sizes": train_class_sizes, "test_class_sizes": test_class_sizes}


def get_cb_weights(class_sizes, beta=0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Get normalized weight (inverse of effective number of samples) per class."""

    class_sizes = torch.as_tensor(
        class_sizes,
        #dtype=torch.long,
        device=device
    )

    class_cnt = class_sizes.shape[0]

    weights = torch.as_tensor(
        [1 - beta] * class_cnt,
        device=device
    )

    weights = torch.div(
        weights, 1 - torch.pow(beta, class_sizes)
    ).to(device)

    # Normalize the cb_weights
    weights = torch.mul(weights, class_cnt / torch.sum(weights))

    weights.requires_grad = False

    return weights.to(device)

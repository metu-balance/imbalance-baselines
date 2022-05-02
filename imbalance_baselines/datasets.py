# TODO: Fully incorporate or remove Google Drive functionality

import json
import numpy as np
import matplotlib.pyplot as plt
from . import sampling
import torch

from numpy.random import RandomState, SeedSequence, MT19937
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Callable, Optional

# TODO: Find a better place shared by all submodules to define this (__init__.py?)
DSET_NAMES = {
  "CIFAR10": "CIFAR10",
  "IMB_CIFAR10": "Long-Tailed CIFAR10",
  "INATURALIST_2017": "iNaturalist 2017",
  "INATURALIST_2018": "iNaturalist 2018"
}


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
    self.rs = RandomState(MT19937(SeedSequence()))
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
        
        self.rs.shuffle(i)
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

    for i in self.classes: cls_cnt_list[i] += 1

    return cls_cnt_list


# TODO: Parameters look cluttered, may simplify
def generate_data(batch_size: int, dataset: str, datasets_path: str, inat_32x32: bool = False, draw_plots: bool = False,
                  use_gdrive: bool = False, cifar_imb_factor: int = 100, sampler=None,
                  device: torch.device = torch.device("cpu")):

  # False if sampler is None or is offline
  sampler_is_online = isinstance(sampler, sampling.OnlineSampler) and (sampler is not None)
  
  if not datasets_path.endswith("/"): datasets_path += "/"
  
  # TODO: Find better jitter, mu and std. values for each dataset
  if dataset == "CIFAR10":
    normalize_mu = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2023, 0.1994, 0.2010)

    im_size = 32
    pad = 4

    #jtr_brightness = 0.4
    #jtr_contrast = 0.4
    #jtr_saturation = 0.4
    #jtr_hue = 0.25
  elif dataset == "IMB_CIFAR10":
    normalize_mu = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)

    im_size = 32
    pad = 4

    #jtr_brightness = 0.4
    #jtr_contrast = 0.4
    #jtr_saturation = 0.4
    #jtr_hue = 0.25
  elif dataset == "INATURALIST_2017":
    normalize_mu = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)

    im_size = 224
    pad = 32

    #jtr_brightness = 0.4
    #jtr_contrast = 0.4
    #jtr_saturation = 0.4
    #jtr_hue = 0.25
  elif dataset == "INATURALIST_2018":
    normalize_mu = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    im_size = 224
    pad = 32

    #jtr_brightness = 0.4
    #jtr_contrast = 0.4
    #jtr_saturation = 0.4
    #jtr_hue = 0.25
  else:
    raise ValueError("The given dataset name is not recognized.")
  
  train_transforms = transforms.Compose([
    transforms.Pad(padding=pad, fill=0, padding_mode="constant"),
    transforms.RandomResizedCrop(im_size),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(jtr_brightness, jtr_contrast, jtr_saturation, jtr_hue),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mu, std=normalize_std, inplace=True)
  ])

  test_transforms = transforms.Compose([
    transforms.CenterCrop(im_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mu, std=normalize_std)  # TODO: Should test normalization be in place too?
  ])

  if dataset == "CIFAR10":
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
  elif dataset == "IMB_CIFAR10":  # Long-Tailed CIFAR10
    train_ds = CIFAR10LT(
      datasets_path + "cifar10",
      imb_factor=cifar_imb_factor,
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
  elif dataset == "INATURALIST_2017":
    train_ds = INaturalist(
        datasets_path + "inat2017",
        "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2017/train2017.json" if use_gdrive else "datasets/inat2017/train2017.json",
        version="2017",
        transform=train_transforms
    )

    test_ds = INaturalist(
        datasets_path + "inat2017/test2017",
        "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2017/test2017.json" if use_gdrive else "datasets/inat2017/test2017.json",
        version="2017",
        transform=test_transforms
    )

    """
    # To use pre-transformed datasets:
    if use_gdrive:
      inat_path =  "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2017_transf" + ("_32/" if inat_32x32 else "")
    else:
      inat_path =  datasets_path + "inat2017_transf" + ("_32/" if inat_32x32 else "/")
    
    train_ds = INaturalist(
        inat_path,
        inat_path + "train2017.json",
        version="2017"
    )

    test_ds = INaturalist(
        inat_path + "test2017/",
        inat_path + "test2017.json",
        version="2017"
    )
    """
  elif dataset == "INATURALIST_2018":
    train_ds = INaturalist(
        datasets_path + "inat2018",
        "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2018/train2018.json" if use_gdrive else "datasets/inat2018/train2018.json",
        version="2018",
        transform=train_transforms
    )

    test_ds = INaturalist(
        datasets_path + "inat2018",
        "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2018/test2018.json" if use_gdrive else "datasets/inat2018/test2018.json",
        version="2018",
        transform=test_transforms
    )

    """
    # To use pre-transformed datasets:
    if use_gdrive:
      inat_path =  "/content/drive/MyDrive/Colab Notebooks/cb_loss_resnet/datasets/inat2018_transf" + ("_32/" if inat_32x32 else "")
    else:
      inat_path =  datasets_path + "inat2018_transf" + ("_32/" if inat_32x32 else "/")
    
    train_ds = INaturalist(
        inat_path,
        inat_path + "train2018.json",
        version="2018"
    )

    test_ds = INaturalist(
        inat_path,
        inat_path + "test2018.json",
        version="2018"
    )
    """
  else:
    raise ValueError("The given dataset name is not recognized.")

  if sampler and not sampler_is_online:
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=2,
        sampler=sampler if not sampler_is_online else None
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=2
    )

  class_cnt = 0

  if dataset == "CIFAR10":
    class_cnt = 10
    train_class_sizes = [5000] * class_cnt
    test_class_sizes = [1000] * class_cnt
  elif dataset == "IMB_CIFAR10":
    class_cnt = 10
    train_class_sizes = train_ds.get_cls_cnt_list()    
    test_class_sizes = [1000] * class_cnt
  elif dataset == "INATURALIST_2017":
    class_cnt = 5089
    train_class_sizes = train_ds.get_cls_cnt_list()    
    test_class_sizes = test_ds.get_cls_cnt_list()
  elif dataset == "INATURALIST_2018":
    class_cnt = 8142
    train_class_sizes = train_ds.get_cls_cnt_list()    
    test_class_sizes = test_ds.get_cls_cnt_list()
  else:
    raise ValueError("The given dataset name is not recognized.")
  
  print("Number of training samples:")
  print(np.array(train_class_sizes))  # Numpy array for cleaner output

  if draw_plots:
    x = np.arange(class_cnt)
    fig, axs = plt.subplots(2, constrained_layout=True)
    
    fig.set_figheight(12)
    fig.set_figwidth(16)

    fig.suptitle('Class Sizes & Weights for ' + DSET_NAMES[dataset])
    
    axs[0].set_title('Size Per Class (Unsorted)')
    axs[0].plot(x, train_class_sizes)
    
    axs[1].set_title('Size Per Class (Sorted)')
    axs[1].plot(x, sorted(train_class_sizes, reverse=True))
    
    if not use_gdrive:
      plt.savefig(f"./plots/{dataset.lower()}_size_per_class.png")
    
    plt.show()

  return train_dl, test_dl, class_cnt, train_class_sizes, test_class_sizes



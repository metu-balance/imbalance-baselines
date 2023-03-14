import numpy as np

from torchvision import datasets
from typing import Callable, Optional
# import dataset_utils as sampling  # TODO rename? #Yes, rename.
from .. import get_global_seed


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
                img_max * ((1 / imb_factor) ** (cls_index / (cls_cnt - 1)))
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
            imb_data = self.sampler([(img, self.targets[i])
                                    for i, img in enumerate(self.data)])

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

import torch
import numpy as np

from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from .. import get_global_seed


class OnlineSampler(Sampler[int]):
    def __init__(self, dataset, num_classes: int):
        # TODO: Call to super class init needed or not?
        self.num_classes = num_classes

        self.index_groups = self.get_index_groups(dataset)
        self.sizes = torch.tensor(
            [len(group) for group in self.index_groups], dtype=torch.int64)
        self.length = self.sizes.sum()
        self.rng = np.random.default_rng(seed=get_global_seed())

    def __len__(self):
        return self.length

    def get_index_groups(self, dataset):
        # This function assumes the dataset to be in the form of
        # [(Feature_1, Label_1), (Feature_2, Label_2), .....]

        num_classes = self.num_classes
        groups = []

        for _ in range(num_classes):
            groups.append([])

        for i, (feature, label) in enumerate(dataset):
            groups[label].append(i)

        return groups


class ClassBalancedSampling(OnlineSampler):
    def __init__(self, dataset, num_classes: int, q_value: float = 0.0):
        OnlineSampler.__init__(self, dataset, num_classes)

        self.q_val = q_value
        self.probs = (self.sizes ** q_value) / (self.sizes ** q_value).sum()

    def __iter__(self):
        self.classes = torch.multinomial(
            self.probs, self.length, replacement=True)

        for i in range(self.length):
            random_class = self.classes[i]
            curr_group = self.index_groups[random_class]
            sample = self.rng.choice(curr_group)

            yield sample


class ClassBalancedDataloader(DataLoader):
    def __init__(self, dataset, num_classes: int, q_value: float = 0.0, **kwargs) -> None:
        self.classBalancedSampler = ClassBalancedSampling(
            dataset=dataset, num_classes=num_classes, q_value=q_value)
        super().__init__(dataset=dataset, sampler=self.classBalancedSampler, **kwargs)

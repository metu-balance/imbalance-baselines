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


class ProgressivelyBalancedSampling(OnlineSampler):
    def __init__(self, dataset, num_classes: int, total_epochs: int):
        OnlineSampler.__init__(self, dataset, num_classes)

        self.class_ratios = self.sizes / self.length
        self.curr_epoch = 0
        self.total_epochs = total_epochs

    def __iter__(self):
        self.probs = (1 - self.curr_epoch / self.total_epochs) * self.class_ratios + (
            self.curr_epoch / self.total_epochs) * (1 / self.num_classes)
        self.classes = torch.multinomial(
            self.probs, self.length, replacement=True)
        self.curr_epoch = (self.curr_epoch + 1) % self.total_epochs

        print("Probs: ", self.probs)

        for i in range(self.length):
            random_class = self.classes[i]
            curr_group = self.index_groups[random_class]
            sample = self.rng.choice(curr_group)

            yield sample


class ProgressivelyBalancedDataloader(DataLoader):
    def __init__(self, dataset, num_classes: int, total_epochs: int, **kwargs):
        self.ProgressiveSampler = ProgressivelyBalancedSampling(
            dataset=dataset, num_classes=num_classes, total_epochs=total_epochs)
        super().__init__(dataset=dataset, sampler=self.ProgressiveSampler, **kwargs)

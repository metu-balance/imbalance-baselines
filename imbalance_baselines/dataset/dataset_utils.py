import torch
import numpy as np

from torch.utils.data import Sampler
#from numpy.random import choice
from . import get_global_seed


class OfflineSampler:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def group_by_labels(self, dataset):
        # This function assumes the dataset to be in the form of
        # [(Feature_1, Label_1), (Feature_2, Label_2), .....]
        # Note that __getitem__ returns (image, label) in torch.Dataset & derivatives.
        num_classes = self.num_classes
        groups = []
        
        for _ in range(num_classes):
            groups.append([])
        
        for (feature, label) in dataset:
            groups[label].append([feature, label])
        
        return groups


class OverSampler(OfflineSampler):
    def __init__(self, num_classes: int, ratio: float = 1.0):
        OfflineSampler.__init__(self, num_classes)
        self.ratio = ratio
        self.rng = np.random.default_rng(seed=get_global_seed())
    
    def __call__(self, dataset):
        ratio = self.ratio
        groups = self.group_by_labels(dataset)
        imbalanced_data = []
        size = [len(group) for group in groups]
        
        lower_limit = int(max(size) * ratio)  # MIN_SIZE / MAX_SIZE = RATIO
        
        for (num, group) in enumerate(groups):
            self.rng.shuffle(group)
            size_sample = size[num]
            
            if size_sample < lower_limit:
                while size_sample < lower_limit:
                    imbalanced_data += group
                    size_sample += size[num]
                
                imbalanced_data += group[0:lower_limit - (size_sample - size[num])]
            
            else:
                imbalanced_data += group
        
        return imbalanced_data


class UnderSampler(OfflineSampler):
    def __init__(self, num_classes: int, ratio: float = 1.0):
        OfflineSampler.__init__(self, num_classes)
        self.ratio = ratio
        self.rng = np.random.default_rng(seed=get_global_seed())
    
    def __call__(self, dataset):
        ratio = self.ratio
        groups = self.group_by_labels(dataset)
        imbalanced_data = []
        size = [len(group) for group in groups]
        
        upper_limit = int(min(size) / ratio)  # MIN_SIZE / MAX_SIZE = RATIO -> MAX_SIZE = MIN_SIZE/RATIO
        
        for (num, group) in enumerate(groups):
            self.rng.shuffle(group)
            sample_size = size[num]
            
            if sample_size > upper_limit: sample_size = upper_limit
            
            imbalanced_data += group[0:sample_size]
        
        return imbalanced_data
import torch

from torch.utils.data import Sampler
from numpy.random import shuffle, choice


class OfflineSampler:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def group_by_labels(self, dataset):
        # This function assumes the dataset to be in the form of
        # [(Feature_1, Label_1), (Feature_2, Label_2), .....]
        # (what if? ([FEATURES], [labels]))
        # TODO: Does get_item return value in the standard way ((feat, label) pairs)?
        #   If so, remove assumption.
        num_classes = self.num_classes
        groups = []
        
        for _ in range(num_classes):
            groups.append([])
        
        for (feature, label) in dataset:
            groups[label].append((feature, label))
        
        return groups


class OverSampler(OfflineSampler):
    def __init__(self, num_classes: int, ratio: float = 1.0):
        OfflineSampler.__init__(self, num_classes)
        self.ratio = ratio
    
    def __call__(self, dataset):
        ratio = self.ratio
        groups = self.group_by_labels(dataset)
        imbalanced_data = []
        size = [len(group) for group in groups]
        
        lower_limit = int(max(size) * ratio)  # MIN_SIZE / MAX_SIZE = RATIO
        
        for (num, group) in enumerate(groups):
            shuffle(group)
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
    
    def __call__(self, dataset):
        ratio = self.ratio
        groups = self.group_by_labels(dataset)
        imbalanced_data = []
        size = [len(group) for group in groups]
        
        upper_limit = int(min(size) / ratio)  # MIN_SIZE / MAX_SIZE = RATIO -> MAX_SIZE = MIN_SIZE/RATIO
        
        for (num, group) in enumerate(groups):
            
            shuffle(group)
            sample_size = size[num]
            
            if sample_size > upper_limit: sample_size = upper_limit
            
            imbalanced_data += group[0:sample_size]
        
        return imbalanced_data


class OnlineSampler(Sampler[int]):
    def __init__(self, dataset, num_classes: int):
        self.num_classes = num_classes
        
        self.index_groups = self.get_index_groups(dataset)
        self.sizes = torch.tensor([len(group) for group in self.index_groups], dtype=torch.int64)
        self.length = self.sizes.sum()
    
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
        self.classes = torch.multinomial(self.probs, self.length, replacement=True)
        
        for i in range(self.length):
            random_class = self.classes[i]
            curr_group = self.index_groups[random_class]
            sample = choice(curr_group)
            
            yield sample


class ProgressivelyBalancedSampling(OnlineSampler):
    def __init__(self, dataset, num_classes: int, total_epochs: int):
        OnlineSampler.__init__(self, dataset, num_classes)
        
        self.class_ratios = self.sizes / self.length
        self.curr_epoch = 0
        self.total_epochs = total_epochs
    
    def __iter__(self):
        self.probs = (1 - self.curr_epoch / self.total_epochs) * self.class_ratios + (
                self.curr_epoch / self.total_epochs) * (1 / self.num_classes)
        self.classes = torch.multinomial(self.probs, self.length, replacement=True)
        self.curr_epoch = (self.curr_epoch + 1) % self.total_epochs
        
        for i in range(self.length):
            random_class = self.classes[i]
            curr_group = self.index_groups[random_class]
            sample = choice(curr_group)
            
            yield sample

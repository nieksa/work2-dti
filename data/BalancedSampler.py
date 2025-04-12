# 继承Sampler类，重写__iter__方法，实现平衡采样
# 我希望是先评估dataset里面各labels的分布，然后尽可能通过权重来对少数类进行过适当过采样，确保在dataloader中每个batch中各类样本的分布尽可能均匀
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch

class BalancedSampler(Sampler):
    def __init__(self, dataset, replacement=True):
        self.dataset = dataset
        self.replacement = replacement
        
        # 计算每个类别的样本数量
        self.labels = dataset.labels
        self.unique_labels = torch.unique(self.labels)
        self.label_counts = {label.item(): (self.labels == label).sum().item() 
                            for label in self.unique_labels}
        
        # 计算每个样本的权重，少数类的样本权重更大
        max_count = max(self.label_counts.values())
        self.weights = torch.zeros(len(self.labels))
        for label in self.unique_labels:
            label = label.item()
            mask = (self.labels == label)
            self.weights[mask] = max_count / self.label_counts[label]
            
        # 创建权重采样器
        self.sampler = torch.utils.data.WeightedRandomSampler(
            self.weights, 
            num_samples=len(self.labels),
            replacement=replacement
        )

    def __iter__(self):
        return iter(self.sampler)
    
    def __len__(self):
        return len(self.labels) 

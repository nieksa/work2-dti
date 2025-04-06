# 继承Sampler类，重写__iter__方法，实现平衡采样
# 我希望是先评估dataset里面各labels的分布，然后尽可能通过权重来对少数类进行过适当过采样，确保在dataloader中每个batch中各类样本的分布尽可能均匀
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch

class BalancedSampler(Sampler):
    def __init__(self, dataset):
        self.subject_id = dataset.subject_id
        self.event_id = dataset.event_id
        self.labels = dataset.labels

        self.unique_labels = np.unique(self.labels)
        self.label_counts = {label: np.sum(self.labels == label) for label in self.unique_labels}
        self.weights = {label: 1.0 / self.label_counts[label] for label in self.unique_labels}

    def __iter__(self): 
        # 打乱数据顺序，确保每次加载顺序不同
        combined = list(zip(self.subject_id, self.event_id, self.labels))
        random.shuffle(combined)
        self.subject_id, self.event_id, self.labels = zip(*combined)

        # 根据权重对少数类进行过适当过采样  
        # 计算每个类别的权重
        weights = [self.weights[label] for label in self.labels]
        
        # 使用torch.utils.data.WeightedRandomSampler进行过采样
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)   
        
        return iter(sampler)
    
    def __len__(self):
        return len(self.labels) 

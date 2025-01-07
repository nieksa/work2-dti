import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from torch.utils.data import Subset, DataLoader
from collections import Counter
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms

class DTIDataset(Dataset):
    def __init__(self, csv_file, args, channels, transform=None, template="s6mm"):
        """
        初始化数据集
        :param csv_file: CSV 文件路径，包含样本路径、参与者ID和标签
        :param args: 参数对象，包含任务类型和数据目录等信息
        :param channels: 使用的模态通道列表
        :param transform: 数据变换函数或变换列表
        :param template: 模板类型，如 "s6mm", "2mm", "1mm"
        """
        self.data_info = pd.read_csv(csv_file, dtype={0: str})
        self.task = args.task
        self.root_dir = args.data_dir
        self.channels = channels
        self.transform = transform
        self.template = template
        # 根据 task 筛选样本
        self.subject_id, self.event_id, self.labels = self._filter_samples()

        self.debug = args.debug
        self.debug_size = 20
        if self.debug:
            self.subject_id = self.subject_id[:self.debug_size]
            self.event_id = self.event_id[:self.debug_size]
            self.labels = self.labels[:self.debug_size]

    def _filter_samples(self):
        """
        根据 task 筛选样本，并转换标签为 0 和 1
        :return: 筛选后的 subject_id, event_id, labels
        """
        if self.task == 'NCvsPD':
            mask = self.data_info.iloc[:, 2].isin([1, 2])
            labels = self.data_info.iloc[:, 2].replace({1: 1, 2: 0})
        elif self.task == 'NCvsProdromal':
            mask = self.data_info.iloc[:, 2].isin([2, 4])
            labels = self.data_info.iloc[:, 2].replace({2: 0, 4: 1})
        elif self.task == 'ProdromalvsPD':
            mask = self.data_info.iloc[:, 2].isin([1, 4])
            labels = self.data_info.iloc[:, 2].replace({4: 0, 1: 1})
        else:
            raise ValueError(f"Unknown task: {self.task}")

        subject_id = self.data_info.iloc[:, 0].values[mask]
        event_id = self.data_info.iloc[:, 1].values[mask]
        labels = labels.values[mask]

        return subject_id, event_id, labels

    def __len__(self):
        return len(self.subject_id)

    def __getitem__(self, idx):
        """
        加载单个样本的数据和标签
        :param idx: 样本索引
        :return: 数据和标签
        """
        data = self._load_niigz(idx)
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)  # 应用数据变换

        return data, label

    def _load_niigz(self, idx):
        """
        加载 .nii.gz 文件
        """
        result = None
        for i, item in enumerate(self.channels):
            specifi_file_pattern = f"{self.subject_id[idx]}_{item}*{self.template}.nii.gz"
            file_path_pattern = os.path.join(self.root_dir, self.event_id[idx], "DTI_Results_GOOD", str(self.subject_id[idx]),
                                             "standard_space", specifi_file_pattern)
            matching_files = glob.glob(file_path_pattern)
            if not matching_files:
                raise FileNotFoundError(f"No file found matching pattern: {file_path_pattern}")
            file_path = matching_files[0]
            img = nib.load(file_path)
            data = img.get_fdata()
            tensor = torch.from_numpy(data).float()
            tensor = tensor.unsqueeze(0)  # 添加通道维度
            if i == 0:
                result = tensor
            else:
                result = torch.cat((result, tensor), dim=0)  # 在通道维度拼接

        return result

# 自定义变换类
class BoundaryCrop:
    def __call__(self, data):
        """
        边界裁剪
        :param data: 输入数据，形状为 (channels, height, width, depth)
        :return: 裁剪后的数据
        """
        shape = data.shape
        if shape[1] == 91:
            data = data[:, 5:-5, 5:-5, 5:-5]
        elif shape[1] == 182:
            data = data[:, 16:-16, 16:-16, 16:-16]
        return data

class CenterCrop:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        """
        中心裁剪到目标大小
        :param data: 输入数据，形状为 (batch, channel, height, width, depth)
        :return: 裁剪后的数据，形状为 (batch, channel, target_size, target_size, target_size)
        """
        channel, height, width, depth = data.shape
        start_height = (height - self.target_size) // 2
        start_width = (width - self.target_size) // 2
        start_depth = (depth - self.target_size) // 2

        # 裁剪 height, width, depth 三个维度
        data = data[
            :,  # 保留 channel 维度
            start_height:start_height + self.target_size,  # 裁剪 height
            start_width:start_width + self.target_size,    # 裁剪 width
            start_depth:start_depth + self.target_size     # 裁剪 depth
        ]
        return data

class IntervalSlice:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        """
        间隔切片法降采样
        :param data: 输入数据，形状为 (batch, channel, height, width, depth)
        :return: 降采样后的数据，形状为 (batch, channel, target_size, target_size, target_size)
        """
        channel, height, width, depth = data.shape
        step_height = max(1, height // self.target_size)  # 确保步长至少为 1
        step_width = max(1, width // self.target_size)    # 确保步长至少为 1
        step_depth = max(1, depth // self.target_size)    # 确保步长至少为 1

        # 间隔切片
        sliced_data = data[
            :,  # 保留 channel 维度
            ::step_height,  # 间隔切片 height
            ::step_width,   # 间隔切片 width
            ::step_depth    # 间隔切片 depth
        ]

        # 如果切片后的尺寸大于目标尺寸，则裁剪到目标尺寸
        if sliced_data.shape[1] > self.target_size:
            sliced_data = sliced_data[:, :self.target_size, :, :]
        if sliced_data.shape[2] > self.target_size:
            sliced_data = sliced_data[:, :, :self.target_size, :]
        if sliced_data.shape[3] > self.target_size:
            sliced_data = sliced_data[:, :, :, :self.target_size]

        return sliced_data

# 当前可以选择的shape是 91, 109, 91
# 或者是 182， 218， 182
# 有没有可能就是说使用一个transforme函数来调整dataset中data的shape增强泛化性呢？
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 192, 218, 192)  # (batch, channels, depth, height, width)

    # 2. 中心裁剪为 (batch, 3, 186, 186, 186)
    transform_crop = CenterCrop(target_size=186)
    cropped_tensor = transform_crop(input_tensor)  # (batch, 3, 186, 186, 186)
    print(cropped_tensor.shape)
    # 3. 间隔切片法降采样为 (batch, 3, 128, 128, 128)
    transform_slice = IntervalSlice(target_size=128)
    sliced_tensor = transform_slice(cropped_tensor)  # (batch, 3, 128, 128, 128)
    print(sliced_tensor.shape)
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
class DTIDataset(Dataset):
    def __init__(self, csv_file, args, channels, transform=None, template="1mm"):
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
        self.root_dir = "./data/ppmi/"
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

class CenterCrop:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
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


def plot_slices(data, title):
    """
    绘制数据的中间切片
    """
    # 选择第一个通道（如果数据是 4D 的）
    if len(data.shape) == 4:
        data = data[0]  # 选择第一个通道

    mid_slice_x = data[data.shape[0] // 2, :, :]  # X轴中间切片
    mid_slice_y = data[:, data.shape[1] // 2, :]  # Y轴中间切片
    mid_slice_z = data[:, :, data.shape[2] // 2]  # Z轴中间切片

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mid_slice_x, cmap='gray')
    axes[0].set_title(f'{title} - X Mid slice')
    axes[0].axis('off')

    axes[1].imshow(mid_slice_y, cmap='gray')
    axes[1].set_title(f'{title} - Y Mid slice')
    axes[1].axis('off')

    axes[2].imshow(mid_slice_z, cmap='gray')
    axes[2].set_title(f'{title} - Z Mid slice')
    axes[2].axis('off')

    plt.show()
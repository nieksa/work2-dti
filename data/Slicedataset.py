import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import torch

class SliceDataset(Dataset):
    def __init__(self, csv_file, args, transform=None):
        """
        初始化数据集
        :param csv_file: CSV 文件路径，包含样本路径、参与者ID和标签
        :param args: 参数对象，包含任务类型和数据目录等信息
        :param transform: 数据变换函数或变换列表
        """
        self.data_info = pd.read_csv(csv_file, dtype={0: str})
        self.task = args.task
        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform

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
        data = self._load_npz(idx)

        # 动态计算切片索引
        depth, height, width = data.shape[1], data.shape[2], data.shape[3]
        x_slice = data[:, depth // 2, :, :]  # 取深度方向的中间切片
        y_slice = data[:, :, height // 2, :]  # 取高度方向的中间切片
        z_slice = data[:, :, :, width // 2]  # 取宽度方向的中间切片

        x_slice = x_slice.unsqueeze(-1)  # 形状变为 [C, H, W, 1]
        y_slice = y_slice.unsqueeze(-1)  # 形状变为 [C, H, W, 1]
        z_slice = z_slice.unsqueeze(-1)  # 形状变为 [C, H, W, 1]

        # 在最后一个维度上拼接三个切片
        combined_data = torch.cat([x_slice, y_slice, z_slice], dim=-1)  # 形状变为 [C, H, W, 3]

        label = self.labels[idx]

        if self.transform:
            combined_data = self.transform(combined_data)  # 应用数据变换

        return combined_data, label

    def _load_npz(self, idx):
        """
        加载 .npz 文件并返回数据
        :param idx: 样本索引
        :return: 数据数组（形状为 (channels, depth, height, width)）
        """
        # 构建文件路径模式
        file_pattern = f"{self.subject_id[idx]}_FA_L1_MD_1mm.npz"
        file_path = os.path.join(
            self.root_dir,
            self.event_id[idx],
            "DTI_Results_GOOD",
            str(self.subject_id[idx]),
            "standard_space",
            file_pattern
        )

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 加载 .npz 文件
        npz_data = np.load(file_path)

        # 检查数据键是否存在
        if "data" not in npz_data:
            raise KeyError(f"Key 'data' not found in .npz file: {file_path}")

        return npz_data["data"]

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
        raw_data = pd.read_csv(csv_file, dtype={"PATNO": str, "EVENT_ID": str})
        self.args = args
        self.task = args.task
        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform

        # 根据 task 筛选样本
        self.subject_id, self.event_id, self.labels = self._filter_samples(raw_data)

        self.debug = args.debug
        self.debug_size = 20

        if self.args.debug:
            self._enable_debug_mode()

    def _enable_debug_mode(self):
        """调试模式优化"""
        print(f"【调试模式】原始数据量: {len(self.subject_id)}")
        self.subject_id = self.subject_id[:self.args.debug_size]
        self.event_id = self.event_id[:self.args.debug_size]
        self.labels = self.labels[:self.args.debug_size]
        print(f"【调试模式】当前数据量: {len(self.subject_id)}")

    def _filter_samples(self, df):
        """根据任务类型筛选样本并转换标签"""
        task_config = {
            'NCvsPD': {'include': [1, 2], 'mapping': {1: 1, 2: 0}},
            'NCvsProdromal': {'include': [2, 4], 'mapping': {2: 0, 4: 1}},
            'ProdromalvsPD': {'include': [1, 4], 'mapping': {4: 0, 1: 1}}
        }

        if self.args.task not in task_config:
            raise ValueError(f"无效任务类型: {self.args.task}。可选: {list(task_config.keys())}")

        cfg = task_config[self.args.task]
        mask = df['APPRDX'].isin(cfg['include'])
        labels = df['APPRDX'].replace(cfg['mapping'])

        return (
            df['PATNO'].values[mask],
            df['EVENT_ID'].values[mask],
            labels.values[mask].astype(np.int64)
        )

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


        label = self.labels[idx]

        return x_slice,y_slice,z_slice, label

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

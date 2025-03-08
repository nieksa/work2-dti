import torch
from monai.data import CacheDataset
import pandas as pd
import os
import numpy as np
import psutil
import random
class ContrastiveDataset(CacheDataset):
    def __init__(self, csv_file, args, transform=None):
        self.task = args.task
        self.args = args
        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform
        # 根据 task 筛选样本
        raw_data = pd.read_csv(csv_file, dtype={"PATNO": str, "EVENT_ID": str})
        self.subject_id, self.event_id, self.labels = self._filter_samples(raw_data)

        self.debug = args.debug
        self.debug_size = 20

        cache_rate = self._get_cache_rate()
        if self.debug:
            self.subject_id = self.subject_id[:self.debug_size]
            self.event_id = self.event_id[:self.debug_size]
            self.labels = self.labels[:self.debug_size]
            super().__init__(
                data=list(zip(self.subject_id, self.event_id, self.labels)),
                transform=transform,
                cache_rate=cache_rate,  # 缓存全部数据
                num_workers=args.num_workers,  # 多线程加载
            )
    def _shuffle_data(self):
        # 打乱数据顺序，确保每次加载顺序不同
        combined = list(zip(self.subject_id, self.event_id, self.labels))
        random.shuffle(combined)
        self.subject_id, self.event_id, self.labels = zip(*combined)

    def _get_cache_rate(self):
        """
        根据内存大小和数据量动态设置 cache_rate
        :return: 缓存比例
        """
        # 获取内存信息
        memory_info = psutil.virtual_memory()
        available_memory = memory_info.available  # 可用内存（字节）

        # 估算数据集大小（假设每个样本 100MB）
        sample_size = 10 * 1024 * 1024  # 100MB
        total_data_size = len(self.subject_id) * sample_size

        # 计算缓存比例
        cache_rate = min(1.0, available_memory / total_data_size)
        return cache_rate

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
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)  # 应用数据变换
        fa = torch.tensor(data[0, :, :, :], dtype=torch.float32).unsqueeze(0)
        md = torch.tensor(data[1, :, :, :], dtype=torch.float32).unsqueeze(0)
        data = (fa, md)
        return data, label

    def _load_npz(self, idx):
        """
        加载 .npz 文件并返回数据
        :param idx: 样本索引
        :return: 数据数组（形状为 (channels, depth, height, width)）
        """
        # 构建文件路径模式
        file_pattern = f"{self.subject_id[idx]}_FA_MD_1mm_float16.npz"
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
        if "arr_0" not in npz_data:
            raise KeyError(f"Key 'data' not found in .npz file: {file_path}")

        return npz_data["arr_0"]
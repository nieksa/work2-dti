import pandas as pd
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class ROIDataset(Dataset):
    def __init__(self, csv_file, args, channels, transform=None):
        self.data_info = pd.read_csv(csv_file, dtype={0: str})
        raw_data = pd.read_csv(csv_file, dtype={"PATNO": str, "EVENT_ID": str})
        self.args = args
        self.task = args.task
        self.root_dir = "./data/ppmi_npz/"
        self.channels = channels
        self.transform = transform
        self.subject_id, self.event_id, self.labels = self._filter_samples(raw_data)

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
        data = self._load_roi(idx)
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)  # 应用数据变换

        return data, label

    def _load_roi(self, idx):
        result = []
        for i, item in enumerate(self.channels):
            specifi_file_pattern = f"{self.subject_id[idx]}_*_{item}-*.txt"
            file_path_pattern = os.path.join(self.root_dir, self.event_id[idx], "DTI_Results_GOOD",
                                             str(self.subject_id[idx]),
                                             "Network", "Deterministic", specifi_file_pattern)

            # 获取匹配的文件列表，并过滤掉包含 "CentroidPos" 的文件
            matching_files = glob.glob(file_path_pattern)
            matching_files = [file for file in matching_files if "CentroidPos" not in file]

            if not matching_files:
                raise FileNotFoundError(f"No file found matching pattern: {file_path_pattern}")

            channel_data = []
            for file in matching_files:
                # 加载数据并确保为 float32 格式
                data = np.loadtxt(file, dtype=np.float32)  # 显式指定 dtype=np.float32
                data = data.reshape(-1, 1)  # 确保形状为 (n, 1)
                channel_data.append(data)

            # 水平拼接，确保结果仍然是 float32
            channel_data = np.hstack(channel_data).astype(np.float32)  # 水平拼接并确保 float32
            result.append(channel_data)

        # 最终拼接，确保结果仍然是 float32
        result = np.hstack(result).astype(np.float32)  # 水平拼接并确保 float32

        return result
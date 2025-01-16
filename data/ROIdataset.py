import pandas as pd
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class ROIDataset(Dataset):
    def __init__(self, csv_file, args, channels, transform=None):
        self.data_info = pd.read_csv(csv_file, dtype={0: str})
        self.task = args.task
        self.root_dir = "./data/ppmi/"
        self.channels = channels
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
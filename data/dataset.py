import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    def __init__(self, csv_file, args):
        """
        初始化数据集
        :param csv_file: CSV 文件路径，包含样本路径、参与者ID和标签
        :param task: 任务类型，用于筛选特定标签的样本
        """
        self.data_info = pd.read_csv(csv_file)
        self.task = args.task

        # 根据 task 筛选样本
        self.file_paths, self.participant_ids, self.labels = self._filter_samples()

        self.debug = args.debug
        self.debug_size = 10
        if self.debug:
            self.file_paths = self.file_paths[:self.debug_size]
            self.participant_ids = self.participant_ids[:self.debug_size]
            self.labels = self.labels[:self.debug_size]

    def _filter_samples(self):
        """
        根据 task 筛选样本，并转换标签为 0 和 1
        :return: 筛选后的 file_paths, participant_ids, labels
        """
        # 根据 task 筛选样本
        if self.task == 'task1':
            # 例如，task1 只保留标签为 1 和 2 的样本，并将标签转换为 0 和 1
            mask = self.data_info.iloc[:, 2].isin([1, 2])  # 假设第三列是标签
            labels = self.data_info.iloc[:, 2].replace({1: 0, 2: 1})  # 将 1 转换为 0，2 转换为 1
        elif self.task == 'task2':
            # 例如，task2 只保留标签为 3 和 4 的样本，并将标签转换为 0 和 1
            mask = self.data_info.iloc[:, 2].isin([3, 4])
            labels = self.data_info.iloc[:, 2].replace({3: 0, 4: 1})
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # 返回筛选后的数据
        file_paths = self.data_info.iloc[:, 0].values[mask]
        participant_ids = self.data_info.iloc[:, 1].values[mask]
        labels = labels.values[mask]

        return file_paths, participant_ids, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        加载单个样本的数据和标签
        :param idx: 样本索引
        :return: 数据和标签
        """
        # 加载 .nii.gz 文件
        data = self._load_niigz(self.file_paths[idx])

        # 获取标签
        label = self.labels[idx]

        return data, label

    def _load_niigz(self, file_path):
        """
        加载 .nii.gz 文件
        :param file_path: .nii.gz 文件路径
        :return: 数据（numpy 数组）
        """
        img = nib.load(file_path)
        data = img.get_fdata()  # 获取数据为 numpy 数组
        return data
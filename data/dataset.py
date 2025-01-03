import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
import os
import glob
import torch.nn.functional as F

class DTIDataset(Dataset):
    def __init__(self, csv_file, args, channels, transform=None):
        """
        初始化数据集
        :param csv_file: CSV 文件路径，包含样本路径、参与者ID和标签
        :param task: 任务类型，用于筛选特定标签的样本
        """
        self.data_info = pd.read_csv(csv_file, dtype={0: str})
        self.task = args.task
        self.root_dir = args.data_dir
        self.channels = channels
        self.transform = transform
        # 根据 task 筛选样本
        self.subject_id, self.event_id, self.labels = self._filter_samples()

        self.debug = args.debug
        self.debug_size = 10
        if self.debug:
            self.subject_id = self.subject_id[:self.debug_size]
            self.event_id = self.event_id[:self.debug_size]
            self.labels = self.labels[:self.debug_size]

    def _filter_samples(self):
        """
        根据 task 筛选样本，并转换标签为 0 和 1
        :return: 筛选后的 file_paths, participant_ids, labels
        """
        # 根据 task 筛选样本
        if self.task == 'NCvsPD':
            # 只保留标签为 1 和 2 的样本，并将标签转换为 0 和 1
            mask = self.data_info.iloc[:, 2].isin([1, 2])  # 假设第三列是标签
            labels = self.data_info.iloc[:, 2].replace({1: 1, 2: 0})  # 将 1 转换为 0，2 转换为 1
        elif self.task == 'NCvsProdromal':
            mask = self.data_info.iloc[:, 2].isin([2, 4])
            labels = self.data_info.iloc[:, 2].replace({2: 0, 4: 1})
        elif self.task == 'ProdromalvsPD':
            mask = self.data_info.iloc[:, 2].isin([1, 4])
            labels = self.data_info.iloc[:, 2].replace({4: 0, 1: 1})
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # 返回筛选后的数据
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

        return data, label

    def _load_niigz(self, idx):
        """
        加载 .nii.gz 文件
        """
        result = None
        # 根据idx找到subject_id 和 event_id 就可以找到目标路径，然后就根据self.channels里面指定的多种模态医学数据来加载，在第二个dim完成拼接
        for i, item in enumerate(self.channels):
            specifi_file_pattern = f"{self.subject_id[idx]}_{item}*_4normalize_to_target_2mm_s6mm.nii.gz"
            file_path_pattern = os.path.join(self.root_dir, self.event_id[idx], "DTI_Results_GOOD", str(self.subject_id[idx]),
                                             "standard_space", specifi_file_pattern)
            matching_files = glob.glob(file_path_pattern)
            if not matching_files:
                raise FileNotFoundError(f"No file found matching pattern: {file_path_pattern}")
            file_path = matching_files[0]  # 取第一个匹配的文件
            img = nib.load(file_path)
            data = img.get_fdata()
            tensor = torch.from_numpy(data).float()
            tensor = tensor.unsqueeze(0)
            if self.transform:
                tensor = self._transform_shape(tensor)
            if i == 0:
                result = tensor
            else:
                result = torch.cat((result, tensor), dim=0)  # 在通道维度拼接

        return result

    def _transform_shape(self, data):
        """
        调整数据的形状为目标形状
        :param data: 输入数据，形状为 (channels, height, width, depth)
        :return: 调整后的数据，形状为 (channels, target_height, target_width, target_depth)
        """
        if self.transform == 'interpolation_91':
            data = data.unsqueeze(0)  # 添加 batch 维度
            data = F.interpolate(data, size=(91,91,91), mode='trilinear', align_corners=False)
            data = data.squeeze(0)  # 去掉 batch 维度
            return data
        if self.transform == 'crop_91':
            _, height, width, depth = data.shape
            target_height, target_width, target_depth = 91

            # 计算裁剪的起始和结束索引
            start_height = (height - target_height) // 2
            start_width = (width - target_width) // 2
            start_depth = (depth - target_depth) // 2

            end_height = start_height + target_height
            end_width = start_width + target_width
            end_depth = start_depth + target_depth

            # 裁剪数据
            data = data[:, start_height:end_height, start_width:end_width, start_depth:end_depth]
            return data

# 当前可以选择的shape是 91, 109, 91
# 或者是 182， 218， 182
# 有没有可能就是说使用一个transforme函数来调整dataset中data的shape增强泛化性呢？
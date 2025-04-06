# 定义一个BaseDataset基类，加载csv文件和任务类型，GraphDataset和ContrastiveDataset都是重写_getitem方法来加载不同数据
# task有 NCvsPD, NCvsProdromal, ProdromalvsPD，NCvsProdromalvsPD。四种情况
# 加载csv文件的时候根据task筛选指定行，并且根据task修改APPRDX的值，在原数据中APPRDX=1是NC，2是PD，4是Prodromal。
# 我希望根据任务映射，如果是2分类，病情轻的作为0，病情重的作为1。
# NCvsProdromal，NC作为0，Prodromal作为1。
# ProdromalvsPD，Prodromal作为0，PD作为1。
# NCvsPD，NC作为0，PD作为1。
# NCvsProdromalvsPD，NC作为0，Prodromal作为1，PD作为2。
# 这个就是BaseDataset的_filter_samples方法
# csv_file的位置是 ./data/data.csv
# 考虑要不要改成CacheDataset
import torch
import nibabel as nib
from monai.data import CacheDataset
import random
import pandas as pd
from torch.utils.data import Dataset
import os
import glob
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, csv_file, args, transform=None):
        self.csv_file = pd.read_csv(csv_file, dtype={'PATNO': 'str'})
        self.task = args.task

        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform

        self.filtered_data = self._filter_samples()
        self.labels = torch.tensor(self.filtered_data['APPRDX'].values, dtype=torch.long)
        self.subject_id = self.filtered_data['PATNO'].values
        self.event_id = self.filtered_data['EVENT_ID'].values

        # self.clinical_scores = 从filtered_data中获取指定指标的值，有些空的列就用0代替

    def _filter_samples(self):
        data = self.csv_file.copy()
        label_mapping = {
            'NCvsPD': {2: 0, 1: 1},
            'NCvsProdromal': {2: 0, 4: 1},
            'ProdromalvsPD': {4: 0, 1: 1},
            'NCvsProdromalvsPD': {2: 0, 4: 1, 1: 2}
        }
        # 1 - 392 - PD
        # 2 - 121 - NC
        # 4 - 123 - Prodromal
        if self.task not in label_mapping:
            raise ValueError(f"Unsupported task: {self.task}")
        task_mapping = label_mapping[self.task]
        valid_labels = list(task_mapping.keys())
        data = data[data['APPRDX'].isin(valid_labels)]
        data['APPRDX'] = data['APPRDX'].map(task_mapping)
        return data

    def __len__(self):
        return len(self.labels)
        
    def _shuffle_data(self):
        # 打乱顺序，避免加载的时候同一类标签连续出现在一个batch中
        combined = list(zip(self.subject_id, self.event_id, self.labels))
        random.shuffle(combined)
        self.subject_id, self.event_id, self.labels = zip(*combined)

    def __getitem__(self, idx):
        # 返回过滤后的数据（可以根据需要修改这里的返回内容）
        return self.csv_file.iloc[idx]

    def _load_dti(self, idx, modality):
        # 构建文件路径模式
        file_pattern = f"{self.subject_id[idx]}_{modality}*1mm.nii.gz"
        search_path = os.path.join(
            self.root_dir,
            self.event_id[idx],
            "DTI_Results_GOOD", 
            str(self.subject_id[idx]),
            "standard_space",
            file_pattern
        )
        matched_files = glob.glob(search_path)
        if not matched_files:
            raise FileNotFoundError(f"找不到匹配的文件: {search_path}")
        nii_img = nib.load(matched_files[0])
        data = nii_img.get_fdata()
        if data.shape != (182, 218, 182):
            raise ValueError(f"数据形状不正确,期望(182,218,182),实际{data.shape}")
        data = data[None, ...]  # 等同于 data.reshape(1, 182, 218, 182)
        return data

    def _load_mri(self, idx):
        file_pattern = f"co_{self.subject_id[idx]}*.nii.gz"
        search_path = os.path.join(
            self.root_dir,
            self.event_id[idx],
            "DTI_Results_GOOD", 
            str(self.subject_id[idx]),
            "T1",
            file_pattern
        )
        matched_files = glob.glob(search_path)
        if not matched_files:
            raise FileNotFoundError(f"找不到匹配的文件: {search_path}")
        nii_img = nib.load(matched_files[0])
        data = nii_img.get_fdata()
        if data.shape != (91, 109, 91):
            raise ValueError(f"数据形状不正确,期望(91,109,91),实际{data.shape}")
        data = data[None, ...]  # 等同于 data.reshape(1, 91, 109, 91)
        return data

    # def _load_graph(self, idx):
    #     # 这个图数据有待改进，连接性矩阵有FA FN 两种，具体如何构建还需要在考虑
    #     subject_id = self.subject_id[idx]
    #     event_id = self.event_id[idx]
    #     subject_path = os.path.join(self.root_dir, event_id, "DTI_Results_GOOD", subject_id, 'Network', 'Deterministic')
    #
    #     modalities = ["06LDHs", "07LDHk", "FA", "L1", "L23m", "MD"]
    #     metrics = ["AvgValue", "CentroidWeight", "MaxValue"]
    #     expected_patterns = [f"{mod}-{metric}" for mod in modalities for metric in metrics]
    #
    #     data_list = []
    #     node_files = sorted([f for f in os.listdir(subject_path) if any(pattern in f for pattern in expected_patterns)])
    #     if len(node_files) != len(expected_patterns):
    #         raise FileNotFoundError("节点特征文件数量不匹配，请检查文件完整性。")
    #     for file_name in node_files:
    #         file_path = os.path.join(subject_path, file_name)
    #         data_list.append(np.loadtxt(file_path))
    #     x = np.stack(data_list, axis=-1)  # 90 x 18
    #
    #     # 加载 FA_matrix.txt 文件，转换为 90x90 邻接矩阵
    #     edge_files = [f for f in os.listdir(subject_path) if "Matrix_FA" in f and f.endswith(".txt")]
    #     if not edge_files:
    #         raise FileNotFoundError(f"在 {subject_path} 目录中找不到包含 'Matrix_FA' 的文件")
    #     edge_file = os.path.join(subject_path, edge_files[0])  # 选择第一个匹配的文件
    #     edge_attr = np.loadtxt(edge_file)  # 90 x 90
    #     edge_index, edge_weight = dense_to_sparse(torch.tensor(edge_attr, dtype=torch.float32))
    #
    #     label = torch.tensor(self.labels[index], dtype=torch.long)
    #
    #     return Data(x=torch.tensor(x, dtype=torch.float32),
    #                          edge_index=edge_index,
    #                          edge_attr=edge_weight,
    #                          y=label)
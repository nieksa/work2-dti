# 这里定义的是图结构数据 (06 07 FA L1 L23m MD)*(AvgValue + CentroidWeight + MaxValue) + ROI SurfaceSize ROIVoxelSize + (06 07 FA L1 L23m MD)*CentroidPos
# 写个脚本文件把 E:\PD_data\ppmi\[0m 12m 24m]\DTI_Results_GOOD\[subject_id是个六位数字]\Network\Deterministic 复制到 D:\Code\work2-dti\data\ppmi_npz 目录下，保持目录结构

import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import os
from torch_geometric.utils import dense_to_sparse
class GraphDataset(Dataset):
    def __init__(self, csv_file, args, transform=None):
        super(GraphDataset, self).__init__()
        self.task = args.task
        self.args = args
        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform
        # 根据 task 筛选样本
        raw_data = pd.read_csv(csv_file, dtype={"PATNO": str, "EVENT_ID": str})
        self.subject_id, self.event_id, self.labels = self._filter_samples(raw_data)

        self.debug = args.debug
        self.debug_size = 20
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

    def _load_data(self, index):
        subject_id = self.subject_id[index]
        event_id = self.event_id[index]
        subject_path = os.path.join(self.root_dir, event_id, "DTI_Results_GOOD", subject_id, 'Network', 'Deterministic')

        modalities = ["06LDHs", "07LDHk", "FA", "L1", "L23m", "MD"]
        metrics = ["AvgValue", "CentroidWeight", "MaxValue"]
        expected_patterns = [f"{mod}-{metric}" for mod in modalities for metric in metrics]

        data_list = []
        node_files = sorted([f for f in os.listdir(subject_path) if any(pattern in f for pattern in expected_patterns)])
        if len(node_files) != len(expected_patterns):
            raise FileNotFoundError("节点特征文件数量不匹配，请检查文件完整性。")
        for file_name in node_files:
            file_path = os.path.join(subject_path, file_name)
            data_list.append(np.loadtxt(file_path))
        x = np.stack(data_list, axis=-1)  # 90 x 18

        # 加载 FA_matrix.txt 文件，转换为 90x90 邻接矩阵
        edge_files = [f for f in os.listdir(subject_path) if "Matrix_FA" in f and f.endswith(".txt")]
        if not edge_files:
            raise FileNotFoundError(f"在 {subject_path} 目录中找不到包含 'Matrix_FA' 的文件")
        edge_file = os.path.join(subject_path, edge_files[0])  # 选择第一个匹配的文件
        edge_attr = np.loadtxt(edge_file)  # 90 x 90
        edge_index, edge_weight = dense_to_sparse(torch.tensor(edge_attr, dtype=torch.float32))

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return Data(x=torch.tensor(x, dtype=torch.float32),
                             edge_index=edge_index,
                             edge_attr=edge_weight,
                             y=label)
        # 根据 self.subject_id 和 self.event_id 能够精准的找到 节点数据 self.root_dir/self.event_id/self.subject_id/'Network'/'Deterministic/*_node.txt文件' 总共16个txt文件合起来构建成一个90*16的二维数据 作为data
        # 根据self.subject_id和self.event_id能够精准的找到节点数据self.root_dir / self.event_id / self.subject_id / 'Network' / 'Deterministic/FA_matrix.txt文件'总共16个txt文件合起来构建成一个90 * 16的二维数据作为edge
        # label = self.labels[index]

    def __getitem__(self, index):
        return self._load_data(index)

    def __len__(self):
        return len(self.subject_id)
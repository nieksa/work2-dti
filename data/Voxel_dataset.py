import torch
from monai.data import CacheDataset
import pandas as pd
import os
import numpy as np
import random
from sklearn.utils import resample
class VoxelDataset(CacheDataset):
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

        # 打乱数据顺序，确保每次加载顺序不同
        self._shuffle_data()

        if self.debug:
            self.subject_id = self.subject_id[:self.debug_size]
            self.event_id = self.event_id[:self.debug_size]
            self.labels = self.labels[:self.debug_size]
            super().__init__(
                data=list(zip(self.subject_id, self.event_id, self.labels)),
                transform=transform,
                num_workers=args.num_workers,  # 多线程加载
            )

    def _shuffle_data(self):
        # 打乱数据顺序，确保每次加载顺序不同
        combined = list(zip(self.subject_id, self.event_id, self.labels))
        random.shuffle(combined)
        self.subject_id, self.event_id, self.labels = zip(*combined)

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

        # 重采样PD数量
        df_filtered = df[mask].copy()
        df_filtered['LABEL'] = df_filtered['APPRDX'].replace(cfg['mapping'])
        if self.args.task == 'NCvsPD' or self.args.task == 'ProdromalvsPD':
            pd_samples = df_filtered[df_filtered['LABEL'] == 1]  # PD 类别
            non_pd_samples = df_filtered[df_filtered['LABEL'] == 0]  # 其他类别
            if len(pd_samples) > 125:
                pd_samples = pd_samples.sample(n=125, random_state=42)  # 下采样
            elif len(pd_samples) < 125:
                pd_samples = resample(pd_samples, replace=True, n_samples=125, random_state=42)  # 过采样

            df_balanced = pd.concat([pd_samples, non_pd_samples])

            return (
                df_balanced['PATNO'].values,
                df_balanced['EVENT_ID'].values,
                df_balanced['LABEL'].values.astype(np.int64)
            )

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
        fa_data = self._load_dti(idx, 'FA')
        mri_data = self._load_mri(idx)
        label = self.labels[idx]

        data = (fa_data, mri_data)

        if self.transform:
            data = self.transform(data)  # 应用数据变换
        return data, label

    def _load_dti(self, idx, modality):
        """
        加载 nii.gz 文件并返回数据
        :param idx: 样本索引
        :return: 数据数组（形状为 182x218x182）
        """
        import nibabel as nib
        import glob
        
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
        
        # 查找匹配的文件
        matched_files = glob.glob(search_path)
        if not matched_files:
            raise FileNotFoundError(f"找不到匹配的文件: {search_path}")
            
        # 加载第一个匹配的nii.gz文件
        nii_img = nib.load(matched_files[0])
        data = nii_img.get_fdata()
        
        # 确保数据形状为182x218x182
        if data.shape != (182, 218, 182):
            raise ValueError(f"数据形状不正确,期望(182,218,182),实际{data.shape}")
        # 在第一个维度添加channel维度
        data = data[None, ...]  # 等同于 data.reshape(1, 182, 218, 182)
        return data
    
    def _load_mri(self, idx):
        """
        加载 nii.gz 文件并返回数据
        :param idx: 样本索引
        :return: 数据数组（形状为 91x109x91）
        """
        import nibabel as nib
        import glob
        
        # 构建文件路径模式
        file_pattern = f"co_{self.subject_id[idx]}*.nii.gz"
        search_path = os.path.join(
            self.root_dir,
            self.event_id[idx],
            "DTI_Results_GOOD", 
            str(self.subject_id[idx]),
            "T1",
            file_pattern
        )
        
        # 查找匹配的文件
        matched_files = glob.glob(search_path)
        if not matched_files:
            raise FileNotFoundError(f"找不到匹配的文件: {search_path}")
            
        # 加载第一个匹配的nii.gz文件
        nii_img = nib.load(matched_files[0])
        data = nii_img.get_fdata()
        
        # 确保数据形状为91x109x91
        if data.shape != (91, 109, 91):
            raise ValueError(f"数据形状不正确,期望(91,109,91),实际{data.shape}")
        data = data[None, ...]  # 等同于 data.reshape(1, 91, 109, 91)
        return data
import os
import psutil
import numpy as np
import pandas as pd
from monai.data import CacheDataset
class NPZdataset(CacheDataset):
    def __init__(self, csv_file, args, transform=None):
        """
        优化后的数据集类，支持动态缓存和分层交叉验证
        :param csv_file: CSV文件路径（格式：subject_id, event_id, apprdx）
        :param args: 包含 task, debug, num_workers 等参数
        :param transform: 数据预处理流水线
        """
        # 初始化基础参数
        self.args = args
        self.root_dir = "./data/ppmi_npz/"
        self.transform = transform

        # 加载并预处理数据
        raw_data = pd.read_csv(csv_file, dtype={"PATNO": str, "EVENT_ID": str})
        self.subject_id, self.event_id, self.labels = self._filter_samples(raw_data)

        # 调试模式处理
        if self.args.debug:
            self._enable_debug_mode()

        # 动态计算缓存比例
        cache_rate = self._calculate_safe_cache_rate()

        # 初始化CacheDataset
        super().__init__(
            data=self._prepare_data_list(),
            transform=transform,
            cache_rate=cache_rate,
            num_workers=self.args.num_workers,
        )

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

    def _prepare_data_list(self):
        """生成MONAI兼容的数据列表"""
        return [
            {
                "subject_id": subj,
                "event_id": evt,
                "label": lbl,
                "npz_path": self._build_npz_path(subj, evt)
            }
            for subj, evt, lbl in zip(self.subject_id, self.event_id, self.labels)
        ]

    def _build_npz_path(self, subject_id, event_id):
        """动态构建npz文件路径"""
        components = [
            self.root_dir,
            event_id,
            "DTI_Results_GOOD",
            subject_id,
            "standard_space",
            f"{subject_id}_FA_L1_MD_2mm.npz"
        ]
        path = os.path.join(*components)

        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件缺失: {path}")
        return path

    def _calculate_safe_cache_rate(self):
        """安全的内存管理策略"""
        # 估算单样本内存占用
        sample_mem = 100 * 1024 ** 2  # 假设每个样本100MB
        total_need = len(self.subject_id) * sample_mem

        # 获取系统可用内存
        safe_margin = 0.2  # 保留20%内存余量
        available_mem = psutil.virtual_memory().available * (1 - safe_margin)

        # 计算安全缓存比例
        return min(1.0, available_mem / total_need)

    def _enable_debug_mode(self):
        """调试模式优化"""
        print(f"【调试模式】原始数据量: {len(self.subject_id)}")
        self.subject_id = self.subject_id[:self.args.debug_size]
        self.event_id = self.event_id[:self.args.debug_size]
        self.labels = self.labels[:self.args.debug_size]
        print(f"【调试模式】当前数据量: {len(self.subject_id)}")

    def __getitem__(self, index):
        """优化数据加载逻辑，确保正确加载.npz文件"""
        # 从缓存中获取元数据
        item = super().__getitem__(index)

        # 动态加载.npz文件（MONAI CacheDataset会缓存已加载的数据）
        try:
            # 加载.npz文件内容
            npz_file = np.load(item["npz_path"])
            data = npz_file["data"]  # 假设数据存储在'data'键下

            # 添加通道维度（如果是3D数据）
            if data.ndim == 3:
                data = np.expand_dims(data, axis=0)

            # 类型转换（确保与模型输入类型匹配）
            data = data.astype(np.float32)

            # 应用数据增强（如果定义）
            if self.transform:
                data = self.transform(data)

            return data, item["label"]

        except Exception as e:
            # 增强错误信息
            error_msg = (
                f"加载样本失败: [索引={index}] "
                f"[路径={item['npz_path']}]\n"
                f"错误类型: {type(e).__name__}\n"
                f"详细信息: {str(e)}"
            )
            raise RuntimeError(error_msg) from e
import numpy as np
from collections import Counter
import logging


def split_by_patno(dataset, indices, n_splits=5):
    """
    按subject_id进行K折交叉验证划分，确保每个折的数据来自不同的subject_id，且数据互不交叉。

    Args:
        dataset: 数据集对象，需要包含subject_id和event_id属性
        indices: 要划分的数据索引列表
        n_splits: 划分的折数，默认为5

    Returns:
        folds: 包含n_splits个(train_indices, val_indices)元组的列表
    """
    # 获取所有唯一的subject_id
    unique_subjects = set()
    subject_to_indices = {}

    for idx in indices:
        subject_id = dataset.subject_id[idx]  # 获取subject_id
        unique_subjects.add(subject_id)
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)

    # 将unique_subjects转换为列表并打乱
    unique_subjects = list(unique_subjects)
    np.random.shuffle(unique_subjects)

    # 计算每折应该包含的subject_id数量
    fold_size = len(unique_subjects) // n_splits
    folds = []

    # 划分subject_id
    for i in range(n_splits):
        # 确定当前折的验证集的subject_id
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_splits - 1 else len(unique_subjects)
        fold_subjects = unique_subjects[start_idx:end_idx]

        # 收集当前折验证集的所有索引
        val_indices = []
        for subject in fold_subjects:
            val_indices.extend(subject_to_indices[subject])

        # 获取训练集的索引（去除验证集的subject_id）
        train_indices = []
        for subject in unique_subjects:
            if subject not in fold_subjects:
                train_indices.extend(subject_to_indices[subject])

        folds.append((train_indices, val_indices))

    return folds


def print_split_info(dataset, train_indices, val_indices, fold_num):
    """
    打印数据划分的详细信息
    
    Args:
        dataset: 数据集对象
        train_indices: 训练集索引
        val_indices: 验证集索引
        fold_num: 当前折数
    """
    # 将标签转换为numpy数组
    train_labels = np.array([dataset.labels[i] for i in train_indices])
    val_labels = np.array([dataset.labels[i] for i in val_indices])
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    # 打印每个fold的Patno分布
    train_patnos = set(dataset.patnos[i] for i in train_indices)
    val_patnos = set(dataset.patnos[i] for i in val_indices)
    
    logging.info(f"\nFold {fold_num + 1}:")
    logging.info(f"Train Patnos: {len(train_patnos)}")
    logging.info(f"Val Patnos: {len(val_patnos)}")
    logging.info(f"Train samples: {len(train_indices)}")
    logging.info(f"Val samples: {len(val_indices)}")
    
    table = [
        "+-------------------+-------+-------+",
        "|                   | Label 0 | Label 1 |",
        "+-------------------+-------+-------+",
        f"| Train            |   {train_counter[0]}    |   {train_counter[1]}    |",
        "+-------------------+-------+-------+",
        f"| Validation       |   {val_counter[0]}    |   {val_counter[1]}    |",
        "+-------------------+-------+-------+"
    ]
    for row in table:
        logging.info(row) 
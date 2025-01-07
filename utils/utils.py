import os
import time
import logging
import argparse
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import json
import numpy as np
def setup_training_environment():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--model_name', type=str, default='ResNet18', help='Name of the model to use.')
    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
    parser.add_argument('--bs', type=int, default=32, help='I3D C3D cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')
    parser.add_argument('--debug', type=bool, default=False, help='small sample for debugging.')
    parser.add_argument('--data_dir', type=str, default='./data/ppmi/')

    # 解析命令行参数
    args = parser.parse_args()

    # 创建日志和模型保存目录
    log_dir = f'./logs/{args.task}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./saved_models/{args.task}', exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(log_dir, f'{args.model_name}_{timestamp}.log')

    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 将日志输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    # 打印训练配置
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        logging.info("Using single GPU.")

    logging.info(f"Training with {device}")

    return args, device, log_file, timestamp

def rename_log_file(log_file, avg_acc, task, model_name, timestamp):
    log_dir = f'./logs/{task}'
    new_logfilename = os.path.join(log_dir, f'{model_name}_{timestamp}_{avg_acc:.2f}.log')
    os.rename(log_file, new_logfilename)
    return new_logfilename

def set_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # CUDA 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子


def evaluate_model(model, val_loader, device):
    """
    评估模型性能，返回真实标签和预测概率
    """
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Evaluating"):
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)  # 获取概率
            preds = torch.argmax(probs, dim=1)  # 获取预测类别

            # 保存结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(labels, preds, class_names, model_name, save_dir):
    """
    绘制混淆矩阵并保存
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def plot_roc_curve(labels, probs_list, model_names, save_dir):
    """
    绘制 ROC 曲线并保存
    """
    plt.figure(figsize=(8, 6))
    for i, probs in enumerate(probs_list):
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])  # 假设二分类任务，使用正类的概率
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

def plot_pr_curve(labels, probs_list, model_names, save_dir):
    """
    绘制 PR 曲线并保存
    """
    plt.figure(figsize=(8, 6))
    for i, probs in enumerate(probs_list):
        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])  # 假设二分类任务，使用正类的概率
        plt.plot(recall, precision, lw=2, label=f'{model_names[i]}')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, "pr_curve.png"))
    plt.close()


# 在每一折结束时记录 labels, preds, probs
def log_fold_results(fold, labels, preds, probs):
    """
    将每一折的结果记录到日志文件中。

    参数:
    - fold: 当前折的编号
    - labels: 真实标签
    - preds: 预测标签
    - probs: 预测概率
    """
    # 将 numpy 数组转换为 Python 列表
    labels_list = labels.tolist()
    preds_list = preds.tolist()
    probs_list = probs.tolist()

    # 构建日志消息
    log_message = {
        'fold': fold,
        'labels': labels_list,
        'preds': preds_list,
        'probs': probs_list
    }

    # 将日志消息转换为 JSON 字符串并记录
    logging.info(json.dumps(log_message))
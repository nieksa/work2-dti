import os
import logging
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import json
import numpy as np


def rename_log_file(log_file, avg_acc, task, model_name, timestamp):
    log_dir = f'./logs/{task}'
    new_logfilename = os.path.join(log_dir, f'{model_name}_{timestamp}_{avg_acc:.2f}.log')
    os.rename(log_file, new_logfilename)
    return new_logfilename


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

    logging.info(json.dumps(log_message))
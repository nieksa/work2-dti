import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import time

# 手动指定任务名称
task = "NCvsProdromal"

# 指定日志文件夹路径
log_folder = f"./{task}"
timestamp = time.strftime("%Y%m%d-%H%M%S")
# 保存图表的目录
save_fig_path = f"../results/{task}/model_comparison_{timestamp}"
os.makedirs(save_fig_path, exist_ok=True)

# 从文件名中提取模型名的函数
def extract_model_name(file_name):
    # 使用正则表达式提取模型名（如 ResNet50）
    match = re.search(r"([A-Za-z0-9]+)_\d{8}_\d{6}_\d+\.\d+\.log", file_name)
    if match:
        return match.group(1)
    return None

# 读取日志文件并提取数据的函数
def load_log_data(log_file):
    all_labels = []
    all_preds = []
    all_probs = []
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r'\{.*\}', line)
            if match:
                log_data = json.loads(match.group())
                all_labels.extend(log_data["labels"])
                all_preds.extend(log_data["preds"])
                all_probs.extend(log_data["probs"])
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# 获取文件夹中的所有日志文件
log_files = [os.path.join(log_folder, f) for f in os.listdir(log_folder) if f.endswith(".log")]

# 加载所有日志文件的数据
model_results = {}
for log_file in log_files:
    model_name = extract_model_name(os.path.basename(log_file))
    if model_name:
        labels, preds, probs = load_log_data(log_file)
        model_results[model_name] = (labels, preds, probs)

# 1. ROC 曲线比较
def plot_roc_curve_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(os.path.join(save_path, "roc_curve_comparison.png"))
        plt.close()
    else:
        plt.show()

# 2. Precision-Recall 曲线比较
def plot_precision_recall_curve_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="lower left")
    if save_path:
        plt.savefig(os.path.join(save_path, "precision_recall_curve_comparison.png"))
        plt.close()
    else:
        plt.show()

# 3. 校准曲线比较
def plot_calibration_curve_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        prob_true, prob_pred = calibration_curve(labels, probs[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve Comparison')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "calibration_curve_comparison.png"))
        plt.close()
    else:
        plt.show()

# 4. 累积增益图比较
def plot_cumulative_gain_curve_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        # 按预测概率排序
        sorted_indices = np.argsort(probs[:, 1])[::-1]
        sorted_labels = labels[sorted_indices]

        # 计算累积增益
        cumulative_gains = np.cumsum(sorted_labels) / np.sum(sorted_labels)

        # 计算随机模型的累积增益
        random_gains = np.linspace(0, 1, len(labels))

        # 绘制累积增益曲线
        plt.plot(np.arange(1, len(labels) + 1) / len(labels), cumulative_gains, label=f'{model_name}')

    # 绘制随机模型的累积增益曲线
    plt.plot(np.arange(1, len(labels) + 1) / len(labels), random_gains, linestyle='--', color='gray', label='Random')

    plt.xlabel('Percentage of Samples')
    plt.ylabel('Cumulative Gain')
    plt.title('Cumulative Gain Curve Comparison')
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path, "cumulative_gain_curve_comparison.png"))
        plt.close()
    else:
        plt.show()

# 5. Lift 曲线比较
def plot_lift_curve_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        sorted_indices = np.argsort(probs[:, 1])[::-1]
        sorted_labels = labels[sorted_indices]
        cumulative_gains = np.cumsum(sorted_labels) / np.sum(sorted_labels)
        lift_values = cumulative_gains / (np.arange(1, len(labels) + 1) / len(labels))
        plt.plot(np.arange(1, len(labels) + 1) / len(labels), lift_values, label=model_name)
    plt.axhline(1, linestyle='--', color='gray', label='Random')
    plt.xlabel('Percentage of Samples')
    plt.ylabel('Lift')
    plt.title('Lift Curve Comparison')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "lift_curve_comparison.png"))
        plt.close()
    else:
        plt.show()

# 6. 概率分布图比较
def plot_probability_distribution_multiple_models(model_results, save_path=None):
    plt.figure()
    for model_name, (labels, _, probs) in model_results.items():
        sns.histplot(probs[:, 1], bins=50, kde=True, label=f'{model_name} (Class 1)')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution Comparison')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "probability_distribution_comparison.png"))
        plt.close()
    else:
        plt.show()

# 调用绘图函数并保存图表
plot_roc_curve_multiple_models(model_results, save_fig_path)
plot_precision_recall_curve_multiple_models(model_results, save_fig_path)
plot_calibration_curve_multiple_models(model_results, save_fig_path)
plot_cumulative_gain_curve_multiple_models(model_results, save_fig_path)
plot_lift_curve_multiple_models(model_results, save_fig_path)
plot_probability_distribution_multiple_models(model_results, save_fig_path)

print(f"图表已保存到目录: {save_fig_path}")
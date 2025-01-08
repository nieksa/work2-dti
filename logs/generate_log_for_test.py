import json
import os
import numpy as np
from datetime import datetime

# 设置随机种子以确保可重复性
np.random.seed(42)

# 定义模型名称
model = "Shitty"

# 定义样本数量
n_samples = 243

# 生成随机数据
labels = np.random.randint(0, 2, size=n_samples).tolist()  # 真实标签 (0 或 1)，转换为列表
preds = np.random.randint(0, 2, size=n_samples).tolist()   # 预测标签 (0 或 1)，转换为列表
probs = np.random.rand(n_samples, 2).tolist()              # 预测概率 (二维数组)，转换为列表

# 归一化概率
probs = [[p[0] / (p[0] + p[1]), p[1] / (p[0] + p[1])] for p in probs]

# 生成日志条目
log_entry = {
    "labels": labels,  # 真实标签列表
    "preds": preds,    # 预测标签列表
    "probs": probs     # 预测概率列表
}

# 定义日志文件夹路径
log_folder = "./NCvsProdromal"
os.makedirs(log_folder, exist_ok=True)

# 生成唯一文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 加入毫秒级时间戳
accuracy = np.round(np.mean(np.array(labels) == np.array(preds)) * 100, 2)  # 计算准确率
log_file = os.path.join(log_folder, f"{model}_{timestamp}_{accuracy}.log")

# 写入日志文件
with open(log_file, "w") as f:
    f.write(json.dumps(log_entry) + "\n")

print(f"生成日志文件: {log_file}")
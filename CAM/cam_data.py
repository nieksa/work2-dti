import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from utils.utils import set_seed
import glob
import argparse
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from data.dataset import DTIDataset
from models.utils import create_model
import time

# 定义 Hook 函数
def hook_fn(module, input, output):
    global feature_map
    feature_map = output
    feature_map.retain_grad()  # 保留梯度

# 定义 Grad-CAM 函数
def grad_cam(model, input_tensor, target_class):
    # 前向传播
    output = model(input_tensor)
    model.zero_grad()

    # 计算目标类别的梯度
    target = output[:, target_class].sum()  # 将张量转换为标量
    target.backward()

    # 获取梯度
    gradients = feature_map.grad

    # 计算权重
    weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)

    # 生成热激活图
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = F.relu(cam)  # 去除负值
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)  # 上采样

    # 归一化
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam.detach().cpu().numpy()

# 在 main 函数中使用
def main():
    seed = 42
    set_seed(seed)
    parser = argparse.ArgumentParser(description='Evaluating models.')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='ResNet18')
    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
    parser.add_argument('--val_bs', type=int, default=16, help='densenet cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')
    parser.add_argument('--debug', type=bool, default=False, help='small sample for debugging.')
    parser.add_argument('--csv_file', type=str, default='../data/ppmi/data.csv')
    parser.add_argument('--data_dir', type=str, default='../data/ppmi/')
    parser.add_argument('--save_dir', type=str, default='./results/')
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, args.task, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    args.debug = True

    channels = ["06"]
    dataset = DTIDataset(args.csv_file, args, channels=channels)

    subject_id = np.array(dataset.subject_id)
    unique_ids = np.unique(subject_id)

    k_folds = 2
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_ids)):
        if fold + 1 == args.fold:
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=args.val_bs, shuffle=False)
            break

    model_name = args.model_name
    print(f'Evaluating model {model_name}')
    weights_file_pattern = f"{model_name}_*_fold_{args.fold}_*.pth"
    weights_path_pattern = os.path.join("../saved_models", args.task, weights_file_pattern)
    matching_files = glob.glob(weights_path_pattern)
    if not matching_files:
        print(f"No weights files found in: {weights_path_pattern}")

    weights_path = matching_files[0]
    model = create_model(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 注册 Hook
    global feature_map
    feature_map = None
    hook = model.layer4.register_forward_hook(hook_fn)

    target_class = 1
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 生成热激活图
        cam = grad_cam(model, inputs, target_class)

        # 保存热激活图数据
        for j in range(inputs.shape[0]):  # 遍历 batch 中的每个样本
            cam_data = cam[j].squeeze()  # 热激活图数据 (depth, height, width)
            save_path = os.path.join(save_dir, f"{model_name}_fold{args.fold}_sample{i}_image{j}.npy")
            np.save(save_path, cam_data)

    hook.remove()

if __name__ == '__main__':
    main()
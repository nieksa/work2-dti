import torch
import torch.nn.functional as F
from utils.utils import set_seed
import glob
import argparse
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from data.dataset import DTIDataset
from models.utils import create_model
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import feature, filters

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

def plot_cam(file_path, save_dir, args, input_image):
    # 加载热激活图数据
    cam_data = np.load(file_path, allow_pickle=True)
    print("Heatmap shape:", cam_data.shape)  # 输出形状，例如 (91, 109, 91)

    # 获取中间切面
    depth, height, width = cam_data.shape
    x_slice_cam = cam_data[depth // 2, ...]  # x 切面 (height, width)
    y_slice_cam = cam_data[:, height // 2, :]  # y 切面 (depth, width)
    z_slice_cam = cam_data[:, :, width // 2]  # z 切面 (depth, height)

    # 获取输入图像的中间切面
    x_slice_input = input_image[depth // 2, ...]  # x 切面 (height, width)
    y_slice_input = input_image[:, height // 2, :]  # y 切面 (depth, width)
    z_slice_input = input_image[:, :, width // 2]  # z 切面 (depth, height)

    sigma = 0.5  # 高斯滤波的标准差，控制平滑程度
    low_threshold = 0.1  # 低阈值
    high_threshold = 0.2  # 高阈值

    # 提取 x 切面的轮廓
    x_edge = feature.canny(x_slice_input, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    x_edge_fixed = x_edge.astype(float)  # 转换为浮点型
    x_edge_fixed[x_edge] = 1.0  # 将轮廓固定为深色（1.0）

    # 提取 y 切面的轮廓
    y_edge = feature.canny(y_slice_input, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    y_edge_fixed = y_edge.astype(float)  # 转换为浮点型
    y_edge_fixed[y_edge] = 1.0  # 将轮廓固定为深色（1.0）

    # 提取 z 切面的轮廓
    z_edge = feature.canny(z_slice_input, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    z_edge_fixed = z_edge.astype(float)  # 转换为浮点型
    z_edge_fixed[z_edge] = 1.0  # 将轮廓固定为深色（1.0）

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制并保存 x 切面
    plt.figure()

    # 边缘检测,这里最好能够增强原始图片的边缘。 三个切面都要增强
    plt.imshow(x_edge_fixed, cmap='gray')  # 原始图像
    plt.imshow(x_slice_cam, cmap='jet', alpha=0.9)  # 热力图，设置透明度
    plt.colorbar()
    plt.title(f"X Slice (Depth = {depth // 2})")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{args.model_name}_fold{args.fold+1}_x_slice.png"))  # 保存 x 切面
    plt.close()

    # 绘制并保存 y 切面
    plt.figure()
    plt.imshow(y_edge_fixed, cmap='gray')  # 原始图像
    plt.imshow(y_slice_cam, cmap='jet', alpha=0.9)  # 热力图，设置透明度
    plt.colorbar()
    plt.title(f"Y Slice (Height = {height // 2})")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{args.model_name}_fold{args.fold+1}_y_slice.png"))  # 保存 y 切面
    plt.close()

    # 绘制并保存 z 切面
    plt.figure()
    plt.imshow(z_edge_fixed, cmap='gray')  # 原始图像
    plt.imshow(z_slice_cam, cmap='jet', alpha=0.9)  # 热力图，设置透明度
    plt.colorbar()
    plt.title(f"Z Slice (Width = {width // 2})")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{args.model_name}_fold{args.fold+1}_z_slice.png"))  # 保存 z 切面
    plt.close()

    print(f"切片图像已保存到: {save_dir}")
def main():
    seed = 42
    set_seed(seed)
    parser = argparse.ArgumentParser(description='Evaluating models.')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='ResNet18')
    parser.add_argument('--task', type=str, default='NCvsProdromal', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
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
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 注册 Hook
    global feature_map
    feature_map = None
    hook = model.layer4.register_forward_hook(hook_fn)
    target_class = 1

    first_sample = next(iter(val_loader))
    inputs, labels = first_sample
    inputs = inputs.to(device)
    input_image = inputs[0].squeeze().cpu().numpy()
    cam = grad_cam(model, inputs, target_class)
    cam_data = cam[0].squeeze()
    save_path = os.path.join(save_dir, f"{model_name}_fold{args.fold}_.npy")
    np.save(save_path, cam_data)

    # for i, (inputs, labels) in enumerate(val_loader):
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)
    #     cam = grad_cam(model, inputs, target_class)
    #
    #     for j in range(inputs.shape[0]):  # 遍历 batch 中的每个样本
    #         cam_data = cam[j].squeeze()  # 热激活图数据 (depth, height, width)
    #         save_path = os.path.join(save_dir, f"{model_name}_fold{args.fold}_sample{i}_image{j}.npy")
    #         np.save(save_path, cam_data)

    hook.remove()
    plot_cam(save_path,save_dir=save_dir, args=args, input_image=input_image)

if __name__ == '__main__':
    main()
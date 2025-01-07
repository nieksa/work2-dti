import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 使用新的 weights API 加载预训练的 ResNet50 模型
weights = ResNet50_Weights.IMAGENET1K_V1  # 或者使用 ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
resnet.fc = nn.Identity()  # 去掉最后的全连接层，只提取特征

# 输入数据的形状为 (batch, 3, 128, 128, 128)
input_tensor = torch.randn(2, 3, 128, 128, 128)  # 示例输入 (batch=2)


# 从三个方向切片并提取特征
def extract_features(input_tensor, model):
    batch_size, channels, depth, height, width = input_tensor.shape

    # 从深度方向切片 (128个切片)
    depth_slices = input_tensor.permute(0, 2, 1, 3, 4)  # (batch, depth, channels, height, width)
    depth_slices = depth_slices.reshape(-1, channels, height, width)  # (batch * depth, channels, height, width)
    depth_features = model(depth_slices)  # 提取特征 (batch * depth, feature_dim)
    depth_features = depth_features.reshape(batch_size, depth, -1)  # (batch, depth, feature_dim)

    # 从高度方向切片 (128个切片)
    height_slices = input_tensor.permute(0, 3, 1, 2, 4)  # (batch, height, channels, depth, width)
    height_slices = height_slices.reshape(-1, channels, depth, width)  # (batch * height, channels, depth, width)
    height_features = model(height_slices)  # 提取特征 (batch * height, feature_dim)
    height_features = height_features.reshape(batch_size, height, -1)  # (batch, height, feature_dim)

    # 从宽度方向切片 (128个切片)
    width_slices = input_tensor.permute(0, 4, 1, 2, 3)  # (batch, width, channels, depth, height)
    width_slices = width_slices.reshape(-1, channels, depth, height)  # (batch * width, channels, depth, height)
    width_features = model(width_slices)  # 提取特征 (batch * width, feature_dim)
    width_features = width_features.reshape(batch_size, width, -1)  # (batch, width, feature_dim)

    return depth_features, height_features, width_features


# 提取特征
depth_features, height_features, width_features = extract_features(input_tensor, resnet)

# 合并特征
combined_features = torch.cat([depth_features, height_features, width_features], dim=1)  # (batch, 128 * 3, feature_dim)

print("Depth features shape:", depth_features.shape)  # (batch, 128, feature_dim)
print("Height features shape:", height_features.shape)  # (batch, 128, feature_dim)
print("Width features shape:", width_features.shape)  # (batch, 128, feature_dim)
print("Combined features shape:", combined_features.shape)  # (batch, 384, feature_dim)
import torch
import torch.nn as nn
from einops import rearrange

class ROIEmbed(nn.Module):
    def __init__(self, roi_dim=128):
        super(ROIEmbed, self).__init__()
        self.roi_dim = roi_dim

        # 定义多层卷积网络
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.AdaptiveAvgPool3d(output_size=1)  # 将空间维度压缩为 1
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=self.roi_dim),
        )

    def forward(self, x):
        """
        输入数据的形状是 (batch_size, 3, 180, 180, 180)
        输出数据是 (batch_size, 729, roi_dim)
        """
        # 使用 einops 划分 patch
        patch_size = (20, 20, 20)
        x = rearrange(x, 'b c (h p1) (w p2) (d p3) -> (b h w d) c p1 p2 p3',
                      p1=patch_size[0], p2=patch_size[1], p3=patch_size[2])

        # 调整输入形状为 (batch_size * num_patches, channels, depth, height, width)
        x = x.permute(0, 1, 4, 2, 3)  # 调整为 (batch_size * 729, 3, 20, 20, 20)

        # 通过卷积层和全局平均池化
        x = self.conv_layers(x)  # 输出形状: (batch_size * 729, 256, 1, 1, 1)

        # 调整形状为 (batch_size * 729, 256)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # 去掉空间维度

        # 通过全连接层
        x = self.fc1(x)  # 输出形状: (batch_size * 729, roi_dim)

        # 调整形状为 (batch_size, 729, roi_dim)
        x = x.view(-1, 729, self.roi_dim)  # 恢复 batch 维度

        return x

# 测试代码
if __name__ == "__main__":
    # 假设输入张量的形状为 (batch_size, 3, 180, 180, 180)
    input_tensor = torch.randn(2, 3, 180, 180, 180)

    # 初始化模型
    model = ROIEmbed(roi_dim=128)

    # 前向传播
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 输出: (1, 729, 2048)
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

# ============== 1. 定义 ViT ==============
class Simple3DViT(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, num_heads=4):
        super(Simple3DViT, self).__init__()
        self.proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)  # 线性投影
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=4
        )
        self.output_layer = nn.Conv3d(hidden_dim, 1, kernel_size=1)  # 降维到 1 维通道
    
    def forward(self, x):
        B, C, D, H, W = x.shape  # (batch, 128, 12, 24, 12)
        x = self.proj(x)  # 变换通道数 (B, 128, 12, 24, 12) -> (B, 128, 12, 24, 12)
        x = x.flatten(2).permute(2, 0, 1)  # 变换为 Transformer 输入 (D*H*W, B, C)
        x = self.transformer(x)  # Transformer 处理
        x = x.permute(1, 2, 0).reshape(B, 128, D, H, W)  # 变回 3D 结构
        x = self.output_layer(x)  # 变换通道数到 1 (B, 1, 12, 24, 12)
        return x

# ============== 2. 计算 SSIM ==============
def ssim_3d(tensor1, tensor2):
    """
    计算两个 3D 张量的 SSIM。
    """
    tensor1_np = tensor1.squeeze().cpu().detach().numpy()
    tensor2_np = tensor2.squeeze().cpu().detach().numpy()
    return ssim(tensor1_np, tensor2_np, data_range=tensor2_np.max() - tensor2_np.min())

# ============== 3. 测试代码 ==============
if __name__ == "__main__":
    # 随机生成两个 128, 12, 24, 12 的输入张量
    torch.manual_seed(42)
    tensor_FA = torch.rand(1, 128, 12, 24, 12)
    tensor_MRI = torch.rand(1, 128, 12, 24, 12)

    # 创建 ViT 模型
    model = Simple3DViT()

    # 经过 ViT 处理
    heatmap_FA = model(tensor_FA)  # 输出 (1, 1, 12, 24, 12)
    heatmap_MRI = model(tensor_MRI)  # 输出 (1, 1, 12, 24, 12)

    # 计算 SSIM
    similarity = ssim_3d(heatmap_FA, heatmap_MRI)

    print(f"SSIM 相似度: {similarity:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchToEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels):
        super(PatchToEmbed, self).__init__()
        # 将 patch_size 转换为元组
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # 计算每个 patch 的体积
        self.patch_volume = patch_size ** 3

        # 线性层，将每个 patch 映射到 embed_dim
        self.projection = nn.Linear(self.patch_volume * in_channels, embed_dim)

    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape

        # 检查输入是否为立方体
        if depth != height or height != width:
            raise ValueError("Input data must be a cube (depth == height == width).")

        # 计算需要的 padding 大小
        pad_depth = (self.patch_size[0] - depth % self.patch_size[0]) % self.patch_size[0]
        pad_height = (self.patch_size[1] - height % self.patch_size[1]) % self.patch_size[1]
        pad_width = (self.patch_size[2] - width % self.patch_size[2]) % self.patch_size[2]

        # 对输入数据进行 padding
        x = F.pad(x, (0, pad_width, 0, pad_height, 0, pad_depth))  # 在深度、高度、宽度方向进行 padding

        # 划分 patch
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]) \
            .unfold(3, self.patch_size[1], self.patch_size[1]) \
            .unfold(4, self.patch_size[2], self.patch_size[2])

        # 调整形状为 (batch_size, num_patches, patch_volume * in_channels)
        x = x.permute(0, 2, 3, 4, 1, 5, 6,
                      7).contiguous()  # (batch_size, num_patches, in_channels, patch_size[0], patch_size[1], patch_size[2])
        x = x.view(batch_size, -1, self.patch_volume * in_channels)

        # 映射到嵌入维度
        x = self.projection(x)  # (batch_size, num_patches, embed_dim)

        return x


class EmbedToPatch(nn.Module):
    def __init__(self, patch_size, embed_dim, out_channels):
        super(EmbedToPatch, self).__init__()
        # 将 patch_size 转换为元组
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # 线性层，将 embed_dim 映射回 patch_volume * out_channels
        self.projection = nn.Linear(embed_dim, patch_size ** 3 * out_channels)

    def forward(self, x, original_shape):
        batch_size, num_patches, embed_dim = x.shape
        _, _, depth, height, width = original_shape  # 原始空间维度

        # 检查输入是否为立方体
        if depth != height or height != width:
            raise ValueError("Input data must be a cube (depth == height == width).")

        # 计算需要的 padding 大小
        pad_depth = (self.patch_size[0] - depth % self.patch_size[0]) % self.patch_size[0]
        pad_height = (self.patch_size[1] - height % self.patch_size[1]) % self.patch_size[1]
        pad_width = (self.patch_size[2] - width % self.patch_size[2]) % self.patch_size[2]

        # 映射回 patch_volume * out_channels
        x = self.projection(x)  # (batch_size, num_patches, patch_volume * out_channels)

        # 调整形状为 (batch_size, num_patches, patch_size[0], patch_size[1], patch_size[2], out_channels)
        x = x.view(batch_size, num_patches, self.patch_size[0], self.patch_size[1], self.patch_size[2],
                   self.out_channels)

        # 重建 3D 数据
        x = x.permute(0, 5, 1, 2, 3,
                      4).contiguous()  # (batch_size, out_channels, num_patches, patch_size[0], patch_size[1], patch_size[2])
        x = x.view(batch_size, self.out_channels, depth + pad_depth, height + pad_height,
                   width + pad_width)  # (batch_size, out_channels, padded_depth, padded_height, padded_width)

        # 去除 padding
        x = x[:, :, :depth, :height, :width]  # (batch_size, out_channels, depth, height, width)

        return x


class Upsample3D(nn.Module):
    def __init__(self, out_channels, out_size):
        super(Upsample3D, self).__init__()
        self.out_channels = out_channels
        # 将 out_size 转换为元组
        self.out_size = (out_size, out_size, out_size)
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=self.out_size, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)  # 3D 卷积
        x = self.upsample(x)  # 上采样到目标形状
        return x


class Transformer3D(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels, out_channels, out_size, num_heads, num_layers):
        super(Transformer3D, self).__init__()
        # 确保 num_heads 是偶数
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Nested Tensor optimization.")

        self.patch_to_embed = PatchToEmbed(patch_size, embed_dim, in_channels)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            batch_first=True,  # 启用 batch_first
        )
        self.embed_to_patch = EmbedToPatch(patch_size, embed_dim, out_channels)
        self.upsample = Upsample3D(out_channels, out_size)

    def forward(self, x):
        original_shape = x.shape  # 保存原始形状
        # 检查输入是否为立方体
        if original_shape[2] != original_shape[3] or original_shape[3] != original_shape[4]:
            raise ValueError("Input data must be a cube (depth == height == width).")

        # PatchToEmbed
        x = self.patch_to_embed(x)  # (batch, num_patches, embed_dim)
        # Transformer
        x = self.transformer(x, x)  # (batch, num_patches, embed_dim)
        # EmbedToPatch
        x = self.embed_to_patch(x, original_shape)  # (batch, out_channels, depth, height, width)
        # Upsample
        x = self.upsample(x)  # (batch, out_channels, out_size, out_size, out_size)
        return x


# 示例使用
patch_size = 3  # 每个 patch 的大小（立方体）
embed_dim = 64  # 嵌入维度
in_channels = 128  # 输入数据的通道数
out_channels = 256  # 输出数据的通道数
out_size = 12  # 目标输出形状（立方体）
num_heads = 2  # Transformer 的头数（偶数）
num_layers = 1  # Transformer 的层数

# 创建模型
model = Transformer3D(patch_size, embed_dim, in_channels, out_channels, out_size, num_heads, num_layers)

# 输入数据 (batch_size, channels, depth, height, width)
input_data = torch.randn(1, in_channels, 23, 23, 23)  # 47 不能被 5 整除

# 前向传播
output_data = model(input_data)
print(output_data.shape)  # 输出形状应为 (1, out_channels, out_size, out_size, out_size)
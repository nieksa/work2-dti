import torch
import torch.nn as nn

"""
这个模块实现了一个3d transformer encoder机制
保持三维数据的输入 输出尺寸
通道数是被out conv改变的 可以略作改动
"""

class SelfAttention3D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # (b, 3*dim, d, h, w)
        q, k, v = map(lambda t: t.reshape(b, self.heads, c // self.heads, d * h * w), qkv)
        # (batch, heads, heads_dim, seq_len)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(b, c, d, h, w)
        return self.proj(out)


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, image_size=64):
        super().__init__()
        self.attn = SelfAttention3D(dim, heads, dim_head)
        self.norm1 = nn.LayerNorm([dim, image_size, image_size, image_size])
        self.norm2 = nn.LayerNorm([dim, image_size, image_size, image_size])

        self.mlp = nn.Sequential(
            nn.Conv3d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(mlp_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x))  # Attention + Residual
        x = self.norm2(x + self.mlp(x))  # MLP + Residual
        return x


class Transformer3D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, image_size):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock3D(dim, heads, dim_head, mlp_dim, image_size) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MRI_Transformer(nn.Module):
    def __init__(self, in_channels=1, image_size=64, dim=64, depth=4, heads=8, dim_head=8, mlp_dim=64):
        super().__init__()
        self.embed = nn.Conv3d(in_channels, dim, kernel_size=1)  # kernel_size=1代替线性映射
        self.transformer = Transformer3D(dim, depth, heads, dim_head, mlp_dim, image_size)

    def forward(self, x):
        x = self.embed(x)  # Embedding block
        x = self.transformer(x)  # Transformer blocks
        return x

if __name__ == '__main__':
    batch_size = 2
    input_channels = 128
    output_channels = 128
    image_size = 4
    model = MRI_Transformer(in_channels=input_channels, image_size=image_size, dim=output_channels, depth=3, heads=4, dim_head=4, mlp_dim=64)
    mri_data = torch.randn(batch_size, input_channels, image_size, image_size, image_size)
    output = model(mri_data)
    print(mri_data.shape)
    print(output.shape)

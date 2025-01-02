from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=2):
        super(LogisticRegressionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class WeightedAveragePoolingClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=2, heads=4, num_layers=2, hidden_dim=128):
        super(WeightedAveragePoolingClassifier, self).__init__()

        # 使用自注意力计算权重
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(0)  # 变成 (1, batch_size, dim) 以便做transformer处理
        attn_output, _ = self.attn(x, x, x)  # 自注意力
        x = attn_output.squeeze(0)  # 恢复为 (batch_size, dim)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
class FCClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2, dropout=0.5):
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)  # 压缩维度
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)  # 恢复维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1)
        y = y.view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)

        x = x * y  # 结果 (b, c, t)

        x = x.sum(dim=1, keepdim=True)
        x = x.squeeze(dim=1)
        return x


class ViT(nn.Module):

    def __init__(self, *, image_size=32, image_patch_size=4, frames=32, frame_patch_size=4, dim=64,
                 depth=3, heads=4, mlp_dim=64, pool='cls', channels=256, dim_head=32, dropout=0.2, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.ca = ChannelAttention(channel=2 * channels + 1)
        self.to_latent = nn.Identity()

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        x = self.ca(x)
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     dilation=1,
                     bias=False)


def conv7x7x7(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     dilation=1,
                     bias=False)


class SmallVoxelResNet(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks):
        super(SmallVoxelResNet, self).__init__()
        self.conv1 = conv3x3x3(in_planes, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 128, num_blocks[0], stride=2)  # 64 -> 128, stride 2
        self.layer2 = self._make_layer(128, 128, num_blocks[1], stride=2)  # 128 -> 128, stride 2
        self.layer3 = self._make_layer(128, 128, num_blocks[2], stride=1)  # 128 -> 128, stride 1 (保持大小不变)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _make_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            conv3x3x3(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            conv3x3x3(out_planes, out_planes),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class LargeVoxelResNet(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks):
        super(LargeVoxelResNet, self).__init__()
        self.conv1 = conv7x7x7(in_planes, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 128, num_blocks[0], stride=2)  # 64 -> 128, stride 2
        self.layer2 = self._make_layer(128, 128, num_blocks[1], stride=2)  # 128 -> 128, stride 2
        self.layer3 = self._make_layer(128, 128, num_blocks[2], stride=1)  # 128 -> 128, stride 1 (保持大小不变)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _make_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            conv7x7x7(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            conv7x7x7(out_planes, out_planes),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DualBranchResNet(nn.Module):
    # (batch, channel = 1, 128, 128, 128) -> (batch, 256, 32, 32, 32)
    # 一半通道是大体素特征， 一半是小体素特征
    def __init__(self, in_channels, out_channels, num_blocks):
        super(DualBranchResNet, self).__init__()
        # 小体素分支
        self.small_voxel_branch = SmallVoxelResNet(in_channels, out_channels, num_blocks)
        # 大体素分支
        self.large_voxel_branch = LargeVoxelResNet(in_channels, out_channels, num_blocks)
    def forward(self, x):
        small_out = self.small_voxel_branch(x)
        large_out = self.large_voxel_branch(x)
        out = torch.cat((small_out, large_out), dim=1)  # 拼接通道维度
        return out

class Design1 (nn.Module):
    def __init__ (self, in_channels=1, out_channel=128, class_num=2, num_blocks=[1,1,1]):
        super().__init__()
        self.dual_resnet = DualBranchResNet(in_channels=in_channels, out_channels=out_channel, num_blocks=num_blocks)
        self.vit = ViT(channels=out_channel*2)
        self.classifier = LogisticRegressionClassifier(input_dim=64, output_dim=class_num)
    def forward(self, x):
        x = self.dual_resnet(x)
        x = self.vit(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Design1(in_channels=1, class_num=2, num_blocks=[1,1,1]).to(device)
    print(model)
    x = torch.randn(8, 1, 128, 128, 128).to(device)
    out = model(x)
    print(out.shape)
# 这个改造一下，做一个多尺度融合模块，256，45，45，45 + 512,23,23,23 + 1024,12,12,12 + 2048,6,6,6

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from gevent.monkey import patch_thread


class BinaryClassificationLayer(nn.Module):
    def __init__(self, in_channels=512):
        super(BinaryClassificationLayer, self).__init__()
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # 全连接层
        self.fc = nn.Linear(in_channels, 1)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状: (batch, 512, 6, 6, 6)
        # 全局平均池化
        x = self.global_avg_pool(x)  # 输出形状: (batch, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)  # 展平为 (batch, 512)
        # 全连接层
        x = self.fc(x)  # 输出形状: (batch, 1)
        # Sigmoid 激活函数
        x = self.sigmoid(x)  # 输出形状: (batch, 1)
        return x


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

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        # 3D 卷积层
        self.conv = nn.Conv3d(
            inp_dim, out_dim, kernel_size, stride,
            padding=(kernel_size - 1) // 2, bias=bias, groups=group
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, f"Input channel {x.size()[1]} does not match inp_dim {self.inp_dim}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, depth, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, depth, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # 使用 PyTorch 的 layer_norm，支持多维输入
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, depth, height, width]
            mean = x.mean(1, keepdim=True)  # 计算均值，保留通道维度
            var = (x - mean).pow(2).mean(1, keepdim=True)  # 计算方差，保留通道维度
            x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
            # 扩展 weight 和 bias 到 3D 数据的维度
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm3d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool3d(1)
        self.avgpool=nn.AdaptiveAvgPool3d(1)
        self.se=nn.Sequential(
            nn.Conv3d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv3d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool3d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            _,_,f_D,f_H,f_W = W_f.shape
            pad_depth = 1 if f_D % 2 != 0 else 0
            pad_height = 1 if f_H % 2 != 0 else 0
            pad_width = 1 if f_W % 2 != 0 else 0
            if pad_depth or pad_height or pad_width:
                W_f = F.pad(W_f, (0, pad_width, 0, pad_height, 0, pad_depth))
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result=self.maxpool(g)
        avg_result=self.avgpool(g)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        g = self.sigmoid(max_out+avg_out) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     dilation=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv7x7x7(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=1,
                     dilation=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.hff1 = HFF_block(ch_1=64, ch_2=64, r_2=16, ch_int=64, ch_out=64, drop_rate=0.1)
        self.hff2 = HFF_block(ch_1=128, ch_2=128, r_2=16, ch_int=128, ch_out=128, drop_rate=0.1)
        self.hff3 = HFF_block(ch_1=256, ch_2=256, r_2=16, ch_int=256, ch_out=256, drop_rate=0.1)
        self.hff4 = HFF_block(ch_1=512, ch_2=512, r_2=16, ch_int=512, ch_out=512, drop_rate=0.1)

        self.t_layer1 = Transformer3D(patch_size=5, embed_dim=16,
                                      in_channels=64,
                                      out_channels=64,
                                      out_size=45,
                                      num_heads=8, num_layers=1)
        self.t_layer2 = Transformer3D(patch_size=3, embed_dim=16,
                                      in_channels=64,
                                      out_channels=128,
                                      out_size=23,
                                      num_heads=8, num_layers=1)
        self.t_layer3 = Transformer3D(patch_size=2, embed_dim=16,
                                      in_channels=128,
                                      out_channels=256,
                                      out_size=12,
                                      num_heads=8, num_layers=1)
        self.t_layer4 = Transformer3D(patch_size=1, embed_dim=16,
                                      in_channels=256,
                                      out_channels=512,
                                      out_size=6,
                                      num_heads=8, num_layers=1)

        self.out = BinaryClassificationLayer(in_channels=512)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def get_resnet_stage_feature(self,x):
        x_s1 = self.layer1(x)
        x_s2 = self.layer2(x_s1)
        x_s3 = self.layer3(x_s2)
        x_s4 = self.layer4(x_s3)
        return x_s1, x_s2, x_s3, x_s4

    def get_transformer_stage_feature(self,x):
        x_t1 = self.t_layer1(x)
        x_t2 = self.t_layer2(x_t1)
        x_t3 = self.t_layer3(x_t2)
        x_t4 = self.t_layer4(x_t3)
        return x_t1, x_t2, x_t3, x_t4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        # patches, D, H, W = self.get_patches(x)
        x_s1,x_s2,x_s3,x_s4 = self.get_resnet_stage_feature(x)

        x_t1,x_t2,x_t3,x_t4 = self.get_transformer_stage_feature(x)

        x_f1 = self.hff1(x_s1,x_t1,None)
        x_f2 = self.hff2(x_s2,x_t2,x_f1)
        x_f3 = self.hff3(x_s3,x_t3,x_f2)
        x_f4 = self.hff4(x_s4,x_t4,x_f3)

        out = self.fc(x_f4)
        return out



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 180
    x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    model = generate_model(18, n_input_channels=3,n_classes=2).to(device)
    print(model)
    out1 = model(x)
    print("out size: {}".format(out1.size()))
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
双分支一路resnet 一路transformer
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
    def __init__(self, in_channels=1, out_channels=1, image_size=64, dim=64, depth=4, heads=8, dim_head=8, mlp_dim=64):
        super().__init__()
        self.embed = nn.Conv3d(in_channels, dim, kernel_size=1)  # kernel_size=1代替线性映射
        self.transformer = Transformer3D(dim, depth, heads, dim_head, mlp_dim, image_size)
        self.out_conv = nn.Conv3d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.embed(x)  # Embedding block
        x = self.transformer(x)  # Transformer blocks
        x = self.out_conv(x)  # Project back to original channels
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.dp = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3, 4).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3, 4).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1, 4).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1, 4).contiguous()

        x_perm3 = x.permute(0, 4, 2, 3, 1).contiguous()
        x_out3 = self.hc(x_perm3)
        x_out31 = x_out3.permute(0, 4, 2, 3, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)  # 应用空间注意力
            x_out = 1 / 4 * (x_out + x_out11 + x_out21 + x_out31)
        else:
            x_out = 1 / 3 * (x_out11 + x_out21 + x_out31)
        return x_out


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
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
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
        self.fc = nn.Linear(2*block_inplanes[3] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()

        self.tf1 = MRI_Transformer(in_channels=block_inplanes[0], out_channels=block_inplanes[1], image_size=32,
                                   dim=2*block_inplanes[0], depth=1, heads=8, dim_head=16, mlp_dim=2*block_inplanes[0])

        self.tf2 = MRI_Transformer(in_channels=block_inplanes[1], out_channels=block_inplanes[2], image_size=16,
                                   dim=2*block_inplanes[1], depth=1, heads=16, dim_head=16, mlp_dim=2*block_inplanes[1])

        self.tf3 = MRI_Transformer(in_channels=block_inplanes[2], out_channels=block_inplanes[3], image_size=8,
                                   dim=2*block_inplanes[2], depth=1, heads=16, dim_head=32, mlp_dim=2*block_inplanes[2])
        self.ta = TripletAttention(no_spatial=False)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x):

        #  1 128 128 128
        x = self.conv1(x)  # 64 64 64 64
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)  # 64 32 32 32

        x = self.layer1(x)  # 64 32 32 32

        x_t = x
        x_t = self.maxpool(self.tf1(x_t))
        x_t = self.maxpool(self.tf2(x_t))
        x_t = self.maxpool(self.tf3(x_t))

        x = self.layer2(x)  # 128 16 16 16
        x = self.layer3(x)  # 256 8 8 8
        x = self.layer4(x)  # 512 4 4 4

        x = torch.concat((x,x_t), dim =1)   # 1024 4 4 4
        x = self.ta(x)  # 这里有一个明显的改进方式，MDL那篇文章画图一样，我们可以当作双分支，多尺度特征提取，分别走不同的注意力通道融合，类似Y型结构

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x



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
    image_size = 128
    x = torch.Tensor(3, 1, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))
    model = generate_model(18)
    out1 = model(x)
    print("out size: {}".format(out1.size()))
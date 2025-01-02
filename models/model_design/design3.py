import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resnet 经典结构，但是每一个layer后面都接上一个 coord attention3d联合多个方向的空间特征和通道注意力，加深网络
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

class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt3D, self).__init__()
        # 自适应平均池化操作
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))  # 深度方向
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))  # 高度方向
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))  # 宽度方向

        mip = max(8, inp // reduction)  # 中间层通道数

        # 降维
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.Hardswish()

        # 用于生成深度、高度、宽度方向的注意力权重
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 保留输入作为残差连接
        n, c, d, h, w = x.size()  # 获取输入尺寸

        # 分别在深度、高度和宽度方向进行池化
        x_d = self.pool_d(x)  # (n, c, d, 1, 1)
        x_h = self.pool_h(x)  # (n, c, 1, h, 1)
        x_w = self.pool_w(x)  # (n, c, 1, 1, w)

        # 调整维度以便拼接
        x_h = x_h.permute(0, 1, 3, 2, 4)  # 交换高度和深度维度
        x_w = x_w.permute(0, 1, 4, 2, 3)  # 交换宽度和深度维度

        # 拼接深度、高度、宽度方向的特征
        y = torch.cat([x_d, x_h, x_w], dim=2)  # 在深度维度上拼接

        # 降维和激活
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分割回深度、高度、宽度方向
        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)

        # 恢复原始维度
        x_h = x_h.permute(0, 1, 3, 2, 4)  # 恢复高度维度
        x_w = x_w.permute(0, 1, 4, 2, 3)  # 恢复宽度维度

        # 分别计算注意力权重
        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 应用注意力权重并与残差相乘
        out = identity * a_d * a_h * a_w

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
        # self.layer4 = self._make_layer(block,
        #                                block_inplanes[3],
        #                                layers[3],
        #                                shortcut_type,
        #                                stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[2] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.coordatt1 = CoordAtt3D(64, 64)
        self.coordatt2 = CoordAtt3D(128, 128)
        self.coordatt3 = CoordAtt3D(256, 256)

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
        x = self.coordatt1(x)
        x = self.layer2(x)  # 128 16 16 16
        x = self.coordatt2(x)
        x = self.layer3(x)  # 256 8 8 8
        x = self.coordatt3(x)
        # x = self.layer4(x) #512 4 4 4

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


# if __name__ == '__main__':
#     coord_att_3d = CoordAtt3D(inp=64, oup=64, reduction=32)
#     input = torch.rand(1, 64, 128, 128, 128)  # 创建一个随机输入
#     output = coord_att_3d(input)  # 通过模块处理输入
#     print("Input shape:", input.shape)
#     print("Output shape:", output.shape)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1,n_classes=2)
    out1 = model(x)
    print("out size: {}".format(out1.size()))
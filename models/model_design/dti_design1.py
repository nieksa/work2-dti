from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from models.model_design.module.MIL import MIL

class CrossStageFusionModule(nn.Module):
    def __init__(self, feature_dim=128):
        super(CrossStageFusionModule, self).__init__()
        self.feature_dim = feature_dim

        # 用于融合的特征变换层
        self.fc_mil2 = nn.Linear(feature_dim, feature_dim)
        self.fc_mil3 = nn.Linear(feature_dim, feature_dim)
        self.fc_mil4 = nn.Linear(feature_dim, feature_dim)
        self.fc_x4 = nn.Linear(feature_dim, feature_dim)

        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 4),  # 输入是拼接后的特征
            nn.ReLU(),
            nn.Linear(feature_dim * 4, 4),  # 输出4个权重（对应mil2, mil3, mil4, x4）
            nn.Softmax(dim=1)  # 对权重进行归一化
        )

        # 最终的特征变换层
        self.fc_final = nn.Linear(feature_dim, feature_dim)

    def forward(self, mil2, mil3, mil4, x4):
        # 输入形状: mil2, mil3, mil4, x4 都是 [batch_size, feature_dim]
        batch_size = mil2.size(0)

        # 对每个特征进行变换
        mil2_transformed = self.fc_mil2(mil2)  # [batch_size, feature_dim]
        mil3_transformed = self.fc_mil3(mil3)  # [batch_size, feature_dim]
        mil4_transformed = self.fc_mil4(mil4)  # [batch_size, feature_dim]
        x4_transformed = self.fc_x4(x4)  # [batch_size, feature_dim]

        # 拼接所有特征
        features = torch.cat([mil2_transformed, mil3_transformed, mil4_transformed, x4_transformed], dim=1)  # [batch_size, feature_dim * 4]

        # 计算注意力权重
        attention_weights = self.attention(features)  # [batch_size, 4]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, 4, 1]

        # 加权求和
        weighted_features = (
            attention_weights[:, 0] * mil2_transformed +
            attention_weights[:, 1] * mil3_transformed +
            attention_weights[:, 2] * mil4_transformed +
            attention_weights[:, 3] * x4_transformed
        )  # [batch_size, feature_dim]

        # 最终特征变换
        output = self.fc_final(weighted_features)  # [batch_size, feature_dim]
        return output

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
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[2] * block.expansion, n_classes)

        self.fc = nn.Linear(128, n_classes)

        # Local positive detector
        self.mil2 = MIL(channels=128, num_instances=12*14*12)
        self.mil3 = MIL(channels=256, num_instances=6*7*6)
        self.mil4 = MIL(channels=512, num_instances=3*4*3)

        self.resnet_out_fc = nn.Linear(block_inplanes[3] * block.expansion,128)

        # Cross stage fusion module
        self.csf = CrossStageFusionModule(feature_dim=128)

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
    def feature_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

    def forward(self, x):
        _, x2, x3, x4 = self.feature_forward(x)
        mil2 = self.mil2(x2)
        mil3 = self.mil3(x3)
        mil4 = self.mil4(x4)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.resnet_out_fc(x4)
        x = self.csf(mil2, mil3, mil4, x4) # 输入3个 batch, 128 的tensor 通过一个复杂的 cross stage fusion module变成一个 batch, 128的tensor
        return self.fc(x)



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
    # image_size = 128
    # x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = torch.Tensor(1,1,91,109,91)
    x = x.to(device)
    print("x size: {}".format(x.size()))
    model = generate_model(18)
    out1 = model(x)
    print("out size: {}".format(out1.size()))
    print(out1)
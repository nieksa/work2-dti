from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial

import torch.nn.functional as F
# helpers
import torch
import torch.nn as nn


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

class Simam3DModule(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam3DModule, self).__init__()
        self.act = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 定义平滑项e_lambda，防止分母为0

    def forward(self, x):
        b, c, d, h, w = x.size()  # 获取输入x的尺寸
        n = w * h * d - 1  # 计算特征图的元素数量减一，用于下面的归一化

        # 计算输入特征x与其均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        # 计算注意力权重y，这里实现了SimAM的核心计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

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

    def adjust_size_if_odd(self, W_f):
        depth, height, width = W_f.shape[2:]
        adjusted_shape = [dim + 1 if dim % 2 != 0 else dim for dim in [depth, height, width]]

        if (depth, height, width) != tuple(adjusted_shape):
            W_f = F.interpolate(W_f, size=adjusted_shape, mode='trilinear', align_corners=False)

        return W_f
    def forward(self, l, g, f):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.adjust_size_if_odd(W_f)
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

# if __name__ == "__main__":
#     model = HFF_block(ch_1=64, ch_2=64, r_2=64, ch_int=64, ch_out=64)
#     g = torch.randn(1, 64, 45, 45, 45)
#     l = torch.randn(1, 64, 45, 45, 45)
#     f = torch.randn(1, 32, 90, 90, 90)
#     f = None
#
#     out = model(l, g, f)
#     print(out.shape)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size,
                 dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,first = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        frames = image_height
        frame_patch_size = patch_height
        if image_size % image_patch_size != 0:
            # 说明要padding
            self.padding_size = (image_patch_size - (image_size % image_patch_size))
        else:
            self.padding_size = 0

        if first:
            out_channel = channels
        else:
            out_channel = channels * 2
        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        image_height += self.padding_size
        image_width += self.padding_size
        frames += self.padding_size

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.ConstantPad3d((0, self.padding_size, 0, self.padding_size, 0, self.padding_size), value=0),
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


        if image_size % 2 != 0:
            self.out_image_size = (image_size // 2 + 1)
        else:
            self.out_image_size = (image_size // 2)


        self.to_3d = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',
                      f=frames // frame_patch_size, h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.Conv3d(in_channels=channels, out_channels=out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=(self.out_image_size, self.out_image_size, self.out_image_size)),
        )


        self.first = first
        self.to_3d_origin = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',
                      f=frames // frame_patch_size, h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.Conv3d(in_channels=channels,out_channels=channels,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0)),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=(image_size,image_size,image_size)),
        )


    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 1:, :]
        if self.first:
            x = self.to_3d_origin(x)
        else:
            x = self.to_3d(x)
        return x

# if __name__ == "__main__":
#     model = ViT(image_size=23, image_patch_size=3, frames=23, frame_patch_size=3, dim=128, depth=1,
#                 heads=4, mlp_dim=128, pool = 'cls', channels = 3, dim_head = 32, dropout = 0.1, emb_dropout = 0.1)
#     x = torch.randn(2, 3, 23, 23, 23)
#     y = model(x)
#     print(y.shape)
# ----------------------------------------------------------------------------------------------------------------------



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

        self.t_layer1 = ViT(image_size=23, image_patch_size=3,
                            dim=512, depth=2, heads=4,mlp_dim=512, pool = 'cls',
                            channels = 64, dim_head = 32, dropout = 0.1, emb_dropout = 0.1,
                            first=True)

        self.t_layer2 = ViT(image_size=23, image_patch_size=3,
                            dim=512, depth=2, heads=4, mlp_dim=512, pool='cls',
                            channels=64, dim_head=128, dropout=0.1, emb_dropout=0.1)

        self.t_layer3 = ViT(image_size=12, image_patch_size=2,
                            dim=128, depth=2, heads=2,mlp_dim=512, pool='cls',
                            channels=128, dim_head=256, dropout=0.1, emb_dropout=0.1)

        self.t_layer4 = ViT(image_size=6, image_patch_size=1,
                            dim=128, depth=2, heads=2,mlp_dim=512, pool='cls',
                            channels=256, dim_head=512, dropout=0.1, emb_dropout=0.1)

        self.fusion1 = HFF_block(ch_1=64, ch_2=64, r_2=64, ch_int=64, ch_out=64)
        self.fusion2 = HFF_block(ch_1=128, ch_2=128, r_2=128, ch_int=128, ch_out=128)
        self.fusion3 = HFF_block(ch_1=256, ch_2=256, r_2=256, ch_int=256, ch_out=256)
        self.fusion4 = HFF_block(ch_1=512, ch_2=512, r_2=512, ch_int=512, ch_out=512)
        self.simam3D = Simam3DModule(e_lambda=1e-4)

        self.tripletatt1 = TripletAttention()
        self.tripletatt2 = TripletAttention()
        self.tripletatt3 = TripletAttention()
        self.tripletatt4 = TripletAttention()

        self.proj = nn.Linear(block_inplanes[3] * block.expansion, 128)

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

    def get_resnet_features(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

    def get_transformer_features(self, x):
        x1 = self.t_layer1(x)
        x1 = self.tripletatt1(x1)
        x2 = self.t_layer2(x1)
        x2 = self.tripletatt2(x2)
        x3 = self.t_layer3(x2)
        x3 = self.tripletatt3(x3)
        x4 = self.t_layer4(x3)
        x4 = self.tripletatt4(x4)
        return x1, x2, x3, x4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1,x2,x3,x4 = self.get_resnet_features(x)
        t1,t2,t3,t4 = self.get_transformer_features(x)

        f1 = self.simam3D(self.fusion1(x1,t1,None))   # 64 23 23 23
        f2 = self.simam3D(self.fusion2(x2,t2,f1)) # 128 12 12 12
        f3 = self.simam3D(self.fusion3(x3,t3,f2)) # 256 6 6 6
        f4 = self.simam3D(self.fusion4(x4,t4,f3)) # 512 3 3 3

        x = self.avgpool(f4)
        x = x.view(x.size(0), -1)
        proj = self.proj(x).squeeze(1)
        x = self.fc(x)

        return x,proj


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

def get_model_memory(model):
    # 计算模型中每个参数的内存占用
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # 将字节转换为 MB
    param_size_MB = param_size / 1024**2
    return param_size_MB

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = 1
    channel = 1
    image_size = 90
    x = torch.Tensor(batch, channel, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=channel,n_classes=2).to(device)
    memory_size = get_model_memory(model)
    print(f"模型参数占用的内存大小: {memory_size:.2f} MB")
    # print(model)
    class_out, proj = model(x)
    print("out size: {}".format(proj.size()))
    # print(proj)


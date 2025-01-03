import time
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
# from module.scale_fusion import LocalAwareLearning

# 这个是3模态的数据
# 其实也是三个resnet分支，最多设计的是 stage上的模态fusion(LAL,三个不同shape，最终输出是4，4，4) 和 最终out的fusion (GAL)，
# 我目前是通过copy dim=1 模拟多模态的情况，然后暂时不考虑ROI的部分
from thop import profile
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch import einsum

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size, img_size, patch_size, types=0):
        super().__init__()
        img_size = img_size
        patch_size = patch_size

        if types == 0:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]) * (
                    img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), hidden_size))
        elif types == 1:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), hidden_size))
        elif types == 2:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]), hidden_size))

    def forward(self, x):
        return x + self.position_embeddings


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            min_indice = self.relative_indices.min()
            self.relative_indices += (-min_indice)
            max_indice = self.relative_indices.max().item()
            self.pos_embedding = nn.Parameter(torch.randn(max_indice + 1, max_indice + 1, max_indice + 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]
        nw_d = n_d // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1], self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2], nw_h=nw_h,
                        nw_w=nw_w, nw_d=nw_d)
        out = self.to_out(out)

        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if attention:
            return attention_output, attention_probs
        return attention_output


class SelfAdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, img_size, patch_size, num_heads, attention_dropout_rate, window_size,
                 pos_embedding=True):
        super().__init__()
        self.attention_x = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.attention_y = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.attention_z = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.window_attention = WindowAttention(dim=hidden_size, heads=num_heads, head_dim=hidden_size // num_heads,
                                                window_size=window_size, relative_pos_embedding=True)
        self.norm = nn.Softmax(dim=-1)
        self.is_position = pos_embedding
        if pos_embedding is True:
            self.pos_embedding1 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)
            self.pos_embedding2 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)
            self.pos_embedding3 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)

    def forward(self, x, y, z):
        B, C, D, H, W = x.shape
        w = x + y + z
        x1 = rearrange(x, "b c d h w -> b (w d h) c")
        y1 = rearrange(y, "b c d h w -> b (w d h) c")
        z1 = rearrange(z, "b c d h w -> b (w d h) c")
        w = w.permute(0, 2, 3, 4, 1)

        if self.is_position:
            x1 = self.pos_embedding1(x1)
            y1 = self.pos_embedding2(y1)
            z1 = self.pos_embedding2(z1)

        x1 = self.attention_x(x1)
        y1 = self.attention_y(y1)
        z1 = self.attention_z(z1)
        w = self.window_attention(w)

        w = rearrange(w, "b d h w c -> b (h w d) c", d=D, h=H, w=W)
        # x3 = rearrange(x3, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)
        # x2 = rearrange(x2, "(b h) (d w) c -> b (h w d) c", d=D, h=H, w=W)
        # x1 = rearrange(x1, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)

        h = self.norm(x1 * y1) * w
        h = self.norm(h)
        out = (h * z1) * w

        return out


class SelfAdaptiveTransformer(nn.Module):
    def __init__(self, hidden_size=128, img_size=(128, 128, 128), patch_size=(8, 8, 8), num_heads=4,
                 attention_dropout_rate=0.2, window_size=(4, 4, 4)):
        super().__init__()
        self.mda = SelfAdaptiveAttention(hidden_size=hidden_size, img_size=img_size, patch_size=patch_size,
                                        num_heads=num_heads,
                                        attention_dropout_rate=attention_dropout_rate, window_size=window_size,
                                        pos_embedding=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, hidden_size * 2, dropout_rate=0.2)

    def forward(self, x, y, z):
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b (h w d) c")
        x = self.norm(x)
        y = rearrange(y, "b c d h w -> b (h w d) c")
        y = self.norm(y)
        z = rearrange(z, "b c d h w -> b (h w d) c")
        z = self.norm(z)
        h = x + y + z
        x = rearrange(x, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        y = rearrange(y, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        z = rearrange(z, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        x1 = self.mda(x, y, z)
        x2 = h + x1
        out = self.norm(x2)
        out = self.mlp(out)
        out = x2 + out

        return out




class Disease_Guide_ROI(nn.Module):
    def __init__(self, dim, heads, i=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.iter = i
        self.x1 = 0.
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cls = nn.Linear(dim, dim, bias=qkv_bias)
        mean = nn.Parameter(torch.randn(1, 1, dim))
        std = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        self.weight = nn.Parameter(torch.normal(mean, std))
        # nn.init.uniform_(self.weight)
        self.gru = nn.GRU(90, 90)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cls_out):
        # x (B, seq, dim)
        B, N, C = x.shape
        weight = self.weight.expand(B, -1, -1).contiguous()
        for _ in range(self.iter):
            weight_prev = weight
            weight1 = weight.reshape(B, self.num_heads, 1, C // self.num_heads)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            q = self.cls(cls_out).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale * weight1
            k = k * weight1
            attn = (q @ k.transpose(-2, -1))
            attn_s = self.softmax(attn)  # (B heads seq seq)
            attn = self.attn_drop(attn_s)
            x1 = attn @ (v * weight1)
            x1 = x1.transpose(1, 2).reshape(B, N, C)
            x1 = x1.transpose(0, 1)
            self.gru.flatten_parameters()
            weight, _ = self.gru(x1, weight_prev.transpose(0, 1))
            weight = weight.transpose(0, 1)

        x = self.proj(x1.transpose(0, 1))
        x = self.proj_drop(x)
        x = x.squeeze(dim=1)

        return x
class FactorizedBilinearPooling(nn.Module):
    def __init__(self, channels):
        super(FactorizedBilinearPooling, self).__init__()
        self.channels = channels
        self.maxpool = nn.MaxPool3d(2)
        self.avgpool = nn.AvgPool3d(2)
        self.conv1 = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(channels // 2, channels, kernel_size=(1, 1, 1))

    def forward(self, x, y, z):
        # x,y has dimensions (b, c, h, w, d)
        b, c, h, w, d = x.size()
        x_mp = self.maxpool(x)
        y_mp = self.maxpool(y)
        z_mp = self.maxpool(z)

        x_ap = self.avgpool(x)
        y_ap = self.avgpool(y)
        z_ap = self.avgpool(z)

        x = x_mp + x_ap
        y = y_mp + y_ap
        z = z_mp + z_ap

        x = x.view(b, c, -1)
        y = y.view(b, c, -1)
        z = z.view(b, c, -1)

        xy = x + y
        xz = x + z
        yz = y + z
        # compute outer product of x, y
        outer_xy = torch.einsum('bci,bcj->bcij', xy, xy)
        outer_xz = torch.einsum('bci,bcj->bcij', xz, xz)
        outer_yz = torch.einsum('bci,bcj->bcij', yz, yz)

        outer = outer_xy + outer_yz + outer_xz
        outer = outer.contiguous().view(b, c, -1)

        # outer = outer.view(b, c, h*h, w*w, d*d)
        # pooled = self.conv2(torch.relu(self.conv1(outer)))
        pooled = outer.sum(dim=2)
        pooled = F.normalize(pooled, dim=-1)

        return pooled
    
class LocalAwareLearning(nn.Module):
    def __init__(self, in_chans1, in_chans2, in_chans3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chans1, in_chans2, 3, 2, 1)
        self.conv2 = nn.Conv3d(in_chans2, in_chans3, 3, 2, 1)
        self.conv3 = nn.Conv3d(in_chans3, in_chans3*2, 3, 2, 1)
        self.s1 = nn.Sequential(
            nn.Conv3d(in_chans1, in_chans1, 1),
            nn.Sigmoid(),
        )
        self.s2 = nn.Sequential(
            nn.Conv3d(in_chans2 * 2, in_chans2, 1),
            nn.Sigmoid(),
        )
        self.s3 = nn.Sequential(
            nn.Conv3d(in_chans3 * 2, in_chans3, 1),
            nn.Sigmoid(),
        )

    def forward(self, s1, s2, s3):
        s1_0 = self.s1(s1)
        s1 = s1 * s1_0
        s1 = self.conv1(s1)

        s2_0 = torch.cat((s2, s1), 1)
        s2_1 = self.s2(s2_0)
        s2 = s2 * s2_1
        s2 = self.conv2(s2)

        s3_0 = torch.cat((s3, s2), 1)
        s3_1 = self.s3(s3_0)
        s3 = s3 * s3_1
        s3 = self.conv3(s3)
        out = s3

        return out


def get_inplanes():
    return [16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


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
    expansion = 2

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


class MDL_Net(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 iter=3,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=2,
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
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
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

        self.global_fusion = SelfAdaptiveTransformer(hidden_size=block_inplanes[3], img_size=(128, 128, 128),
                                                       patch_size=(32, 32, 32), num_heads=4, attention_dropout_rate=0.2,
                                                       window_size=(2, 2, 2))

        self.local_fusion = LocalAwareLearning(in_chans1=block_inplanes[0] * block.expansion,
                           in_chans2=block_inplanes[1] * block.expansion,
                           in_chans3=block_inplanes[2] * block.expansion)
        
        self.fbc = FactorizedBilinearPooling(block_inplanes[3] * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.MaxPool3d(2)
        
        self.fc_roi = nn.Linear(block_inplanes[3] * block.expansion * 3, 90)
        self.fc_cls2roi = nn.Linear(n_classes, 90)
        self.roi = Disease_Guide_ROI(90, 6, i=iter)

        self.dropout = nn.Dropout(0.5)

        self.roi_fc = nn.Linear(90, n_classes)

        self.pred_fc = nn.Linear(block_inplanes[3] * block.expansion * 3, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

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
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        # return out
        return s1, s2, s3, s4

    def forward(self, x):
        # Upsample and Split Multimodal Feature
        # x = F.interpolate(x, [128, 128, 128], mode='trilinear')
        x1 = x[:, 0, :, :, :]
        x1 = torch.unsqueeze(x1, dim=1)    # The unsqueeze operation is to add channel of single-modal dataset
        x2 = x[:, 1, :, :, :]
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = x[:, 2, :, :, :]
        x3 = torch.unsqueeze(x3, dim=1)
        # x1 = x2 = x3 = x
        # Extract single modal feature
        x1_s1, x1_s2, x1_s3, x1_s4 = self.feature_forward(x1)
        x2_s1, x2_s2, x2_s3, x2_s4 = self.feature_forward(x2)
        x3_s1, x3_s2, x3_s3, x3_s4 = self.feature_forward(x3)

        # Global-aware Learning
        # 这个global fusion是融合了多模态数据，原文有3个分支，最后会有3个128，4， 4， 4 ----> 64, 128
        x1_s4 = self.dropout(x1_s4)
        x2_s4 = self.dropout(x2_s4)
        x3_s4 = self.dropout(x3_s4)
        b, c, h, w, d = x1_s4.size()
        fusion = self.global_fusion(x1_s4, x2_s4, x3_s4)    # 64, 128
        fusion = rearrange(fusion, 'b (h w d) c->b c h w d', h=h, w=w, d=d) # 128, 4, 4, 4
        fusion = self.avgpool(fusion)   # 128, 1, 1, 1
        out_global = fusion.view(fusion.shape[0], -1)   # ,128 特征

        # Latent-space aware Learning
        # 感觉上类似GAL ，也是3个 128，4，4，4 合并操作输出   ----> 128
        out_latent = self.fbc(x1_s4, x2_s4, x3_s4)  # ,128
        out = torch.cat((out_global, out_latent), dim=1)    # GAL+LAL 都是用最后一个stage来获得（，128）的输出。cat后变 (, 256)

        # Local-aware Learning
        # 总的来说是先做模态性融合，最后阶段性融合
        # out是单个模态不同resnet的阶段性特征 这个可以算作模态性融合
        out_s1 = x1_s1 + x2_s1 + x3_s1
        out_s2 = x1_s2 + x2_s2 + x3_s2
        out_s3 = x1_s3 + x2_s3 + x3_s3
        # out_s4 = x1_s4 + x2_s4 + x3_s4
        # local fusion是 阶段性融合
        fusion_l = self.local_fusion(out_s1, out_s2, out_s3)    # 3个不同阶段不同size的特征图，通过local_fusion后变成resnet最终输出的 128, 4, 4, 4
        fusion_l = self.avgpool(fusion_l)   # 128, 1, 1, 1
        fusion_l = fusion_l.view(fusion_l.shape[0], -1) # ,128
        out = torch.cat((out, fusion_l), dim=1) # ,384
        class_out = self.pred_fc(out)

        # Diseased-incuded ROI Learning
        # 输出真正的分类 和 ROI区域关注点，结合的是特征图和类别映射
        f_roi = self.fc_roi(out).unsqueeze(dim=1)   # ,384 ---> 1,90 把上面的特征图变为90个ROI的权重
        cls_roi = self.fc_cls2roi(class_out).unsqueeze(dim=1)   # 线性层从类别反映射回ROI权重 1,90
        roi = self.roi(f_roi, cls_roi)  # 结合上面两个1,90 回推 ,90 各个roi的权重
        roi_cls = self.roi_fc(roi) # 新权重再反映射回类别
        
        # Classification 
        class_out = class_out + roi_cls
        return class_out
        # return class_out, roi


def generate_model(model_depth, in_planes, num_classes, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = MDL_Net(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 18:
        model = MDL_Net(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 34:
        model = MDL_Net(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 50:
        model = MDL_Net(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 101:
        model = MDL_Net(Bottleneck, [3, 4, 23, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 152:
        model = MDL_Net(Bottleneck, [3, 8, 36, 3], get_inplanes(), n_input_channels=in_planes,
                      n_classes=num_classes,
                      **kwargs)
    elif model_depth == 200:
        model = MDL_Net(Bottleneck, [3, 24, 36, 3], get_inplanes(), n_input_channels=in_planes, n_classes=num_classes,
                      **kwargs)

    return model


if __name__ == '__main__':
    # print('test self adaptive attention:\n')
    # x = torch.randn(2, 4096, 512)
    # x1 = torch.randn(1, 128, 4, 4, 4)
    # # pos = PositionEmbedding(hidden_size=128, img_size=[128, 128, 128], patch_size=[8, 8, 8])
    # # mda = SelfAdaptiveAttention(hidden_size=512, img_size=(128, 128, 128), patch_size=(8, 8, 8), num_heads=4,
    # #                            attention_dropout_rate=0.2, window_size=(4, 4, 4))
    # mdt = SelfAdaptiveTransformer(hidden_size=128, img_size=(128, 128, 128), patch_size=(32, 32, 32), num_heads=4,
    #                              attention_dropout_rate=0.2, window_size=(4, 4, 4))
    # # out = pos(x)
    # out = mdt(x1, x1, x1)
    # print(out.shape)
    # print("SAA end\n")
    '''
    The shpae of x: [b, m, h, w, d]
    m is the number of modality, i.e, if have three multimodal dataset, m is 3
    '''
    # x = torch.ones(3, 3, 128, 128, 128)
    # y = torch.randn(1, 90)
    # time_start = time.time()
    # model = generate_model(model_depth=18, in_planes=1, num_classes=3)

    # output, roi = model(x)
    # time_over = time.time()
    # print(output.shape)
    # print(roi.shape)
    # flops, params = profile(model, (x,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # print('time:{}s'.format(time_over - time_start))

    # 简易测试 
    x = torch.randn(3,3,128,128,128)
    model = generate_model(model_depth=18,in_planes=1,num_classes=2)
    out = model(x)
    print(out)
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch import einsum

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size[0]) for y in range(window_size[1])]))
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
                    img_size[1] // patch_size[1]), hidden_size))
        elif types == 1:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[1] // patch_size[1]), hidden_size))

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
            self.pos_embedding = nn.Parameter(torch.randn(max_indice + 1, max_indice + 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1], window_size[0] * window_size[1]))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], nw_h=nw_h, nw_w=nw_w)
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
        B, C, H, W = x.shape
        w = x + y + z
        x1 = rearrange(x, "b c h w -> b (w h) c")
        y1 = rearrange(y, "b c h w -> b (w h) c")
        z1 = rearrange(z, "b c h w -> b (w h) c")
        w = w.permute(0, 2, 3, 1)

        if self.is_position:
            x1 = self.pos_embedding1(x1)
            y1 = self.pos_embedding2(y1)
            z1 = self.pos_embedding2(z1)

        x1 = self.attention_x(x1)
        y1 = self.attention_y(y1)
        z1 = self.attention_z(z1)
        w = self.window_attention(w)

        w = rearrange(w, "b h w c -> b (h w) c", h=H, w=W)
        # x3 = rearrange(x3, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)
        # x2 = rearrange(x2, "(b h) (d w) c -> b (h w d) c", d=D, h=H, w=W)
        # x1 = rearrange(x1, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)

        h = self.norm(x1 * y1) * w
        h = self.norm(h)
        out = (h * z1) * w

        return out


class SelfAdaptiveTransformer(nn.Module):
    def __init__(self, hidden_size=128, img_size=(180,180), patch_size=(8, 8, 8), num_heads=4,
                 attention_dropout_rate=0.2, window_size=(4, 4)):
        super().__init__()
        self.mda = SelfAdaptiveAttention(hidden_size=hidden_size, img_size=img_size, patch_size=patch_size,
                                        num_heads=num_heads,
                                        attention_dropout_rate=attention_dropout_rate, window_size=window_size,
                                        pos_embedding=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, hidden_size * 2, dropout_rate=0.2)
        self.fc = nn.Linear(hidden_size, 128)
    def forward(self, x, y, z):
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        y = rearrange(y, "b c h w -> b (h w) c")
        y = self.norm(y)
        z = rearrange(z, "b c h w -> b (h w) c")
        z = self.norm(z)
        h = x + y + z
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        y = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)
        z = rearrange(z, "b (h w) c -> b c h w", h=H, w=W)
        x1 = self.mda(x, y, z)
        x2 = h + x1
        out = self.norm(x2)
        out = self.mlp(out)
        out = x2 + out
        out = out.mean(dim=1)
        out = self.fc(out)
        return out
class SingleViewMultiScaleFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
        [nn.Conv2d(c, channel[2], kernel_size=3, stride=1, padding=1) for c in channel]
    )
        self.final_conv = nn.Conv2d(channel[2], channel[2] * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        ans = torch.ones_like(x[2])
        target_size = x[2].shape[-2:]
        for i, x in enumerate(x):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, (target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)
        h, w = ans.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            ans = F.pad(ans, (0, 1, 0, 1))  # 在右侧和底部填充 1 行/列

        ans = self.final_conv(ans)
        return ans

class CrossFusion(nn.Module):
    def __init__(self, initial_channels):
        super().__init__()
        self.initial_channels = initial_channels

        # 定义融合模块
        self.fusion1 = self._make_fusion_layer(num_inputs=3,channels=initial_channels)
        self.fusion2 = self._make_fusion_layer(num_inputs=4, channels=initial_channels * 2)
        self.fusion3 = self._make_fusion_layer(num_inputs=4, channels=initial_channels * 4)

        # 定义下采样模块
        self.downsample1 = self._make_downsample_layer(initial_channels, initial_channels * 2)
        self.downsample2 = self._make_downsample_layer(initial_channels * 2, initial_channels * 4)
        self.downsample3 = self._make_downsample_layer(initial_channels * 4, initial_channels * 8)

    def _make_fusion_layer(self, num_inputs, channels):
        """创建一个融合模块，将多个输入特征融合成一个特征"""
        return nn.Sequential(
            nn.Conv2d(channels * num_inputs, channels, kernel_size=1),  # 融合通道数
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _make_downsample_layer(self, in_channels, out_channels):
        """创建一个下采样模块，包括卷积和池化"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, y1, y2, y3, z1, z2, z3):
        # 第一步：融合 x1, y1, z1
        h = self.fusion1(torch.cat([x1, y1, z1], dim=1))  # 融合
        h = self.downsample1(h)  # 下采样，通道翻倍

        # 第二步：融合 h, x2, y2, z2
        h = self.fusion2(torch.cat([h, x2, y2, z2], dim=1))  # 融合
        h = self.downsample2(h)  # 下采样，通道翻倍

        # 第三步：融合 h, x3, y3, z3
        h = self.fusion3(torch.cat([h, x3, y3, z3], dim=1))  # 融合
        h = self.downsample3(h)  # 下采样，通道翻倍

        return h
class ModifiedResNet(torch.nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        out1 = x
        x = self.resnet.layer2(x)
        out2 = x
        x = self.resnet.layer3(x)
        out3 = x
        x = self.resnet.layer4(x)
        out4 = x

        return out1, out2, out3, out4


class TriViewResNet18(nn.Module):
    def __init__(self, num_classes):
        super(TriViewResNet18, self).__init__()

        # 加载预训练的 ResNet18 模型
        self.resnet18_x = ModifiedResNet()
        self.resnet18_y = ModifiedResNet()
        self.resnet18_z = ModifiedResNet()


        self.fusion_multisize = SingleViewMultiScaleFusion([64,128,256])
        self.down_fusion = CrossFusion(initial_channels=64)
        self.late_fusion = LateMultiViewFusion(channels=512)
        self.sat = SelfAdaptiveTransformer(hidden_size=512,img_size=(180,180),patch_size=(30,30),num_heads=4,
                                         attention_dropout_rate=0.2,window_size=(2,2))
        # self.up_fusion = SAT(x4,y4,z4)                    self adaptive transformer 结合三个 512,6,6,6的数据，我觉得可以变成 216，512的Voxel推断
    def forward(self, x):
        # 验证输入数据形状是否正确
        assert x.dim() == 5, "输入数据应为 (batch_size, C, H, W, D)"

        # 提取三个切面
        x_slice = x[:, :, :, :, 0]
        y_slice = x[:, :, :, :, 1]
        z_slice = x[:, :, :, :, 2]

        # 通过每个 ResNet18 提取特征
        x1,x2,x3,x4 = self.resnet18_x(x_slice)
        y1,y2,y3,y4 = self.resnet18_y(y_slice)
        z1,z2,z3,z4 = self.resnet18_z(z_slice)

        # 这个是中期多视角融合
        down_out = self.down_fusion(x1,x2,x3,y1,y2,y3,z1,z2,z3)

        # 这个是单视角多尺度融合数据
        x_fusion = self.fusion_multisize([x1,x2,x3])
        y_fusion = self.fusion_multisize([y1,y2,y3])
        z_fusion = self.fusion_multisize([z1,z2,z3])

        down_out = self.late_fusion(x_fusion,y_fusion,z_fusion, down_out)

        # 这个是特征提取后多视角融合
        up_out = self.sat(x4,y4,z4)

        out = torch.concat([up_out,down_out], dim=1)

        return out


class LateMultiViewFusion(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.channels = channels

        # 全局平均池化（2D）
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 第一阶段融合：融合 x_fusion, y_fusion, z_fusion
        self.fusion_xyz = nn.Sequential(
            nn.Linear(channels * 3, channels),  # 输入是 x_pool, y_pool, z_pool 的拼接
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),  # 输出 h 的特征
            nn.Softmax(dim=1)  # 对权重进行归一化
        )

        # 第二阶段融合：融合 h 和 down_out
        self.fusion_h_down = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),  # 输入是 h 和 down_pool 的拼接
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, 2),  # 输出 2 个权重（对应 h 和 down_out）
            nn.Softmax(dim=1)  # 对权重进行归一化
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 128)

    def forward(self, x_fusion, y_fusion, z_fusion, down_out):
        # 池化和展平
        x_pool = self.gap(x_fusion).flatten(1)  # (batch, 512)
        y_pool = self.gap(y_fusion).flatten(1)  # (batch, 512)
        z_pool = self.gap(z_fusion).flatten(1)  # (batch, 512)
        down_pool = self.gap(down_out).flatten(1)  # (batch, 512)

        # 第一阶段融合：融合 x_fusion, y_fusion, z_fusion
        xyz_fused = torch.cat([x_pool, y_pool, z_pool], dim=1)  # (batch, 512 * 3)
        h_weights = self.fusion_xyz(xyz_fused)  # (batch, 512)
        h_weights = h_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 512, 1, 1)

        # 加权融合 x_fusion, y_fusion, z_fusion
        h = (x_fusion * h_weights[:, 0:1] +
             y_fusion * h_weights[:, 1:2] +
             z_fusion * h_weights[:, 2:3])  # (batch, 512, 6, 6)

        # 第二阶段融合：融合 h 和 down_out
        h_pool = self.gap(h).flatten(1)  # (batch, 512)
        h_down_fused = torch.cat([h_pool, down_pool], dim=1)  # (batch, 512 * 2)
        final_weights = self.fusion_h_down(h_down_fused)  # (batch, 2)
        final_weights = final_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)

        # 加权融合 h 和 down_out
        fused_feature = (h * final_weights[:, 0] +
                         down_out * final_weights[:, 1])  # (batch, 512, 6, 6)
        fused_feature = self.pool(fused_feature)
        fused_feature = fused_feature.squeeze(-1).squeeze(-1)
        fused_feature = self.fc(fused_feature)
        return fused_feature

if __name__ == '__main__':
    model = TriViewResNet18(num_classes=2)
    x = torch.randn(1, 3, 180, 180, 3)  # 示例输入
    y = model(x)
    print("分类结果:", y.shape)



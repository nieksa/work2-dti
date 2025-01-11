# input shape 3, 180, 180, 180
# 结合regionTransformer的思想划分 4*4*4 的 Local Patch 和 20*20*20 的 Region Patch
from einops import rearrange
import torch
import torch.nn as nn


class ROIEmbed(nn.Module):
    def __init__(self, roi_dim=128):
        super(ROIEmbed, self).__init__()
        self.roi_dim = roi_dim

        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.AdaptiveAvgPool3d(output_size=(9, 9, 9))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),  # 输入维度为 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=self.roi_dim),
        )

    def forward(self, x):
        """
        输入数据的形状是 (batch_size, 3, 180, 180, 180)
        中间数据是 (batch_size, roi_dim, 9, 9, 9)
        输出数据是 (batch_size, 729, roi_dim)
        """
        # 通过卷积层和池化层
        x = self.conv_layers(x) # (batch_size, roi_dim, 9, 9, 9)
        # 展平除了batch维度以外的所有维度
        x = x.view(x.size(0), x.size(1), -1)  # 变成 (batch_size, roi_dim, 729)
        # 转置维度使得输出形状为 (batch_size, 729, roi_dim)
        x = x.permute(0, 2, 1)  # (batch_size, 729, roi_dim)
        x = self.fc1(x)

        return x


class VoxelEmbed(nn.Module):
    def __init__(self, voxel_dim=256, patch_size=20):
        super(VoxelEmbed, self).__init__()
        self.voxel_dim = voxel_dim
        self.patch_size = patch_size

        # 定义卷积层，用于降采样并增加通道数
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=self.voxel_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(self.voxel_dim),
            nn.ReLU(),
            nn.Dropout3d(0.1)
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x):
        """
        输入数据的形状是 (batch_size, 3, 180, 180, 180)
        输出的数据形状是 (batch_size, 729, 125, voxel_dim)
        """
        x = rearrange(x, 'b c (h p1) (w p2) (t p3) -> (b h w t) c p1 p2 p3', p1=self.patch_size, p2=self.patch_size,
                      p3=self.patch_size)


        x = self.conv1(x)  # (batch * 729, 64, 20, 20, 20)
        x = self.conv2(x)  # (batch * 729, voxel_dim, 10, 10, 10)
        x = self.pool(x)
        # Step 4: 展平空间维度，使得形状变成 (batch * 729, voxel_dim, 125)
        x = x.view(x.size(0), x.size(1), -1)  # (batch * 729, voxel_dim, 125)

        # Step 5: 将数据重组回原来的形状 (batch_size, 729, 125, voxel_dim)
        x = rearrange(x, '(b n) d p -> b n p d', b=x.size(0) // 729)

        return x


class ROI_Voxel_Attention(nn.Module):
    def __init__(self, roi_dim, voxel_dim, num_heads=8):
        super(ROI_Voxel_Attention, self).__init__()

        self.roi_dim = roi_dim
        self.voxel_dim = voxel_dim

        # 用于更新 roi token 的注意力块
        self.roi_attention = nn.MultiheadAttention(embed_dim=roi_dim, num_heads=num_heads)
        self.fc_voxel_to_roi = nn.Linear(voxel_dim, roi_dim)

        # 用于更新voxel token的注意力模块
        self.voxel_attention = nn.MultiheadAttention(embed_dim=voxel_dim, num_heads=num_heads)
        self.fc_roi_to_voxel = nn.Linear(roi_dim, voxel_dim)

    def forward(self, roi_tokens, voxel_tokens):
        """
        roi_tokens: (batch_size, 729, roi_dim)
        voxel_tokens: (batch_size, 729, 125, voxel_dim)
        """
        batch_size = roi_tokens.size(0)
        roi_num = roi_tokens.size(1)
        voxel_nums_in_roi = voxel_tokens.size(2)

        # Update roi information
        voxel_for_roi = self.fc_voxel_to_roi(voxel_tokens)
        voxel_for_roi = voxel_for_roi.view(batch_size * roi_num, voxel_nums_in_roi,
                                                 -1)  # (batch * 729, 125, roi_dim)
        voxel_for_roi = voxel_for_roi.permute(1, 0, 2)
        # roi to Attention
        roi_tokens_reshaped = roi_tokens.view(batch_size * roi_num, -1).unsqueeze(0)
        updated_roi_tokens, _ = self.roi_attention(roi_tokens_reshaped, voxel_for_roi, voxel_for_roi)
        updated_roi_tokens = updated_roi_tokens.squeeze(0).view(batch_size, roi_num, -1)

        # Update voxel information
        roi_for_voxel = self.fc_roi_to_voxel(updated_roi_tokens)
        roi_for_voxel = roi_for_voxel.unsqueeze(2).expand(-1, -1, voxel_nums_in_roi,
                                                                -1)  # (batch_size, 729, 125, voxel_dim)
        roi_for_voxel = roi_for_voxel.view(batch_size * roi_num, voxel_nums_in_roi,
                                                 -1)  # (batch * 729, 125, voxel_dim)
        roi_for_voxel = roi_for_voxel.permute(1, 0, 2)  # (125, batch * 729, voxel_dim)

        # 将voxel_tokens调整为注意力输入格式
        voxel_tokens_reshaped = voxel_tokens.view(batch_size * roi_num, voxel_nums_in_roi, -1).permute(1, 0,
                                                                                                             2)

        updated_voxel_tokens, _ = self.voxel_attention(voxel_tokens_reshaped, roi_for_voxel,
                                                       roi_for_voxel)  # (125, batch * 729, voxel_dim)
        updated_voxel_tokens = updated_voxel_tokens.permute(1, 0, 2).view(batch_size, roi_num, voxel_nums_in_roi,
                                                                          -1)  # (batch_size, 729, 125, voxel_dim)

        return updated_roi_tokens, updated_voxel_tokens

class ROI_Voxel_fusion_block(nn.Module):
    def __init__(self, target_dim):
        super(ROI_Voxel_fusion_block, self).__init__()
        pass
    def forward(self, roi_tokens, voxel_tokens):
        """
        roi_tokens: (batch_size, 729, roi_dim)
        voxel_tokens: (batch_size, 729, 125, voxel_dim)
        """
        # 我希望最终fusion成(batch_size, target_dim)
        pass

class ROIVisionTransformer(nn.Module):
    def __init__(self, roi_feature_dim, voxel_feature_dim, num_layers, num_heads=8):
        super(ROIVisionTransformer, self).__init__()
        self.num_layers = num_layers

        self.roi_embed = ROIEmbed(roi_dim=roi_feature_dim)
        self.voxel_embed = VoxelEmbed(voxel_dim=voxel_feature_dim)

        self.layers = nn.ModuleList([
            ROI_Voxel_Attention(roi_dim=roi_feature_dim, voxel_dim=voxel_feature_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.mil_pooling = nn.Sequential(
            nn.Linear(roi_feature_dim, 1),
            nn.Sigmoid()
        )

        # Classification Head
        self.classifier = nn.Linear(729, 2)


    def forward(self, x):

        roi_tokens = self.roi_embed(x)
        voxel_tokens = self.voxel_embed(x)  # (batch_size, 729, 125, voxel_dim)

        for layer in self.layers:
            roi_tokens, voxel_tokens = layer(roi_tokens, voxel_tokens)

        mil_scores = self.mil_pooling(roi_tokens).squeeze(-1)
        logits = self.classifier(mil_scores)

        return logits

if __name__ == '__main__':
    model = ROIVisionTransformer(roi_feature_dim=128, voxel_feature_dim=128, num_layers=1, num_heads=1)
    # model = ROIEmbed(roi_dim=128)
    x = torch.randn((1, 3, 180, 180, 180))
    logits = model(x)
    print(logits.shape)
    # print(voxel_tokens.shape)
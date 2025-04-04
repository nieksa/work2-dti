import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, input_shape=(91,109,91)):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, 128)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feature_map = x
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x, feature_map

# 定义一个分类器，把128维的embedding分类成2类
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        return self.fc(x)
    
# 定义一个融合模块，把fa_embedding和mri_embedding融合起来然后再二分类的模块
class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        
    def forward(self, fa_emb, mri_emb):
        # 计算注意力权重
        fa_weight = self.attention(fa_emb)
        mri_weight = self.attention(mri_emb)
        
        # 归一化权重
        total = fa_weight + mri_weight
        fa_weight = fa_weight / total
        mri_weight = mri_weight / total
        
        # 加权融合
        fused = fa_weight * fa_emb + mri_weight * mri_emb
        
        # 分类
        out = self.classifier(fused)
        return out
    
class Simple3DViT(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, num_heads=4):
        super(Simple3DViT, self).__init__()
        self.proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)  # 线性投影
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=4
        )
        self.output_layer = nn.Conv3d(hidden_dim, 1, kernel_size=1)  # 降维到 1 维通道
    
    def forward(self, x):
        B, C, D, H, W = x.shape  # (batch, 128, 12, 24, 12)
        x = self.proj(x)  # 变换通道数 (B, 128, 12, 24, 12) -> (B, 128, 12, 24, 12)
        x = x.flatten(2).permute(2, 0, 1)  # 变换为 Transformer 输入 (D*H*W, B, C)
        x = self.transformer(x)  # Transformer 处理
        x = x.permute(1, 2, 0).reshape(B, 128, D, H, W)  # 变回 3D 结构
        x = self.output_layer(x)  # 变换通道数到 1 (B, 1, 12, 24, 12)
        return x
    
class feature_map_to_heatmap(nn.Module):
    def __init__(self):
        super(feature_map_to_heatmap, self).__init__()
        # self.fa_heatmap = 一个接受128,23,28,23的vit模型，然后输出1,1,23,28,23的heatmap
        # self.mri_heatmap = 一个接受128,12,14,12的vit模型，然后输出1,1,12,14,12的heatmap
        # 下采样模块，把128，23，28，23的fa_feature_map下采样到128，12，14，12
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        )
        self.fa_heatmap = Simple3DViT(in_channels=128, hidden_dim=128, num_heads=4)
        self.mri_heatmap = Simple3DViT(in_channels=128, hidden_dim=128, num_heads=4)
    def forward(self, fa, mri):
        fa = self.downsample(fa)
        fa_heatmap = self.fa_heatmap(fa)
        mri_heatmap = self.mri_heatmap(mri)
        return fa_heatmap, mri_heatmap
        

class FA_MRI_contrastive_model(nn.Module):
    def __init__(self, **kwargs):
        super(FA_MRI_contrastive_model, self).__init__()
        self.model_fa = ResNet(BasicBlock, [2, 2, 2, 2], input_shape=(182,218,182))
        self.model_mri = ResNet(BasicBlock, [2, 2, 2, 2], input_shape=(91,109,91))
        self.fa_classifier = Classifier()
        self.mri_classifier = Classifier()
        self.fusion_block = FusionModule()
        self.feature_map_to_heatmap = feature_map_to_heatmap()
        # 添加FA数据的下采样层
        self.fa_downsample = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
        
    def forward(self, fa, mri):
        fa_embedding, fa_feature_map = self.model_fa(fa)
        mri_embedding, mri_feature_map = self.model_mri(mri)
        fa_logit = self.fa_classifier(fa_embedding)
        mri_logit = self.mri_classifier(mri_embedding)
        fusion_logit = self.fusion_block(fa_embedding, mri_embedding)
        fa_heatmap, mri_heatmap = self.feature_map_to_heatmap(fa_feature_map, mri_feature_map)
        return fa_logit,fa_heatmap,fa_embedding,mri_logit,mri_heatmap,mri_embedding,fusion_logit

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fa = torch.randn(1, 1, 182, 218, 182).to(device)  # FA数据尺寸改为182x218x182
    mri = torch.randn(1, 1, 91, 109, 91).to(device)
    model = FA_MRI_contrastive_model().to(device)
    fa_logit,fa_feature_map,fa_embedding,mri_logit,mri_feature_map,mri_embedding,fusion_logit = model(fa, mri)
    print(fa_embedding.shape)
    print(mri_embedding.shape)


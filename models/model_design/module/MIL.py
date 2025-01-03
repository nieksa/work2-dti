import torch.nn as nn
import torch

class InstanceScore(nn.Module):
    def __init__(self, channels):
        super(InstanceScore, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),  # 降维
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, 1),  # 输出[batch_size, 1, num_instances]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入 x: [batch_size, channels, num_instances]
        attn_weights = self.attention(x)  # 输出 [batch_size, 1, num_instances]
        attn_weights = attn_weights.squeeze(1)  # [batch_size, num_instances]
        return self.softmax(attn_weights)

class SelfAttentionPool(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, 1, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: [batch_size, channels, num_instances]
        attn_weights = self.attention(x)  # [batch_size, 1, num_instances]
        weighted_sum = torch.sum(x * attn_weights, dim=-1, keepdim=True)
        return weighted_sum
class MIL(nn.Module):
    def __init__(self, channels, num_instances):
        super(MIL, self).__init__()
        self.channels = channels
        self.num_instances = num_instances

        # 最大池化、平均池化和自注意力池化

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.selfadaptivepool = SelfAttentionPool(num_instances)

        self.instance_score = InstanceScore(3)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_instances, 128)
    def forward(self, x):
        # x的形状为 [batch_size, channels, x,y,z]
        batch_size, channels, height,width,depth = x.shape
        num_instances = height*width*depth
        x = x.view(batch_size, channels, num_instances)

        x = x.permute(0, 2, 1)
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        self_pool = self.selfadaptivepool(x)

        # 这一步是在channel维度聚合池化特征
        x = torch.concat([max_pool, avg_pool, self_pool], dim=-1)
        x = x.permute(0, 2, 1)
        # 下一步是计算各个实例的权重
        instance_score = self.instance_score(x)
        x = self.fc(instance_score)
        return x

if __name__ == '__main__':
     x = torch.randn(1, 128, 12, 14,12)
     model = MIL(128, 12*14*12)
     y = model(x)
     print(y.shape)
     print("Instance Scores Distribution (Softmax probabilities):")
     print(f"Min: {y.min()}")
     print(f"Max: {y.max()}")
     print(f"Mean: {y.mean()}")
     print(f"Std: {y.std()}")

     x = torch.randn(1, 256, 6, 7, 6)
     model = MIL(256, 6 * 7 * 6)
     y = model(x)
     print(y.shape)
     print("Instance Scores Distribution (Softmax probabilities):")
     print(f"Min: {y.min()}")
     print(f"Max: {y.max()}")
     print(f"Mean: {y.mean()}")
     print(f"Std: {y.std()}")

     # 打印所有实例得分的和
     print("Sum of instance scores:", y.sum())
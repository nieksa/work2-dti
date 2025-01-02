import torch
import torch.nn as nn
"""

有提升！！！！！！！！！

轻量级的注意力机制，旨在通过捕获输入特征图的均值信息来计算每个通道的注意力权重。
SimAM的核心思想是基于输入特征与其均值之差的平方来计算注意力，并进行平滑处理，从而得到一个简单有效的通道注意力机制。
适合添加有公式推导
"""
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

if __name__ == "__main__":
    input_tensor = torch.randn(4, 16, 64, 64, 64) 
    simam3d = Simam3DModule(e_lambda=1e-4)
    output_tensor = simam3d(input_tensor)
    print(output_tensor.shape)

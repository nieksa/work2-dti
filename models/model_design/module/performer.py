from performer_pytorch import Performer
import torch
# 定义 Performer 模型
model = Performer(
    dim = 64,          # 特征维度
    depth = 1,          # 层数
    heads = 1,          # 注意力头数
    dim_head = 64,      # 每个头的维度
    causal = False      # 非自回归
)

# 输入数据 (batch_size, seq_len, dim)
x = torch.randn(1, 91125, 64)  # 假设已经将数据投影到 512 维

# 前向传播
output = model(x)  # (64, 91125, 512)
print(output.shape)
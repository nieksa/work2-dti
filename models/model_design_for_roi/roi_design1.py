import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=90):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=18, d_model=64, num_heads=4, num_layers=4, dim_feedforward=256, dropout=0.1, max_len=90):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(d_model * max_len, 2)  # 二分类任务，输出 2 个 logits

    def forward(self, x):
        # 输入形状: (batch, seq_len=90, input_dim=18)
        batch_size, seq_len, _ = x.size()
        x = self.input_embedding(x)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        x = x.reshape(batch_size, -1)  # (batch, seq_len * d_model)
        x = self.fc(x)  # (batch, 2)

        return x


# 测试模型
if __name__ == "__main__":
    model = TransformerClassifier(input_dim=18, d_model=64, num_heads=4, num_layers=4, dim_feedforward=256, dropout=0.1)
    batch_size = 7
    seq_len = 90
    input_dim = 18
    x = torch.randn(batch_size, seq_len, input_dim)  # 输入形状: (batch, seq_len, input_dim)
    output = model(x)
    print(output.shape)
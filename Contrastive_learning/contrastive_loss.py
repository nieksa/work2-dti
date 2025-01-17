import torch
import torch.nn.functional as F

def nt_xent_loss(fa_proj, l1_proj, md_proj, temperature=0.5, eps=1e-8):
    """
    多模态对比学习的 NT-Xent 损失
    参数:
        fa_proj: FA 模态投影向量 (batch_size, embedding_dim)
        l1_proj: L1 模态投影向量 (batch_size, embedding_dim)
        md_proj: MD 模态投影向量 (batch_size, embedding_dim)
        temperature: 温度参数
        eps: 防止数值问题的微小值
    返回:
        loss: 多模态对比学习损失
    """
    # 拼接所有模态的投影向量
    all_proj = torch.cat([fa_proj, l1_proj, md_proj], dim=0)  # (3 * batch_size, embedding_dim)

    # 归一化
    all_proj = F.normalize(all_proj, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(all_proj, all_proj.T)  # (3 * batch_size, 3 * batch_size)
    similarity_matrix /= temperature

    # 限制相似度矩阵的范围，避免值为零
    similarity_matrix = torch.clamp(similarity_matrix, min=eps)

    # 构造正样本掩码
    batch_size = fa_proj.shape[0]
    total_size = 3 * batch_size
    mask = torch.eye(total_size, device=fa_proj.device, dtype=torch.bool)  # 自身对角线为 True
    mask = mask.logical_not()  # 排除自身相似度

    # 使用 softmax 归一化相似度
    exp_similarity = torch.exp(similarity_matrix)  # 按元素取指数
    exp_similarity = exp_similarity * mask  # 只保留非对角线部分
    denom = exp_similarity.sum(dim=1, keepdim=True) + eps  # 防止分母为零

    # 正样本对的相似度（索引分布在相邻模态中）
    positive_indices = (
        torch.arange(batch_size, device=fa_proj.device).repeat(3) +
        torch.tensor([batch_size, 2 * batch_size, 0], device=fa_proj.device).repeat_interleave(batch_size)
    )

    positive_similarity = similarity_matrix[
        torch.arange(total_size, device=fa_proj.device), positive_indices
    ] + eps  # 确保正样本相似度大于零

    # 计算 NT-Xent 损失
    loss = -torch.log(positive_similarity / denom.squeeze())
    loss = loss.mean()  # 对 batch 平均

    return loss


if __name__ == '__main__':
    # 示例输入张量
    fa = torch.randn(16, 128)
    l1 = torch.randn(16, 128)
    md = torch.randn(16, 128)

    # 计算损失
    loss = nt_xent_loss(fa, l1, md)
    print(f"Loss: {loss.item():.4f}")

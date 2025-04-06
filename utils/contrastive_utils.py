import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIM3D(nn.Module):
    def __init__(self, window_size=5, channels=1, sigma=1.5):
        super().__init__()
        # 初始化3D高斯核
        if not isinstance(window_size, int):
            raise ValueError("window_size must be an integer")

        self.window = self._gaussian_kernel3d(window_size, sigma, channels)

        # 动态注意力网络
        self.attention_net = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 对DTI_FA数据下采样与MRI数据匹配
        # self.downsample = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

    def _gaussian_kernel3d(self, size, sigma, channels):
        coords = torch.arange(size).float() - (size - 1) / 2.0
        grid = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(grid[0]**2 + grid[1]**2 + grid[2]**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size, size).repeat(channels, 1, 1, 1, 1)

    def forward(self, fa_map, mri_map):
        # 基础SSIM计算
        C1 = (0.01 * 1.0)**2
        C2 = (0.03 * 1.0)**2

        # 下采样使用那种方式呢？
        # fa_map = self.downsample(fa_map)
        fa_map = F.interpolate(fa_map, size=(12, 14, 12), mode='trilinear', align_corners=False)

        window = self.window.to(fa_map.device)

        mu_fa = F.conv3d(fa_map, window, padding=2, groups=1)
        mu_mri = F.conv3d(mri_map, window, padding=2, groups=1)

        sigma_fa_sq = F.conv3d(fa_map * fa_map, window, padding=2, groups=1) - mu_fa**2
        sigma_mri_sq = F.conv3d(mri_map * mri_map, window, padding=2, groups=1) - mu_mri**2
        sigma_famri = F.conv3d(fa_map * mri_map, window, padding=2, groups=1) - mu_fa * mu_mri

        ssim_map = ((2 * mu_fa * mu_mri + C1) * (2 * sigma_famri + C2)) / \
                   ((mu_fa**2 + mu_mri**2 + C1) * (sigma_fa_sq + sigma_mri_sq + C2))

        # 动态注意力生成
        attention_input = torch.cat([fa_map, mri_map], dim=1)
        attention_mask = self.attention_net(attention_input)

        # 注意力加权
        weighted_ssim = ssim_map * attention_mask
        return 1 - weighted_ssim.mean()


def cross_modal_alignment_loss(fa_emb, mri_emb, tau=0.07, hard_neg=True):
    """
    改进版跨模态对齐损失，无需显式sample_ids
    :param fa_emb/mri_emb: 已归一化的特征 [B, D]
    :param tau: 温度系数，推荐0.05-0.1
    :param hard_neg: 是否启用难负样本挖掘
    """
    # 1. 特征归一化
    fa_emb = F.normalize(fa_emb, p=2, dim=1)
    mri_emb = F.normalize(mri_emb, p=2, dim=1)

    # 2. 相似度矩阵计算
    sim_matrix = fa_emb @ mri_emb.T

    # 3. 自动构建正样本掩码（对角线为True）
    pos_mask = torch.eye(fa_emb.size(0), dtype=torch.bool, device=fa_emb.device)

    # 4. 难负样本筛选
    if hard_neg:
        with torch.no_grad():
            neg_sim = sim_matrix.masked_fill(pos_mask, -np.inf)
            hard_neg_indices = neg_sim.topk(k=1, dim=1).indices
            neg_mask = torch.zeros_like(sim_matrix).scatter(1, hard_neg_indices, 1).bool()
    else:
        neg_mask = ~pos_mask

    # 5. 双向对比损失计算
    exp_sim = torch.exp(sim_matrix / tau)

    # 计算正样本相似度
    pos_term = exp_sim.diag()  # 正样本相似度

    # 计算负样本聚合
    denom_i2t = (exp_sim * neg_mask).sum(dim=1)
    denom_t2i = (exp_sim * neg_mask).sum(dim=0)

    # 添加平滑项，避免除零
    epsilon = 1e-8  # 一个小常数，避免除零
    denom_i2t = torch.max(denom_i2t, torch.tensor(epsilon, device=denom_i2t.device))
    denom_t2i = torch.max(denom_t2i, torch.tensor(epsilon, device=denom_t2i.device))
    pos_term = torch.max(pos_term, torch.tensor(epsilon, device=pos_term.device))

    # 损失计算
    loss = -0.5 * (torch.log(pos_term / denom_i2t) + torch.log(pos_term / denom_t2i)).mean()

    return loss


def supervised_infonce_loss(embeddings, labels, temperature=0.07,
                            hard_neg=True, topk=5, pos_threshold=0.8):
    # 特征归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 计算余弦相似度矩阵
    sim_matrix = embeddings @ embeddings.T / temperature

    # 构建正样本掩码
    pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).bool()
    pos_mask.fill_diagonal_(False)

    with torch.no_grad():
        neg_scores = sim_matrix.masked_fill(pos_mask, -np.inf)  # 排除正样本

        if hard_neg:
            num_neg = neg_scores.shape[1] - 1
            k = min(topk, num_neg)
            hard_neg_mask = neg_scores.topk(k=k, dim=1).indices
            neg_mask = torch.zeros_like(pos_mask).scatter(1, hard_neg_mask, 1).bool()
        else:
            # 传统负样本模式（全量负样本）
            neg_mask = ~pos_mask

        # 过滤高相似度假正样本
        high_sim = sim_matrix > pos_threshold
        pos_mask &= ~high_sim  # 排除相似度过高的潜在假正样本

    # 对比损失计算
    exp_sim = torch.exp(sim_matrix)

    # 分子：正样本相似度聚合
    pos_term = (sim_matrix * pos_mask).sum(dim=1)

    # 分母：难负样本聚合
    neg_denominator = (exp_sim * neg_mask).sum(dim=1)

    # 处理分母为零的情况，防止log(0)导致无穷大
    eps = 1e-8  # 一个小常数来避免除以零
    neg_denominator = torch.max(neg_denominator, torch.tensor(eps, device=neg_denominator.device))

    # 处理分子为零的情况，防止log(0)导致无穷大
    pos_term = torch.max(pos_term, torch.tensor(eps, device=pos_term.device))

    # 损失计算
    log_probs = -torch.log((exp_sim.diag() * pos_term) / (neg_denominator + eps))  # 加上一个小常数以避免除零

    valid_pos = pos_mask.sum(dim=1).clamp(min=1)
    loss = (log_probs / valid_pos.float()).mean()

    return loss


def triplet_loss(fa_emb, labels, margin=1.0, topk=3):
    dist_mat = torch.cdist(fa_emb, fa_emb, p=2)  # [B, B]

    pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1))  # 同类样本
    neg_mask = ~pos_mask

    losses = []
    for i in range(len(fa_emb)):
        pos_dist = dist_mat[i][pos_mask[i]]
        pos_dist = pos_dist[pos_dist != 0]  # 移除自身距离
        hard_pos = pos_dist.argmax() if len(pos_dist) > 0 else None

        neg_dist = dist_mat[i][neg_mask[i]]
        k = min(len(neg_dist), topk)
        hard_neg = neg_dist.topk(k, largest=False).indices

        if hard_pos is not None:
            # 确保 hard_pos 可迭代（它可能是标量）
            for hp in hard_pos.unsqueeze(0) if hard_pos.dim() == 0 else hard_pos:
                for hn in hard_neg:
                    loss = F.relu(dist_mat[i, hp] - dist_mat[i, hn] + margin)
                    losses.append(loss)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)



# def triplet_loss(fa_emb, positive_pairs, negative_pairs, margin=1.0):
#     # 计算正样本对的欧氏距离
#     positive_distance = 0
#     negative_distance = 0
#
#     for (anchor_idx, positive_idx), (anchor_idx, negative_idx) in zip(positive_pairs, negative_pairs):
#         anchor_emb = fa_emb[anchor_idx]
#         positive_emb = fa_emb[positive_idx]
#         negative_emb = fa_emb[negative_idx]
#
#         # 计算正样本对的距离
#         positive_distance += torch.norm(anchor_emb - positive_emb, p=2)  # 欧氏距离
#         negative_distance += torch.norm(anchor_emb - negative_emb, p=2)  # 欧氏距离
#
#     positive_distance /= len(positive_pairs)
#     negative_distance /= len(negative_pairs)
#     loss = torch.max(positive_distance - negative_distance + margin, torch.tensor(0.0, device=fa_emb.device))
#
#     return loss

def create_positive_negative_pairs(labels):
    """
    构造正负样本对。
    :param labels: 样本的标签，形状为 (batch_size,)
    :return: positive_pairs 和 negative_pairs，分别记录标签为1和0的样本索引
    """
    label_1_indices = [i for i, label in enumerate(labels) if label == 1]
    label_0_indices = [i for i, label in enumerate(labels) if label == 0]

    # 如果任何类别没有样本，返回 None
    if len(label_0_indices) == 0 or len(label_1_indices) == 0:
        return None, None

    # 1. 构建正样本对：同一类别的不同样本
    pos_pairs = []
    # 同一类别的正样本对 (label=0)
    pos_pairs += list(itertools.combinations(label_0_indices, 2))
    # 同一类别的正样本对 (label=1)
    pos_pairs += list(itertools.combinations(label_1_indices, 2))

    # 2. 构建负样本对：来自不同类别的样本对
    neg_pairs = []
    num_negative_pairs = len(labels)
    for _ in range(num_negative_pairs):
        idx_0 = random.choice(label_0_indices)
        idx_1 = random.choice(label_1_indices)
        neg_pairs.append((idx_0, idx_1))

    # 确保正负样本对数量相同
    pos_pairs = random.sample(pos_pairs, min(len(pos_pairs), len(neg_pairs)))

    return pos_pairs, neg_pairs


def contrastive_loss(embedding_fa, embedding_md, positive_pairs, negative_pairs, margin=1.0):
    """
    计算对比损失，包括跨模态和跨样本的正负样本对损失
    :param embedding_fa: FA 模态的嵌入，形状为 (batch_size, embedding_dim)
    :param embedding_md: MD 模态的嵌入，形状为 (batch_size, embedding_dim)
    :param positive_pairs: 正样本对，形状为 (2, num_positive_pairs)，分别对应 FA 和 MD 的样本索引
    :param negative_pairs: 负样本对，形状为 (2, num_negative_pairs)，分别对应 FA 和 MD 的样本索引
    :param margin: 负样本对的边界，控制负样本的损失
    :return: 计算的对比损失
    """
    positive_loss = 0.0
    negative_loss = 0.0
    for pair in positive_pairs:
        idx1, idx2 = pair[0], pair[1]
        positive_distance_fa_md = F.pairwise_distance(embedding_fa[idx1].unsqueeze(0), embedding_md[idx1].unsqueeze(0), p=2)
        positive_distance_fa = F.pairwise_distance(embedding_fa[idx1].unsqueeze(0), embedding_fa[idx2].unsqueeze(0), p=2)
        positive_distance_md = F.pairwise_distance(embedding_md[idx1].unsqueeze(0), embedding_md[idx2].unsqueeze(0), p=2)
        positive_loss += (positive_distance_fa_md.pow(2) + positive_distance_fa.pow(2) + positive_distance_md.pow(2))
    positive_loss /= (len(positive_pairs)*3)

    for pair in negative_pairs:
        idx1, idx2 = pair[0], pair[1]
        negative_distances_fa_md = F.pairwise_distance(embedding_fa[idx1].unsqueeze(0),embedding_md[idx1].unsqueeze(0), p=2)
        negative_distances_fa = F.pairwise_distance(embedding_fa[idx1].unsqueeze(0),embedding_fa[idx2].unsqueeze(0), p=2)
        negative_distances_md = F.pairwise_distance(embedding_md[idx1].unsqueeze(0),embedding_md[idx2].unsqueeze(0), p=2)
        negative_loss += F.relu(margin - negative_distances_fa_md).pow(2) + F.relu(margin - negative_distances_fa).pow(2) + F.relu(margin - negative_distances_md).pow(2)
    negative_loss /= (len(negative_pairs)*3)
    total_loss = positive_loss + negative_loss
    return total_loss



if __name__ == "__main__":
    labels = [1, 0, 1, 0, 1, 1, 0]  # 假设这是你的标签
    pos_pairs, neg_pairs = create_positive_negative_pairs(labels)

    print("正样本对:", pos_pairs)
    print("负样本对:", neg_pairs)

    fa_map = torch.randn(2, 1, 24, 28, 24)
    mri_map = torch.randn(2, 1, 12, 14, 12)
    model = SSIM3D(window_size=5, channels=1, sigma=1.5)
    ssim_loss = model(fa_map, mri_map)
    print("SSIM:", ssim_loss)
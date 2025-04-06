import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MedicalCrossModalLoss(nn.Module):
    """
    这个怎么理解，可以用在ssim里面吗？
    """
    def __init__(self, tau=0.07, alpha=0.3):
        super().__init__()
        self.tau = tau
        self.alpha = alpha  # ROI损失权重

    def forward(self, fa_emb, mri_emb, roi_attn):
        """
        :param roi_attn: ROI注意力图 [B, H, W]（参考网页4）
        """
        # 全局对齐损失
        global_loss = cross_modal_alignment_loss(fa_emb, mri_emb, self.tau)

        # ROI感知对齐
        # 1. ROI区域特征池化
        fa_roi = (roi_attn.unsqueeze(1) * fa_emb).sum(dim=(2, 3))  # [B,D]
        mri_roi = (roi_attn.unsqueeze(1) * mri_emb).sum(dim=(2, 3))

        # 2. 局部对比损失
        roi_sim = F.cosine_similarity(fa_roi, mri_roi, dim=1)
        roi_loss = -torch.log(torch.exp(roi_sim / self.tau) / torch.exp(roi_sim / self.tau).sum())

        return global_loss + self.alpha * roi_loss


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
            hard_neg_indices = neg_sim.topk(k=5, dim=1).indices
            neg_mask = torch.zeros_like(sim_matrix).scatter(1, hard_neg_indices, 1).bool()
    else:
        neg_mask = ~pos_mask

    # 5. 双向对比损失计算
    exp_sim = torch.exp(sim_matrix / tau)
    pos_term = exp_sim.diag()  # 分子：正样本相似度
    denom_i2t = (exp_sim * neg_mask).sum(dim=1)  # 分母：负样本聚合
    denom_t2i = (exp_sim * neg_mask).sum(dim=0)

    loss = -0.5 * (torch.log(pos_term / denom_i2t) + torch.log(pos_term / denom_t2i)).mean()
    return loss


def supervised_infonce_loss(embeddings, labels, temperature=0.07,
                            hard_neg=True, topk=5, pos_threshold=0.8):
    """
    改进版：支持难样本挖掘的监督对比损失
    :param hard_neg: 是否启用难负样本筛选（网页5/7）
    :param topk: 每个锚点选择topk最相似负样本（网页4/7）
    :param pos_threshold: 正样本相似度过滤阈值（网页6）
    """
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
            # 选择topk最相似负样本（网页4）
            hard_neg_mask = neg_scores.topk(k=topk, dim=1).indices
            neg_mask = torch.zeros_like(pos_mask).scatter(1, hard_neg_mask, 1).bool()
        else:
            # 传统负样本模式（全量负样本）
            neg_mask = ~pos_mask

        # 过滤高相似度假正样本（网页6）
        high_sim = sim_matrix > pos_threshold
        pos_mask &= ~high_sim  # 排除相似度过高的潜在假正样本

    # 对比损失计算（网页5/7）
    exp_sim = torch.exp(sim_matrix)

    # 分子：正样本相似度聚合（网页2）
    pos_term = (sim_matrix * pos_mask).sum(dim=1)

    # 分母：难负样本聚合（网页5）
    neg_denominator = (exp_sim * neg_mask).sum(dim=1)

    # 损失计算（网页5公式改造）
    log_probs = -torch.log((exp_sim.diag() * pos_term) / (neg_denominator + 1e-8))
    valid_pos = pos_mask.sum(dim=1).clamp(min=1)
    loss = (log_probs / valid_pos.float()).mean()

    return loss


def triplet_loss(fa_emb, labels, margin=1.0, topk=3):
    dist_mat = torch.cdist(fa_emb, fa_emb, p=2)  # [B,B]

    pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1))  # 同类样本
    neg_mask = ~pos_mask

    losses = []
    for i in range(len(fa_emb)):
        pos_dist = dist_mat[i][pos_mask[i]]
        pos_dist = pos_dist[pos_dist != 0]
        hard_pos = pos_dist.argmax() if len(pos_dist) > 0 else 0

        neg_dist = dist_mat[i][neg_mask[i]]
        hard_neg = neg_dist.topk(topk, largest=False).indices

        for hp in hard_pos:
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
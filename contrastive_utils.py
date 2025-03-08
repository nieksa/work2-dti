import torch
import torch.nn.functional as F
import random
def create_positive_negative_pairs(labels):
    """
    构造正负样本对。
    :param labels: 样本的标签，形状为 (batch_size,)
    :return: positive_pairs 和 negative_pairs，分别记录标签为1和0的样本索引
    """
    label_1_indices = [i for i, label in enumerate(labels) if label == 1]
    label_0_indices = [i for i, label in enumerate(labels) if label == 0]
    if len(label_0_indices) == 0 or len(label_1_indices) == 0:
        return None, None
    pos_pairs = []
    neg_pairs = []
    # 1. 正样本对：同一类别的不同样本
    for idx_0 in label_0_indices:
        for idx_1 in label_0_indices:
            if idx_0 != idx_1:  # 同一类别，不同样本
                pos_pairs.append([idx_0, idx_1])
    for idx_1 in label_1_indices:
        for idx_2 in label_1_indices:
            if idx_1 != idx_2:  # 同一类别，不同样本
                pos_pairs.append([idx_1, idx_2])
    # 2. 负样本对：从 label_0_indices 和 label_1_indices 中随机选取负样本对
    num_negative_pairs = len(labels)
    for _ in range(num_negative_pairs):
        idx_0 = random.choice(label_0_indices)
        idx_1 = random.choice(label_1_indices)
        neg_pairs.append([idx_0, idx_1])
    pos_pairs = random.sample(pos_pairs, min(len(pos_pairs), len(neg_pairs)))
    return pos_pairs, neg_pairs



def gaussian_3d(window_size, sigma):
    """ 生成 3D 高斯窗口 """
    kernel = torch.arange(window_size).float() - window_size // 2
    kernel = torch.exp(-kernel**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_3d = torch.einsum('i,j,k->ijk', kernel, kernel, kernel)
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    return kernel_3d

def compute_3d_ssim(img1, img2, window_size=11, sigma=1.5):
    """ 计算 3D SSIM """
    assert img1.shape == img2.shape, "输入图像形状必须一致"

    # 生成 3D 高斯窗口
    window = gaussian_3d(window_size, sigma).to(img1.device)

    # 计算局部均值
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=1)

    # 计算局部方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    # SSIM 计算
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def attention_pooling(feature_map):
    """
    通道维度的注意力池化（Attention Pooling），保持输出形状 (1, w, h, d)
    :param feature_map: 输入特征图，形状 (C, w, h, d)
    :return: 经过注意力池化的特征图，形状 (1, w, h, d)
    """
    # 计算通道维度上的注意力权重
    attn_weights = torch.softmax(feature_map.mean(dim=(1, 2, 3), keepdim=True), dim=0)  # (C, 1, 1, 1)
    pooled_feature = torch.sum(feature_map * attn_weights, dim=0, keepdim=True)  # (1, w, h, d)

    return pooled_feature

def compute_3d_max_pool_ssim(map_fa, map_md, idx1, idx2, window_size=11, sigma=1.5):
    """
    计算通过最大池化后的 3D SSIM

    :param map_fa: FA 模态的特征图，形状为 (batch, channel, w, h, d)
    :param map_md: MD 模态的特征图，形状为 (batch, channel, w, h, d)
    :param idx1: 当前正样本对的第一个索引
    :param idx2: 当前正样本对的第二个索引
    :param window_size: 3D 高斯窗口的大小
    :param sigma: 3D 高斯窗口的标准差
    :return: 计算的 3D SSIM 值
    """

    # 1. 根据索引获取特定样本, 形状 (channel, w, h, d)
    fa_sample = map_fa[idx1]  # (channel, w, h, d)
    md_sample = map_md[idx2]  # (channel, w, h, d)

    # 2. 在通道维度上做最大池化，得到 (1, w, h, d)
    fa_pooled = attention_pooling(fa_sample)  # (1, w, h, d)
    md_pooled = attention_pooling(md_sample)  # (1, w, h, d)

    # 3. 计算 3D SSIM
    ssim_value = compute_3d_ssim(fa_pooled.unsqueeze(0), md_pooled.unsqueeze(0), window_size, sigma)

    return ssim_value


def compute_contrastive_ssim_loss(map_fa, map_md, positive_pairs, negative_pairs, margin=1.0):
    """
    计算对比学习中的 SSIM 损失（考虑正负样本对）
    :param map_fa: FA 模态的特征图，形状为 (batch_size, C, D, H, W) batch, 128, 23, 23, 23
    :param map_md: MD 模态的特征图，形状为 (batch_size, C, D, H, W) batch, 128, 23, 23, 23
    :param positive_pairs: 正样本对，形状为 (2, num_positive_pairs)
    :param negative_pairs: 负样本对，形状为 (2, num_negative_pairs)
    :param margin: 负样本对的 SSIM 边界，控制负样本的损失
    :return: 计算的 SSIM 损失
    """
    positive_loss = 0.0
    negative_loss = 0.0

    # 计算正样本对的 SSIM 损失
    for pair in positive_pairs:
        idx1, idx2 = pair[0], pair[1]
        ssim_fa_md = compute_3d_max_pool_ssim(map_fa, map_md, idx1, idx2)
        positive_loss += (1 - ssim_fa_md)  # SSIM 越高，损失越低
    positive_loss /= len(positive_pairs)
    # 计算负样本对的 SSIM 损失
    for pair in negative_pairs:
        idx1, idx2 = pair[0], pair[1]
        ssim_fa_md = compute_3d_max_pool_ssim(map_fa, map_md, idx1, idx2)
        negative_loss += max(0, margin - ssim_fa_md)  # 超过 margin 则增加损失
    negative_loss /= len(negative_pairs)
    total_loss = positive_loss + negative_loss
    return total_loss


def contrastive_loss(embedding_fa, embedding_md, positive_pairs, negative_pairs, margin=1.0):
    # 1. 提取所有正负样本对的索引
    positive_indices_fa = [pair[0] for pair in positive_pairs]
    positive_indices_md = [pair[1] for pair in positive_pairs]
    negative_indices_fa = [pair[0] for pair in negative_pairs]
    negative_indices_md = [pair[1] for pair in negative_pairs]

    # 2. 计算所有正样本对的损失
    positive_distances_fa = F.pairwise_distance(embedding_fa[positive_indices_fa], embedding_fa[positive_indices_md],
                                                p=2)
    positive_distances_md = F.pairwise_distance(embedding_md[positive_indices_fa], embedding_md[positive_indices_md],
                                                p=2)
    positive_loss = (positive_distances_fa.pow(2) + positive_distances_md.pow(2)).mean()

    # 3. 计算所有负样本对的损失
    negative_distances_fa = F.pairwise_distance(embedding_fa[negative_indices_fa], embedding_fa[negative_indices_md],
                                                p=2)
    negative_distances_md = F.pairwise_distance(embedding_md[negative_indices_fa], embedding_md[negative_indices_md],
                                                p=2)
    negative_loss = F.relu(margin - negative_distances_fa).pow(2).mean() + F.relu(margin - negative_distances_md).pow(
        2).mean()

    # 总损失
    total_loss = positive_loss + negative_loss
    return total_loss


if __name__ == "__main__":
    # 设定随机种子保证可复现
    torch.manual_seed(42)

    # 生成随机3D数据 (batch=2, C=128, D=23, H=23, W=23)
    batch_size, C, D, H, W = 2, 128, 23, 23, 23
    fa_map = torch.rand((batch_size, C, D, H, W))
    md_map = torch.rand((batch_size, C, D, H, W))

    # 指定正样本对索引
    idx1, idx2 = 0, 1

    # 计算 SSIM
    ssim_value = compute_3d_max_pool_ssim(fa_map, md_map, idx1, idx2)

    # 打印结果
    print(f"SSIM between FA[{idx1}] and MD[{idx2}]: {ssim_value.item():.4f}")

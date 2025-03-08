import logging
from tqdm import tqdm
from contrastive_utils import create_positive_negative_pairs, contrastive_loss, compute_contrastive_ssim_loss
import torch
def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    contrastive_loss_total = 0
    classification_loss_total = 0
    ssim_loss_total = 0
    step = 0
    loss_function = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        # 1. 提取 fa, md 数据
        fa_data, md_data = data
        labels = labels.to(device)

        # 2. 获取嵌入 (第一阶段: 对比学习)
        fa_logit, fa_map, fa_emb, md_logit, md_map, md_emb, out_logit = model(fa_data, md_data)

        # 3. 构造正负样本对
        pos_pairs, neg_pairs = create_positive_negative_pairs(labels)
        if not pos_pairs or not neg_pairs:
            continue

        step += 1
        # 4. 计算对比损失
        ssim_loss = compute_contrastive_ssim_loss(fa_map, md_map, pos_pairs, neg_pairs, margin=1.0)
        contrastive_loss_val = contrastive_loss(fa_emb, md_emb, pos_pairs, neg_pairs, margin=1.0)

        # 5. 计算分类损失
        fa_loss = loss_function(fa_logit, labels)
        md_loss = loss_function(md_logit, labels)
        classification_loss = loss_function(out_logit, labels)

        # 6. 计算总损失
        alpha, beta, gamma = 1, 1, 1  # 可调超参数
        total_loss = gamma * ssim_loss + alpha * contrastive_loss_val + beta * (fa_loss + md_loss + classification_loss)

        # 7. 反向传播 + 优化
        total_loss.backward()
        optimizer.step()

        # 8. 记录损失值
        ssim_loss_total += ssim_loss.item()
        contrastive_loss_total += contrastive_loss_val.item()
        classification_loss_total += classification_loss.item()

    # 计算平均损失
    avg_epoch_loss = (ssim_loss_total + contrastive_loss_total + classification_loss_total) / step
    avg_ssim_loss = ssim_loss_total / step
    avg_contrastive_loss = contrastive_loss_total / step
    avg_classification_loss = classification_loss_total / step

    logging.info(
        f"Epoch Training - Total Loss: {avg_epoch_loss:.4f}, "
        f"SSIM Loss: {avg_ssim_loss:.4f}, "
        f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
        f"Classification Loss: {avg_classification_loss:.4f}"
    )
    scheduler.step()
    return
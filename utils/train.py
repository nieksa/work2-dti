import logging
from tqdm import tqdm
from contrastive_utils import create_positive_negative_pairs, contrastive_loss, compute_contrastive_ssim_loss
import torch
import math

def calculate_stage_weights(epoch, total_epochs=100):
    """
    计算多阶段非线性权重
    阶段1 (0-30%): 重点对比学习
    阶段2 (30-70%): 平衡学习
    阶段3 (70-100%): 重点分类
    """
    progress = epoch / total_epochs
    
    # 对比损失权重
    if progress < 0.3:  # 阶段1
        contrastive_weight = 1.0 - 0.5 * (progress / 0.3)  # 从1.0降到0.5
    elif progress < 0.7:  # 阶段2
        contrastive_weight = 0.5 - 0.3 * ((progress - 0.3) / 0.4)  # 从0.5降到0.2
    else:  # 阶段3
        contrastive_weight = 0.2 - 0.15 * ((progress - 0.7) / 0.3)  # 从0.2降到0.05
    
    # 分类损失权重
    if progress < 0.3:  # 阶段1
        classification_weight = 0.2 * (progress / 0.3)  # 从0增加到0.2
    elif progress < 0.7:  # 阶段2
        classification_weight = 0.2 + 0.3 * ((progress - 0.3) / 0.4)  # 从0.2增加到0.5
    else:  # 阶段3
        classification_weight = 0.5 + 0.4 * ((progress - 0.7) / 0.3)  # 从0.5增加到0.9
    
    # FA和MRI损失的独立权重
    if progress < 0.3:  # 阶段1
        fa_weight = 0.1 * (progress / 0.3)  # 从0增加到0.1
        mri_weight = 0.1 * (progress / 0.3)  # 从0增加到0.1
    elif progress < 0.7:  # 阶段2
        fa_weight = 0.1 + 0.2 * ((progress - 0.3) / 0.4)  # 从0.1增加到0.3
        mri_weight = 0.1 + 0.2 * ((progress - 0.3) / 0.4)  # 从0.1增加到0.3
    else:  # 阶段3
        fa_weight = 0.3 + 0.2 * ((progress - 0.7) / 0.3)  # 从0.3增加到0.5
        mri_weight = 0.3 + 0.2 * ((progress - 0.7) / 0.3)  # 从0.3增加到0.5
    
    return {
        'contrastive': contrastive_weight,
        'classification': classification_weight,
        'fa': fa_weight,
        'mri': mri_weight
    }

def train_contrastive_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    contrastive_loss_total = 0
    classification_loss_total = 0
    ssim_loss_total = 0
    fa_loss_total = 0
    mri_loss_total = 0
    step = 0
    loss_function = torch.nn.CrossEntropyLoss()
    
    # 获取当前阶段的权重
    weights = calculate_stage_weights(epoch)
    
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        # 1. 提取 fa, md 数据
        fa_data, mri_data = data
        fa_data = fa_data.to(torch.float32).to(device)
        mri_data = mri_data.to(torch.float32).to(device)
        labels = labels.to(device)

        # 2. 获取嵌入 (第一阶段: 对比学习)
        fa_logit, fa_map, fa_emb, mri_logit, mri_map, mri_emb, out_logit = model(fa_data, mri_data)

        # 3. 构造正负样本对
        pos_pairs, neg_pairs = create_positive_negative_pairs(labels)
        if not pos_pairs or not neg_pairs:
            continue

        step += 1
        # 4. 计算对比损失
        # ssim_loss = compute_contrastive_ssim_loss(fa_map, md_map, pos_pairs, neg_pairs, margin=1.0)
        contrastive_loss_val = contrastive_loss(fa_emb, mri_emb, pos_pairs, neg_pairs, margin=1.0)

        # 5. 计算分类损失
        fa_loss = loss_function(fa_logit, labels)
        mri_loss = loss_function(mri_logit, labels)
        classification_loss = loss_function(out_logit, labels)

        # 6. 使用多阶段非线性权重
        total_loss = (
            weights['contrastive'] * contrastive_loss_val +
            weights['classification'] * classification_loss +
            weights['fa'] * fa_loss +
            weights['mri'] * mri_loss
        )

        # 7. 反向传播 + 优化
        total_loss.backward()
        optimizer.step()

        # 8. 记录损失值
        contrastive_loss_total += contrastive_loss_val.item()
        classification_loss_total += classification_loss.item()
        fa_loss_total += fa_loss.item()
        mri_loss_total += mri_loss.item()

    # 计算平均损失
    avg_contrastive_loss = contrastive_loss_total / step
    avg_classification_loss = classification_loss_total / step
    avg_fa_loss = fa_loss_total / step
    avg_mri_loss = mri_loss_total / step

    logging.info(
        f"Epoch {epoch+1} - "
        f"Contrastive Loss (w={weights['contrastive']:.2f}): {avg_contrastive_loss:.4f}, "
        f"Classification Loss (w={weights['classification']:.2f}): {avg_classification_loss:.4f}, "
        f"FA Loss (w={weights['fa']:.2f}): {avg_fa_loss:.4f}, "
        f"MRI Loss (w={weights['mri']:.2f}): {avg_mri_loss:.4f}"
    )
    scheduler.step()
    return

def graph_train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    classification_loss_total = 0
    step = 0
    loss_function = torch.nn.CrossEntropyLoss()
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        labels = data.y
        optimizer.zero_grad()
        out_logit = model(data)

        step += 1
        classification_loss = loss_function(out_logit, labels)

        # 7. 反向传播 + 优化
        classification_loss.backward()
        optimizer.step()

        classification_loss_total += classification_loss.item()

    avg_classification_loss = classification_loss_total / step

    logging.info(
        f"Epoch {epoch+1} - Total Loss: {avg_classification_loss:.4f}"
    )
    scheduler.step()
    return
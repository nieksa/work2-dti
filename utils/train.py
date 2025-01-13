from tqdm import tqdm
import logging
import torch

def train_epoch(model, train_loader, loss_function, optimizer, device):
    """
    训练一个 epoch。

    参数:
    - model: 模型
    - train_loader: 训练集的 DataLoader
    - loss_function: 损失函数
    - optimizer: 优化器
    - device: 设备（如 'cuda' 或 'cpu'）

    返回:
    - epoch_loss: 训练集的平均损失
    - epoch_metrics: 训练集的指标字典（accuracy, f1, precision, recall, specificity）
    """
    model.train()
    epoch_loss = 0
    step = 0
    # epoch_preds = []
    # epoch_labels = []

    for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
        step += 1
        data = data.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        # epoch_preds.extend(preds.cpu().numpy())
        # epoch_labels.extend(labels.cpu().numpy())

    epoch_loss /= step
    logging.info(f"Train Loss       : {epoch_loss:.4f}")

    return
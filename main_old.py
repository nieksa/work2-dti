from data import ContrastiveDataset
from collections import Counter
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR
from models import create_model
from utils import rename_log_file, log_confusion_matrix, log_fold_results
import numpy as np
import logging
from statistics import mean, stdev
from utils.eval import save_best_model
from sklearn.model_selection import KFold
from utils.utils import set_seed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.eval import calculate_metrics
import argparse
import torch
import time
from contrastive_utils import create_positive_negative_pairs, contrastive_loss, compute_contrastive_ssim_loss
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
def setup_training_environment():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--model_name', type=str, default='contrastive_model1', help='Name of the model to use.')
    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
    parser.add_argument('--bs', type=int, default=16, help='I3D C3D cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')
    parser.add_argument('--debug', type=bool, default=False, help='small sample for debugging.')
    args = parser.parse_args()
    log_dir = f'./logs/{args.task}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./saved_models/{args.task}', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(log_dir, f'{args.model_name}_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 将日志输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        logging.info("Using single GPU.")
    logging.info(f"Training with {device}")
    return args, device, log_file, timestamp

seed = 42
set_seed(seed)
args, device, log_file, timestamp = setup_training_environment()

csv_file = 'data/data.csv'
dataset = ContrastiveDataset(csv_file, args)


all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                         'precision', 'recall', 'specificity']}

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
indices = np.arange(len(dataset))

for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_labels = dataset.labels[train_indices]
    val_labels = dataset.labels[val_indices]

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)

    table = [
        "+-------------------+-------+-------+",
        "|                   | Label 0 | Label 1 |",
        "+-------------------+-------+-------+",
        f"| Train            |   {train_counter[0]}    |   {train_counter[1]}    |",
        "+-------------------+-------+-------+",
        f"| Validation       |   {val_counter[0]}    |   {val_counter[1]}    |",
        "+-------------------+-------+-------+"
    ]
    for row in table:
        logging.info(row)

    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    model = create_model(args.model_name)
    model = DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    loss_function = torch.nn.CrossEntropyLoss()

    best_metric = {
        'accuracy': 0,
        'f1': 0,
    }
    best_metric_model = {
        'accuracy': None,
        'f1': None,
    }
    max_epochs = args.epochs

    val_start = 20
    val_interval = 1

    early_stop_start = 30
    patience = 5
    min_delta = 0.001
    best_val_metric = 0
    epochs_without_improvement = 0

    result_metric = None
    result_cm = None

    result_labels = None
    result_preds = None
    result_probs = None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        contrastive_loss_total = 0
        classification_loss_total = 0
        ssim_loss_total = 0
        step = 0

        for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
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
            # ssim_loss = 0
            contrastive_loss_val = contrastive_loss(fa_emb, md_emb, pos_pairs, neg_pairs, margin=1.0)
            # 5. 计算分类损失
            fa_loss = loss_function(fa_logit,labels)
            md_loss = loss_function(md_logit, labels)
            classification_loss = loss_function(out_logit, labels)
            # 6. 计算总损失
            alpha = 1
            beta = 1
            gamma = 1
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
            f"Epoch {epoch + 1} - Total Loss: {avg_epoch_loss:.4f}, "
            f"SSIM Loss: {avg_ssim_loss:.4f}, "
            f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
            f"Classification Loss: {avg_classification_loss:.4f}"
        )
        scheduler.step()
        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= val_start:
            model.eval()
            all_labels = []
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    fa_data, md_data = data
                    labels = labels.to(device)
                    fa_logit, fa_map, fa_emb, md_logit, md_map, md_emb, out_logit = model(fa_data, md_data)
                    preds = torch.argmax(out_logit, dim=1)
                    probs = torch.nn.functional.softmax(out_logit, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            cm = confusion_matrix(all_labels, all_preds)
            result = calculate_metrics(cm)
            accuracy = result['accuracy']
            balanced_accuracy = result['balanced_accuracy']
            kappa = result['kappa']
            recall = result['recall']
            specificity = result['specificity']
            precision = result['precision']
            f1 = result['f1']
            try:
                auc = roc_auc_score(all_labels, all_probs[:, 1], average='macro', multi_class='ovr')
            except ValueError:
                auc = 0.0
            avg_metrics = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'kappa': kappa,
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'specificity': specificity
            }
            logging.info(
                f"Val:{epoch + 1} | "
                f"Accuracy: {avg_metrics['accuracy']:.4f} | "
                f"BA: {avg_metrics['balanced_accuracy']:.4f} | "
                f"Kappa: {avg_metrics['kappa']:.4f} | "
                f"AUC: {avg_metrics['auc']:.4f} | "
                f"F1: {avg_metrics['f1']:.4f} | "
                f"Pre: {avg_metrics['precision']:.4f} | "
                f"Recall: {avg_metrics['recall']:.4f} | "
                f"Spec: {avg_metrics['specificity']:.4f}"
            )
            eval_metrics = avg_metrics
            current_val_metric = eval_metrics['accuracy']
            if result_cm is None:
                result_metric = eval_metrics
                result_cm = cm
                result_labels = all_labels
                result_preds = all_preds
                result_probs = all_probs
            if (epoch+1) > early_stop_start:
                if current_val_metric > (best_val_metric + min_delta):
                    best_val_metric = current_val_metric
                    epochs_without_improvement = 0
                    best_model_weights = model.state_dict().copy()
                    save_best_model(best_model_weights,
                                    eval_metrics,
                                    best_metric,
                                    best_metric_model,
                                    args,
                                    timestamp,
                                    fold=fold,
                                    epoch=epoch+1,
                                    metric_name='accuracy')
                    result_metric = eval_metrics
                    result_cm = cm
                    result_labels = all_labels
                    result_preds = all_preds
                    result_probs = all_probs
                else:
                    epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            logging.info(f"Early Stopping at Epoch {epoch + 1}. Val Metric did not improve for {patience} epochs.")
            break
    avg_metrics = result_metric
    log_confusion_matrix(result_cm)
    logging.info(
        f"Accuracy : {avg_metrics['accuracy']:.4f} | "
        f"BA: {avg_metrics['balanced_accuracy']:.4f} | "
        f"Kappa: {avg_metrics['kappa']:.4f} | "
        f"AUC: {avg_metrics['auc']:.4f} | "
        f"F1: {avg_metrics['f1']:.4f} | "
        f"Pre: {avg_metrics['precision']:.4f} | "
        f"Recall: {avg_metrics['recall']:.4f} | "
        f"Spec: {avg_metrics['specificity']:.4f}"
    )
    log_fold_results(fold + 1, result_labels, result_preds, result_probs)
    logging.info(f"Fold {fold + 1} end")
    logging.info('--------------------------------------------------------------------\n')

    for metric, value in avg_metrics.items():
        all_metrics[metric].append(value)

result_message = ''
for metric, values in all_metrics.items():
    avg = mean(values)
    std = stdev(values)
    result_message += f"{avg * 100:.2f}±{std * 100:.2f}\t"

avg_acc = mean(all_metrics['accuracy']) * 100
logging.info(f"\n{result_message}")
logging.shutdown()

rename_log_file(log_file, avg_acc, args.task, args.model_name, timestamp)
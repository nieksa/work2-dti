import torch
from data import Contrastive_dataset
from collections import Counter
from torch.utils.data import DataLoader, Subset
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from models import create_model
from utils import setup_training_environment, rename_log_file, train_epoch, log_confusion_matrix, log_fold_results
import numpy as np
import logging
from statistics import mean, stdev
from utils.eval import eval_model, save_best_model
from sklearn.model_selection import KFold
from utils.utils import set_seed
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.eval import calculate_metrics
from contrastive_loss import nt_xent_loss

args, device, log_file, timestamp = setup_training_environment()

csv_file = '../data/data.csv'

dataset = Contrastive_dataset(csv_file, args)

seed = 42
set_seed(seed)
# 下面这两项取反可确保可重复性，但是降低训练速度
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # CuDNN 的自动优化



all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                         'precision', 'recall', 'specificity']}


subject_id = np.array(dataset.subject_id)
unique_ids = np.unique(subject_id)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

subject_to_indices = defaultdict(list)
for idx, subject in enumerate(subject_id):
    subject_to_indices[subject].append(idx)

for fold, (train_ids, val_ids) in enumerate(kfold.split(unique_ids)):
    logging.info(f'Fold {fold+1} Start')
    logging.info('--------------------------------------------------------------------')

    train_participants = unique_ids[train_ids]
    val_participants = unique_ids[val_ids]

    train_indices = np.concatenate([subject_to_indices[subject] for subject in train_participants])
    val_indices = np.concatenate([subject_to_indices[subject] for subject in val_participants])

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_labels = dataset.labels[train_indices]
    val_labels = dataset.labels[val_indices]

    # 统计标签分布
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)

    # 打印标签分布表
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

    # 创建DataLoader
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    model = create_model(args.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
    loss_function = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids).to(device)

    best_metric = {
        'accuracy': 0,
        'f1': 0,
    }
    best_metric_model = {
        'accuracy': None,
        'f1': None,
    }
    max_epochs = args.epochs

    val_start = 0
    val_interval = 1

    early_stop_start = 20
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
        step = 0

        for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
            step += 1
            # 获取FA、L1、MD模态数据
            data_fa = data[0].to(device)
            data_l1 = data[1].to(device)
            data_md = data[2].to(device)
            labels = labels.to(device).long()

            # 前向传播：获取分类输出和投影向量
            fa_class_out, fa_projection = model(data_fa)
            l1_class_out, l1_projection = model(data_l1)
            md_class_out, md_projection = model(data_md)

            # 对比学习损失（以NT-Xent为例）
            contrastive_loss = nt_xent_loss(fa_projection, l1_projection, md_projection, temperature=0.3)

            combined_class_out = (fa_class_out + l1_class_out + md_class_out) / 3
            classification_loss = loss_function(combined_class_out, labels)

            # 总损失
            total_loss = 0.5 * contrastive_loss + 0.5 * classification_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += total_loss.item()

        epoch_loss /= step
        logging.info(f"Epoch {epoch + 1} - Train Loss: {epoch_loss:.4f}")

        # 验证阶段
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                data_fa = data[0].to(device)
                data_l1 = data[1].to(device)
                data_md = data[2].to(device)
                labels = labels.to(device)

                # 前向传播
                fa_class_out, _ = model(data_fa)
                l1_class_out, _ = model(data_l1)
                md_class_out, _ = model(data_md)

                # 分类输出融合
                combined_class_out = (fa_class_out + l1_class_out + md_class_out) / 3
                probs = combined_class_out.softmax(dim=1)
                _, preds = torch.max(combined_class_out, 1)

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
        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= val_start:
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
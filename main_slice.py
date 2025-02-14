import torch
from data import SliceDataset
from collections import Counter
from torch.utils.data import DataLoader, Subset
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from models import create_model
from utils import rename_log_file, train_epoch, log_confusion_matrix, log_fold_results
import numpy as np
import logging
from statistics import mean, stdev
from utils.eval import eval_model, save_best_model
from sklearn.model_selection import KFold
from utils.utils import set_seed
from collections import defaultdict
import os
import time
import logging
import argparse

def setup_training_environment():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--model_name', type=str, default='3D_ResNet18', help='Name of the model to use.')
    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
    parser.add_argument('--bs', type=int, default=32, help='I3D C3D cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')
    parser.add_argument('--debug', type=bool, default=False, help='small sample for debugging.')

    # 解析命令行参数
    args = parser.parse_args()

    # 创建日志和模型保存目录
    log_dir = f'./logs/{args.task}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'./saved_models/{args.task}', exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(log_dir, f'{args.model_name}_{timestamp}.log')

    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 将日志输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    # 打印训练配置
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        logging.info("Using single GPU.")

    logging.info(f"Training with {device}")

    return args, device, log_file, timestamp

args, device, log_file, timestamp = setup_training_environment()
csv_file = './data/data.csv'

dataset = SliceDataset(csv_file, args)

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

    val_start = 20
    val_interval = 1

    early_stop_start = 50
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
        train_epoch(model, train_loader, loss_function, optimizer, device)
        eval_metrics, cm, all_labels, all_preds, all_probs = eval_model(model=model, dataloader=val_loader, device=device, epoch=epoch+1)
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
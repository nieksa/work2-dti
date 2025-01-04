import torch
from data import DTIDataset
from torch.utils.data import DataLoader, Subset
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from models import create_model
from utils.utils import setup_training_environment, rename_log_file
import numpy as np
import logging
from statistics import mean, stdev
from utils.eval import eval_model, save_best_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
from tqdm import tqdm
from utils.utils import set_seed

seed = 42
set_seed(seed)
# 下面这两项取反可确保可重复性，但是降低训练速度
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # CuDNN 的自动优化


args, device, log_file, timestamp = setup_training_environment()
all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                         'precision', 'recall', 'specificity']}


csv_file = './data/ppmi/data.csv'

# channels = ["06","07","FA","L1","L23m","MD"]
channels = ["06"]
# template = ["1mm" "2mm" "s6mm"]
template = "s6mm"
dataset = DTIDataset(csv_file, args, channels=channels, transform=None, template=template)

subject_id = np.array(dataset.subject_id)
unique_ids = np.unique(subject_id)

k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

for fold, (train_ids, val_ids) in enumerate(kfold.split(unique_ids)):
    logging.info(f'Fold {fold+1} Start')
    logging.info('--------------------------------------------------------------------')

    train_participants = unique_ids[train_ids]
    val_participants = unique_ids[val_ids]

    train_indices = np.where(np.isin(subject_id, train_participants))[0]
    val_indices = np.where(np.isin(subject_id, val_participants))[0]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_labels = [train_subset.dataset[i][1] for i in train_subset.indices]
    val_labels = [val_subset.dataset[i][1] for i in val_subset.indices]
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

    train_loader = DataLoader(train_subset, batch_size=args.train_bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.val_bs, shuffle=False)

    model = create_model(args.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
    loss_function = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids).to(device)

    metric_values = []

    # 这个是考虑根据哪个指标来保存模型
    best_metric = {
        'accuracy': 0,
        'f1': 0,
    }
    best_metric_model = {
        'accuracy': None,
        'f1': None,
    }
    max_epochs = args.epochs

    # 验证开始轮次和验证间隔
    val_start = 30
    val_interval = 1

    # 早停设置
    patience = 5
    min_delta = 0.001
    best_val_metric = 0
    epochs_without_improvement = 0
    best_model_weights = None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_preds = []
        epoch_labels = []

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
            epoch_len = len(train_subset) // train_loader.batch_size
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            _, preds = torch.max(outputs, 1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())


        epoch_loss /= step
        epoch_acc = accuracy_score(epoch_labels, epoch_preds)
        epoch_f1 = f1_score(epoch_labels, epoch_preds, average='binary')
        epoch_precision = precision_score(epoch_labels, epoch_preds, average='binary', zero_division=0)
        epoch_recall = recall_score(epoch_labels, epoch_preds, average='binary')
        tn, fp, fn, tp = confusion_matrix(epoch_labels, epoch_preds).ravel()
        epoch_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        logging.info(f"Epoch {epoch + 1} ----------------------------------------------")
        logging.info(f"Train Loss       : {epoch_loss:.4f}")
        logging.info(f"Train Accuracy   : {epoch_acc:.4f}")
        logging.info(f"Train F1 Score   : {epoch_f1:.4f}")
        logging.info(f"Train Precision  : {epoch_precision:.4f}")
        logging.info(f"Train Recall     : {epoch_recall:.4f}")
        logging.info(f"Train Specificity: {epoch_spec:.4f}")

        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= val_start:
            eval_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch=epoch + 1)
            current_val_metric = eval_metrics['accuracy']
            if current_val_metric > best_val_metric + min_delta:
                epochs_without_improvement = 0
                best_model_weights = model.state_dict().copy()
                save_best_model(model, eval_metrics, best_metric, best_metric_model, args, timestamp,
                                fold=fold, epoch=epoch+1, metric_name='accuracy')
                best_val_metric = current_val_metric
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            model.load_state_dict(best_model_weights)
            logging.info(f"Early Stopping at Epoch {epoch + 1}. Val Metric did not improve for {patience} epochs.")
            break


    avg_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch='FINAL')
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
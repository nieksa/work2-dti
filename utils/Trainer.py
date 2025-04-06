from utils.utils import rename_log_file
from utils.eval import calculate_metrics
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from models import create_model
from statistics import mean, stdev
import torch
import os
import re
import numpy as np
import random
from sklearn.model_selection import KFold
from data.BalancedSampler import BalancedSampler

class BaseTrainer:
    def __init__(self, dataset, optimizer, scheduler, args, timestamp, seed=42):
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.seed = seed
        self.save_model_path = "./save_models"
        self.timestamp = timestamp

    def set_seed(self):
        # 设置随机种子
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self, train_indices, val_indices):
        # 加载数据集
        # 接受start_training()传入的indices,记录数据分布指标
        # 然后根据indices划分数据集
        # 返回train_loader, val_loader
        # 如果是图结构数据，则需要重写，因为图结构数据需要用dataloader的collate_fn来处理
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

        # 实例化 balanced sampler
        train_sampler = BalancedSampler(train_subset)

        train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True, sampler=train_sampler)
        val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)
        return train_loader, val_loader


    def model_evaluate(self, data_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for data, labels in tqdm(data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                out_logit = self.model_output(data)
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

        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1], average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0

        avg_metrics = {
            'accuracy': result['accuracy'],
            'balanced_accuracy': result['balanced_accuracy'],
            'kappa': result['kappa'],
            'auc': auc,
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'specificity': result['specificity']
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
        return avg_metrics, cm, all_labels, all_preds, all_probs

    def start_training(self):
        all_metrics = []
        kfold = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=self.seed)
        indices = np.arange(len(dataset))
        for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            print(f"Training fold {fold + 1}/{self.args.k_folds}")
            self.model = create_model(self.args.model_name)
            train_loader, val_loader = self.load_data(train_indices, val_indices)
            fold_metrics = self.cross_validation(fold, train_loader, val_loader)
            all_metrics.append(fold_metrics)

        result_message = ''
        for metric, values in all_metrics.items():
            avg = mean(values)
            std = stdev(values)
            result_message += f"{avg * 100:.2f}±{std * 100:.2f}\t"
        avg_acc = mean(all_metrics['accuracy']) * 100
        logging.info(f"\n{result_message}")
        logging.shutdown()
        rename_log_file(log_file, avg_acc, args.task, args.model_name, timestamp)


    def cross_validation(self, fold, train_loader, val_loader):
        # 进入每个epoch训练模型，评估模型，记录指标
        best_metrics = {
            'accuracy': 0,
            'balanced_accuracy': 0,
            'kappa': 0,
            'auc': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0,
            'specificity': 0,
        }
        best_epoch = 0
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch, train_loader)
            
            if epoch > self.args.val_start and epoch % self.args.val_interval == 0:
                metrics, cm, all_labels, all_preds, all_probs = self.model_evaluate(epoch, val_loader)
                
                if metrics[self.args.pick_metric_name] > best_metrics[self.args.pick_metric_name]:
                    best_metrics = metrics
                    best_epoch = epoch
                    self.save_model(epoch, metrics, fold)
                    self.early_stop_counter = 0
                else:
                    # 如果当前验证指标没有提高，早停计数器增加
                    self.early_stop_counter += 1

                # 如果早停计数器超过耐心次数，则提前停止
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        # 如果已经达到最大epoch，保存最终模型
        if epoch == self.args.epochs - 1:
            print(f"Maximum epochs reached. Saving model at epoch {epoch + 1}.")
            self.save_model(epoch, best_metrics, fold)

        return best_metrics

    def save_model(self, epoch, metrics, fold):
        model_filename = f"{self.args.model_name}_fold_{fold + 1}_epoch_{epoch + 1}_{self.timestamp}_acc_{metrics['accuracy']:.4f}.pth"
        model_filepath = os.path.join(self.save_model_path, self.args.task, model_filename)

        if not os.path.exists(os.path.dirname(model_filepath)):
            os.makedirs(os.path.dirname(model_filepath))

        pattern = f"{self.args.model_name}_fold_{fold + 1}_.*_{self.timestamp}_.*.pth"
        model_files = [f for f in os.listdir(os.path.join(self.save_model_path, self.args.task)) if re.match(pattern, f)]

        for old_model_file in model_files:
            os.remove(os.path.join(self.save_model_path, self.args.task, old_model_file))

        torch.save(self.model.state_dict(), model_filepath)
        logging.info(f"Model saved at {model_filepath}")

    def model_output(self, data):
        out_logit = self.model(data)
        return out_logit

    def train_epoch(self, epoch, train_loader):
        # 具体的训练过程和反向传播优化，并且记录损失，如果是多任务学习就需要重写记录损失的逻辑
        raise NotImplementedError("train_epoch() should be implemented by subclass")




class ContrastiveTrainer(BaseTrainer):
    def __init__(self, dataset, optimizer, scheduler, args, timestamp, seed=42):
        super().__init__(dataset, optimizer, scheduler, args, timestamp, seed)

    def model_output(self, data):
        # 针对多任务学习模型的重写
        pass

    def train_epoch(self, epoch, train_loader):
        # 具体的训练过程和反向传播优化，并且记录损失，如果是多任务学习就需要重写记录损失的逻辑
        raise NotImplementedError("train_epoch() should be implemented by subclass")



class GraphTrainer(BaseTrainer):
    def __init__(self, dataset, optimizer, scheduler, args, timestamp, seed=42):
        super().__init__(dataset, optimizer, scheduler, args, timestamp, seed)

    def load_data(self, train_indices, val_indices):
        # 针对图数据集的加载数据步骤
        pass

    def model_output(self, data):
        # 针对多任务学习模型的重写
        pass

    def train_epoch(self, epoch, train_loader):
        # 具体的训练过程和反向传播优化，并且记录损失，如果是多任务学习就需要重写记录损失的逻辑
        raise NotImplementedError("train_epoch() should be implemented by subclass")
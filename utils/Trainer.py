from utils.eval import calculate_metrics
from collections import Counter
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils import rename_log_file, log_confusion_matrix, log_fold_results
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.nn import DataParallel
from tqdm import tqdm
from models import create_model
from statistics import mean, stdev
import torch
import os
import re
import numpy as np
import random
from sklearn.model_selection import KFold
from data.BalancedSampler import BalancedSampler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import logging

class BaseTrainer:
    def __init__(self, dataset, args, timestamp, seed=42):
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.seed = seed
        self.save_model_path = "./saved_models"
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
        # 如果是图结构数据，则需要重写，因为图结构数据需要用dataloader的collate_fn来处理
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)

        train_labels = self.dataset.labels[train_indices].tolist()
        val_labels = self.dataset.labels[val_indices].tolist()

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
        # train_sampler = BalancedSampler(train_subset.dataset)
        train_loader = DataLoader(train_subset, batch_size=self.args.bs, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.args.bs, shuffle=False)
        return train_loader, val_loader


    def model_evaluate(self, epoch, data_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for data, labels in tqdm(data_loader):
                # data = data.to(self.device)
                # labels = labels.to(self.device)
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

    def start_training(self, log_file):
        all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                                 'precision', 'recall', 'specificity']}
        kfold = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=self.seed)
        indices = np.arange(len(self.dataset))
        for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            logging.info(f"Training fold {fold + 1}/{self.args.k_folds}")
            train_loader, val_loader = self.load_data(train_indices, val_indices)
            self.model = create_model(self.args.model_name)
            self.model = DataParallel(self.model).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
            # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
            

            fold_metrics = self.cross_validation(fold, train_loader, val_loader)
            for metric, value in fold_metrics.items():
                all_metrics[metric].append(value)

        result_message = ''
        for metric, values in all_metrics.items():
            avg = mean(values)
            std = stdev(values)
            result_message += f"{avg * 100:.2f}±{std * 100:.2f}\t"
        avg_acc = mean(all_metrics['accuracy']) * 100
        logging.info(f"\n{result_message}")
        logging.shutdown()
        rename_log_file(log_file, avg_acc, self.args.task, self.args.model_name, self.timestamp)


    def cross_validation(self, fold, train_loader, val_loader):
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
        best_cm = None
        best_all_labels = None
        best_all_preds = None
        best_all_probs = None
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch, train_loader)
            
            if (epoch+1) >= self.args.val_start and (epoch+1) % self.args.val_interval == 0:
                metrics, cm, all_labels, all_preds, all_probs = self.model_evaluate(epoch, val_loader)
                
                if metrics[self.args.pick_metric_name] > best_metrics[self.args.pick_metric_name]:
                    best_metrics = metrics
                    best_cm = cm
                    best_all_labels = all_labels
                    best_all_preds = all_preds
                    best_all_probs = all_probs
                    self.save_model(epoch, metrics, fold)
                    self.early_stop_counter = 0
                else:
                    if epoch > self.args.early_stop_start:
                        self.early_stop_counter += 1

                if self.early_stop_counter >= self.args.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}.")
                    break
        logging.info(f"Maximum epochs reached. Saving model at epoch {self.args.epochs}.")
        log_confusion_matrix(best_cm)
        log_fold_results(fold,best_all_labels,best_all_preds,best_all_probs)
        
        self.save_model(self.args.epochs, best_metrics, fold)

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
    def __init__(self, dataset, args, timestamp, seed=42):
        super().__init__(dataset, args, timestamp, seed)

    def model_output(self, data):
        fa_data, mri_data = data
        fa_data = fa_data.to(torch.float32).to(self.device)
        mri_data = mri_data.to(torch.float32).to(self.device)
        _, _, _, _, _, _, out_logit = self.model(fa_data, mri_data)
        return out_logit

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        classification_loss_total = 0
        dti_loss_total = 0
        nce_loss_total = 0
        ssim_loss_total = 0
        step = 0
        loss_function = torch.nn.CrossEntropyLoss()

        weights = {
            'contrastive': 0.3,
            'classification': 1,
            'ssim': 0.3
        }

        for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            self.optimizer.zero_grad()
            fa_data, mri_data = data
            fa_data = fa_data.to(torch.float32).to(self.device)
            mri_data = mri_data.to(torch.float32).to(self.device)
            labels = labels.to(self.device)

            fa_logit, fa_map, fa_emb, mri_logit, mri_map, mri_emb, out_logit = self.model(fa_data, mri_data)
            # print(mri_emb)
            # print(out_logit)
            step += 1
            
            # 1.构建正负样本对
            # from utils.contrastive_utils import create_positive_negative_pairs
            # pos_pairs, neg_pairs = create_positive_negative_pairs(labels)

            # 2.使用三元组损失计算DTI数据的对比损失，因为DTI数据是各向异性的。
            from utils.contrastive_utils import triplet_loss
            from utils.contrastive_utils import supervised_infonce_loss
            # dti_loss = triplet_loss(fa_emb, labels, margin=1.0, topk=5, grad_clip_norm=1.0)
            dti_loss = supervised_infonce_loss(fa_emb, labels, temperature=0.2,
                            hard_neg=True, topk=5, pos_threshold=0.8, grad_clip_norm=1.0)

            # 3.使用InfoNCE损失计算MRI数据的对比损失，因为MRI数据是各向同性的。
            nce_loss = supervised_infonce_loss(mri_emb, labels, temperature=0.2,
                            hard_neg=True, topk=5, pos_threshold=0.8, grad_clip_norm=1.0)

            # 4.使用交叉熵损失计算分类损失
            classification_loss = loss_function(out_logit, labels)

            # 5.使用跨模态对齐损失计算fa_emb和mri_emb之间的对齐损失
            from utils.contrastive_utils import cross_modal_alignment_loss
            cross_modal_loss = cross_modal_alignment_loss(fa_emb, mri_emb, tau=0.07, hard_neg=True, grad_clip_norm=1.0)

            # 6.使用ssim计算fa_map和mri_map之间的相似度损失，这个的医学支撑是脑部病变发生在相同的ROI区域，所以热图关注应该是在同一个地方
            from utils.contrastive_utils import SSIM3D
            ssim_model = SSIM3D(window_size=5, channels=1, sigma=1.5)
            ssim_model = DataParallel(ssim_model).to(self.device)
            ssim_loss = ssim_model(fa_map, mri_map)
            ssim_loss = ssim_loss.mean()
            ssim_loss = ssim_loss.detach()
            # print(f"ssim_loss:{ssim_loss}")

            total_loss = classification_loss + dti_loss + nce_loss + cross_modal_loss + ssim_loss
            # 7. 反向传播 + 优化
            total_loss.backward()
            self.optimizer.step()

            classification_loss_total += classification_loss.item()
            dti_loss_total += dti_loss.item()
            nce_loss_total += nce_loss.item()
            cross_modal_loss_total += cross_modal_loss.item()
            ssim_loss_total += ssim_loss.item()

        avg_classification_loss = classification_loss_total / step
        avg_dti_loss = dti_loss_total / step
        avg_nce_loss = nce_loss_total / step
        avg_ssim_loss = ssim_loss_total / step
        avg_cross_modal_loss = cross_modal_loss_total / step

        logging.info(
            f"Epoch {epoch + 1} - "
            f"Classification Loss (w={weights['classification']:.2f}): {avg_classification_loss:.4f}, "
            f"DTI Loss (w={weights['contrastive']:.2f}): {avg_dti_loss:.4f}, "
            f"MRI Loss (w={weights['contrastive']:.2f}): {avg_nce_loss:.4f}, "
            f"Cross-Modal Loss (w={weights['cross_modal']:.2f}): {avg_cross_modal_loss:.4f}, "
            f"SSIM Loss (w={weights['ssim']:.2f}): {avg_ssim_loss:.4f}, "
        )
        self.scheduler.step()
        return


class GraphTrainer(BaseTrainer):
    def __init__(self, dataset, args, timestamp, seed=42):
        super().__init__(dataset, args, timestamp, seed)

    def load_data(self, train_indices, val_indices):
        # 针对图数据集的加载数据步骤
        pass

    def model_output(self, data):
        # 针对多任务学习模型的重写
        pass

    def train_epoch(self, epoch, train_loader):
        # 具体的训练过程和反向传播优化，并且记录损失，如果是多任务学习就需要重写记录损失的逻辑
        raise NotImplementedError("train_epoch() should be implemented by subclass")
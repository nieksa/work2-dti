import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import logging
import os
from tabulate import tabulate

def log_confusion_matrix(cm):
    """
    使用 tabulate 库将混淆矩阵记录到日志文件中
    :param cm: 混淆矩阵，格式为 [[TN, FP], [FN, TP]]
    """
    table = [
        ["Actual \\ Predicted", "Negative", "Positive"],
        ["Negative", cm[0, 0], cm[0, 1]],
        ["Positive", cm[1, 0], cm[1, 1]]
    ]
    logging.info("Confusion Matrix:\n" + tabulate(table, headers="firstrow", tablefmt="grid"))

def calculate_metrics(cm):
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    accuracy = (TP + TN) / np.sum(cm)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    balanced_accuracy = (sensitivity + specificity) / 2
    p_o = (TP + TN) / (TP + FP + FN + TN)
    p_e = ((TP + FP) / (TP + FP + FN + TN)) * ((TP + FN) / (TP + FP + FN + TN)) + ((TN + FN) / (TP + FP + FN + TN)) * ((TN + FP) / (TP + FP + FN + TN))
    kappa = (p_o - p_e) / (1 - p_e)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    specificity = TN / (TN + FP)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'kappa': kappa,
        'recall': recall,
        'specificity': specificity,
        'precision' : precision,
        'f1': f1
    }
def eval_model(model, dataloader, device, epoch):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型推理
            outputs = model(inputs)
            probs = outputs.softmax(dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    all_probs = np.array(all_probs)  # 先转换为一个单一的 numpy.ndarray
    all_probs = torch.tensor(all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    result = calculate_metrics(cm)

    if epoch=='FINAL':
        log_confusion_matrix(cm)

    accuracy = result['accuracy']
    balanced_accuracy = result['balanced_accuracy']
    kappa = result['kappa']
    recall = result['recall']
    specificity = result['specificity']
    precision = result['precision']
    f1 = result['f1']

    try:
        auc = roc_auc_score(all_labels, all_probs[:,1], average='macro', multi_class='ovr')
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
    # logging.info(
    #     f"Epoch:{epoch} | "
    #     f"Accuracy: {avg_metrics['accuracy']:.4f} | "
    #     f"BA: {avg_metrics['balanced_accuracy']:.4f} | "
    #     f"Kappa: {avg_metrics['kappa']:.4f} | "
    #     f"AUC: {avg_metrics['auc']:.4f} | "
    #     f"F1: {avg_metrics['f1']:.4f} | "
    #     f"Pre: {avg_metrics['precision']:.4f} | "
    #     f"Recall: {avg_metrics['recall']:.4f} | "
    #     f"Spec: {avg_metrics['specificity']:.4f}"
    # )
    return avg_metrics


def save_best_model(model, eval_metric, best_metric, best_metric_model, args, timestamp, fold, epoch, metric_name):
    model_name = args.model_name
    task = args.task
    if eval_metric[metric_name] >= best_metric[metric_name]:
        best_metric[metric_name] = eval_metric[metric_name]
        model_path = f'./saved_models/{task}/{model_name}_{timestamp}_fold_{fold + 1}_epoch_{epoch}_{metric_name}_{best_metric[metric_name]:.2f}.pth'
        if metric_name in best_metric_model and best_metric_model[metric_name]:
            old_model_path = best_metric_model[metric_name]
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
        best_model = model
        best_metric_model[metric_name] = model_path
        torch.save(best_model.state_dict(),best_metric_model[metric_name])




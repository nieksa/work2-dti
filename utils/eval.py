import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import logging
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
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    p_o = (TP + TN) / (TP + FP + FN + TN)
    p_e = ((TP + FP) / (TP + FP + FN + TN)) * ((TP + FN) / (TP + FP + FN + TN)) + ((TN + FN) / (TP + FP + FN + TN)) * ((TN + FP) / (TP + FP + FN + TN))
    kappa = (p_o - p_e) / (1 - p_e)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'kappa': kappa,
        'recall': recall,
        'specificity': specificity,
        'precision' : precision,
        'f1': f1
    }

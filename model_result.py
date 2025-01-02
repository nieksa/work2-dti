import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import torch
from data import DTIDataset
import argparse
from models.utils import create_model
import os
import glob
from utils import set_seed, plot_confusion_matrix, evaluate_model, plot_pr_curve, plot_roc_curve
import time

# 下面这两项取反可确保可重复性，但是降低训练速度
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # CuDNN 的自动优化
def main():
    seed = 42
    set_seed(seed)
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model_names', nargs='+', default=['ResNet18', 'Design6'],
                        help='List of model names to evaluate.')
    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal'])
    parser.add_argument('--val_bs', type=int, default=16, help='densenet cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')
    parser.add_argument('--debug', type=bool, default=False, help='small sample for debugging.')
    parser.add_argument('--csv_file', type=str, default='./data/ppmi/data.csv')
    parser.add_argument('--data_dir', type=str, default='./data/ppmi/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, args.task, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # channels = ["06","07","FA","L1","L23m","MD"]
    channels = ["06"]
    dataset = DTIDataset(args.csv_file, args, channels=channels)

    subject_id = np.array(dataset.subject_id)
    unique_ids = np.unique(subject_id)

    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_ids)):
        if fold + 1 == args.fold:
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=args.val_bs, shuffle=False)
            break

    if args.task == 'NCvsPD':
        class1 = 'NC'  # 正常对照组
        class2 = 'PD'  # 帕金森病组
    elif args.task == 'NCvsProdromal':
        class1 = 'NC'  # 正常对照组
        class2 = 'Prodromal'  # 前驱期组
    elif args.task == 'ProdromalvsPD':
        class1 = 'Prodromal'  # 前驱期组
        class2 = 'PD'  # 帕金森病组
    else:
        raise ValueError(f"Unknown task: {args.task}")


    all_labels = None
    all_probs_list = []
    model_list = []
    for model_name in args.model_names:
        print(f'Evaluating model {model_name}')
        weights_file_pattern = f"{model_name}_*_fold_{args.fold}_*.pth"
        weights_path_pattern = os.path.join("saved_models", args.task, weights_file_pattern)
        matching_files = glob.glob(weights_path_pattern)
        if not matching_files:
            # raise FileNotFoundError(f"No weights files found in: {weights_path_pattern}")
            print(f"No weights files found in: {weights_path_pattern}")
            continue
        weights_path = matching_files[0]
        model = create_model(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

        labels, preds, probs = evaluate_model(model, val_loader, device)
        if all_labels is None:
            all_labels = labels
        all_probs_list.append(probs)
        model_list.append(model_name)

        plot_confusion_matrix(labels, preds, class_names=[class1, class2], model_name = model_name, save_dir=save_dir)

    if all_probs_list:
        plot_roc_curve(all_labels, all_probs_list, model_list, save_dir)
        plot_pr_curve(all_labels, all_probs_list, model_list, save_dir)


if __name__ == "__main__":
    main()
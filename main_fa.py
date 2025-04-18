import argparse
import torch
import time
import os
import logging
from data import FADataset
from utils.Trainer import SingleModalTrainer

def setup_training_environment():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--work_type', type=str, default="Contrastive", choices=['Contrastive', 'Graph'])
    parser.add_argument('--model_name', type=str, default='contrastive_model2', help='Name of the model to use.')

    parser.add_argument('--task', type=str, default='NCvsPD', choices=['NCvsPD', 'ProdromalvsPD', 'NCvsProdromal', 'NCvsProdromalvsPD'])

    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--k_folds', type=int, default=5)

    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers.')

    parser.add_argument('--val_start', type=int, default=1, help='Epoch to start validation.')
    parser.add_argument('--val_interval', type=int, default=1, help='How often to perform validation.')

    parser.add_argument('--early_stop_start', type=int, default=30, help='Epoch to start monitoring for early stopping.')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before stopping if no improvement.')

    parser.add_argument('--pick_metric_name', type=str, default='accuracy', choices=['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1', 'precision', 'recall', 'specificity'], help='Metric used for evaluation.')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0, help='Set 1 for debug mode, 0 for normal mode.')
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

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    # logging.info(f"Training with {device}")
    # return args, device, log_file, timestamp
    return args, log_file, timestamp

def main():
    seed = 42
    # args, device, log_file, timestamp = setup_training_environment()
    args, log_file, timestamp = setup_training_environment()
    args.debug = bool(args.debug)
    csv_file = 'data/data.csv'

    if args.work_type == "Contrastive":
        transform = None
        downsample_pd = 125
        dataset = FADataset(csv_file, args, transform=transform, downsample_pd=downsample_pd)
        trainer = SingleModalTrainer(dataset, args, timestamp, seed=seed)
        trainer.start_training(log_file)
    # elif args.work_type == "Graph":
    #     dataset = GraphDataset(csv_file, args)
    #     trainer = GraphTrainer(dataset, args, timestamp, seed=seed)
    #     trainer.start_training()

if __name__ == "__main__":
    main()
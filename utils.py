# utils.py
import os
import time
import logging
import argparse
import torch


def setup_training_environment():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training script for models.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--model_name', type=str, default='Design1', help='Name of the model to use.')
    parser.add_argument('--task', type=str, default='PDvsNC', choices=['PDvsNC', 'PDvsSWEDD', 'NCvsSWEDD'])
    parser.add_argument('--train_bs', type=int, default=16, help='I3D C3D cuda out of memory.')
    parser.add_argument('--val_bs', type=int, default=16, help='densenet cuda out of memory.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of CPU workers.')
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

def rename_log_file(log_file, avg_acc, task, model_name, timestamp):
    log_dir = f'./logs/{task}'
    new_logfilename = os.path.join(log_dir, f'{model_name}_{timestamp}_{avg_acc:.2f}.log')
    os.rename(log_file, new_logfilename)
    return new_logfilename

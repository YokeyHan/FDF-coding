import torch
from argparse import Namespace
from args import get_args
from data_provider.data_factory import load_data
from models.FDF import FDF
from exp.train import ModelTrainer
import os
import random
import numpy as np
def set_seed(seed=0):
    # Python随机模块
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # 如果使用CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # cuDNN相关设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)  # 设置随机种子为42


def count_model_parameters(model):
    return sum(param.numel() for param in model.parameters())

def initialize_device(args):
    if args.use_gpu:
        if not args.use_multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            device = torch.device(f'cuda:{args.gpu}')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    return device

def execute_training(args: Namespace):
    train_loader, val_loader, test_loader, test_dataset = load_data(args)

    # Define model architecture
    model_architectures = {
        'FDF': FDF
    }

    #print(f"Selected Model: {args.model_name}")
    
    model = model_architectures[args.model_name](args)

    if args.verbose:
        print(f"Model parameters count: {count_model_parameters(model)}")

    # Set the device
    if args.model_name == 'D3VAE':
        device = initialize_device(args)
    else:
        device = args.device

    # Initialize the model trainer
    trainer = ModelTrainer(
        args=args,
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset
    )

    # Training loop
    if args.train_flag == 1:
        trainer.train()

    # Evaluate
    metrics = trainer.evaluate_test()
    return metrics

if __name__ == "__main__":
    args = get_args()

    if args.verbose:
        experiment_info = (
            ">" * 15
            + f"{args.task_name}_{args.dataset}_{args.train_settings}"
            + "<" * 15
            + "\n"
        )
        #print(experiment_info)
        #print(f"Experiment Configuration: \n{args}\n")

    # Set torch-specific parameters
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)  # Ensure reproducibility
    torch.cuda.empty_cache()

    # Determine task and start the process
    if args.task_name == "train":
        metrics = execute_training(args)
    else:
        print("Invalid task name specified.")

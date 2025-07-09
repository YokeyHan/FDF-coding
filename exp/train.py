# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from argparse import Namespace
# from pathlib import Path
# from dataclasses import dataclass
# from tqdm import tqdm
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import ExponentialLR
# from utils.tools import EarlyStopping, EpochTimer
# import time
# from thop import profile
# import torch.nn.functional as F
# import random
# import numpy as np
# def set_seed(seed=0):
#     # Python随机模块
#     random.seed(seed)

#     # NumPy
#     np.random.seed(seed)

#     # PyTorch
#     torch.manual_seed(seed)

#     # 如果使用CUDA
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # 多GPU情况

#     # cuDNN相关设置
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# set_seed(0)  # 设置随机种子为42

# @dataclass
# class PerformanceMetrics:
#     mse_loss: float = 0.0
#     mae_loss: float = 0.0
#     loss: float = 0.0
#     kl_loss: float = 0.0

#     def __repr__(self):
#         return f"MSE Loss: {self.mse_loss:.6f}, MAE Loss: {self.mae_loss:.6f}, Loss: {self.loss:.6f}, KL Loss: {self.kl_loss:.6f}"


# class ModelTrainer:
#     def __init__(
#         self,
#         args: Namespace,
#         model: Namespace,
#         device: str,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         test_loader: DataLoader,
#         test_dataset: Namespace
#     ):
#         self.args = args
#         self.verbose = args.verbose
#         self.model = model.to(device)
#         self.device = device
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.test_loader = test_loader
#         self.test_dataset = test_dataset
#         self.kl_weight_start = 0.0001  # 初始KL权重
#         self.kl_weight_end = 1  # 最终KL权重

#         # Loss functions
#         self.mse_criterion = nn.MSELoss()
#         self.mae_criterion = nn.L1Loss()

#         # Optimizer and Scheduler
#         self.optimizer = AdamW(
#             self.model.parameters(),
#             lr=args.learning_rate,
#             weight_decay=args.weight_decay,
#         )
#         self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=args.lr_decay)

#         # Training parameters
#         self.num_epochs = args.num_epochs
#         self.eval_freq = args.eval_frequency

#         # Paths
#         self.save_dir = Path(args.save_dir)
#         self.train_log_path = self.save_dir / "kl0.001_1_train_log.txt"
#         self.val_log_path = self.save_dir / "kl0.001_1_val_log.txt"
#         self.test_log_path = self.save_dir / "kl0.001_1_test_log.txt"
#         self.model_save_path = self.save_dir / "kl0.001_1_model_checkpoint.pth"

#         self.early_stopping = EarlyStopping(self.args)
#         self.epoch_timer = EpochTimer()




#     def train(self):
#         self.train_log_path.write_text("")
#         self.val_log_path.write_text("")



#         for epoch in range(self.num_epochs):
#             # 动态计算当前epoch的KL权重
#             current_kl_weight = self.kl_weight_start + (self.kl_weight_end - self.kl_weight_start) * (
#                         epoch / self.num_epochs)
#             self.epoch_timer.start()

#             train_metrics, train_speed= self._train_one_epoch(current_kl_weight)

#             with self.train_log_path.open("a") as log_file:
#                 log_file.write(f"Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s\n")
#             if self.verbose:
#                 print(f"Training Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s")

#             self.epoch_timer.stop()
#             if self.verbose:
#                 self.epoch_timer.print_duration(epoch=epoch + 1, total_epochs=self.num_epochs)

#             if (epoch + 1) % self.eval_freq == 0:
#                 val_metrics, val_speed = self._validate_one_epoch(current_kl_weight)

#                 with self.val_log_path.open("a") as log_file:
#                     log_file.write(f"Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s\n")
#                 if self.verbose:
#                     print(f"Validation Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s")

#                 self.early_stopping(val_metrics.mse_loss, self.model, self.model_save_path)
#                 if self.early_stopping.early_stop:
#                     if self.verbose:
#                         print("Early stopping triggered.")
#                     break

#     def _train_one_epoch(self, kl_weight):
#         self.model.train()
#         metrics = PerformanceMetrics()
#         total_iters = 0
#         total_time = 0
#         epsilon = 1e-10
#         for x, y in tqdm(self.train_loader, desc="Training", disable=not self.args.use_tqdm):
#             start_time = time.time()
#             self.optimizer.zero_grad()

#             predictions, z_seq, q= self.model(x, task="train")

#             weight = q ** 2 / (q.sum(0) + epsilon)
#             p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)

#             kl_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')
#             mse_loss = self.mse_criterion(predictions, y)
#             mae_loss = self.mae_criterion(predictions, y)
#             loss = mse_loss + kl_weight*kl_loss
#             #mse_loss.backward()
#             loss.backward()
#             self.optimizer.step()
#             flops, params = profile(self.model, inputs=(x, "train"), verbose=False)

#             total_time += time.time() - start_time
#             total_iters += 1
#             iter_speed = total_iters / total_time

#             metrics.mse_loss += mse_loss.item()
#             metrics.mae_loss += mae_loss.item()
#             metrics.loss += loss.item()
#             metrics.kl_loss += kl_loss.item()

#         self.scheduler.step()

#         metrics.mse_loss /= len(self.train_loader)
#         metrics.mae_loss /= len(self.train_loader)
#         metrics.loss /= len(self.train_loader)
#         metrics.kl_loss /= len(self.train_loader)
#         avg_iter_speed = total_iters / total_time

#         return metrics, avg_iter_speed

#     @torch.no_grad()
#     def _validate_one_epoch(self, kl_weight):
#         self.model.eval()
#         metrics = PerformanceMetrics()
#         total_iters = 0
#         total_time = 0
#         epsilon = 1e-10
#         for x, y in tqdm(self.val_loader, desc="Validation", disable=not self.args.use_tqdm):
#             start_time = time.time()

#             predictions, z_seq, q = self.model(x, task="valid")

#             weight = q ** 2 / (q.sum(0)+epsilon)
#             p = weight / (weight.sum(dim=1, keepdim=True)+epsilon)

#             kl_loss = F.kl_div((q+epsilon).log(), p, reduction='batchmean')
#             mse_loss = self.mse_criterion(predictions, y)
#             mae_loss = self.mae_criterion(predictions, y)
#             loss = mse_loss + kl_weight*kl_loss
#             total_time += time.time() - start_time
#             total_iters += 1
#             iter_speed = total_iters / total_time

#             metrics.mse_loss += mse_loss.item()
#             metrics.mae_loss += mae_loss.item()
#             metrics.loss += loss.item()
#             metrics.kl_loss += kl_loss.item()

#         metrics.mse_loss /= len(self.val_loader)
#         metrics.mae_loss /= len(self.val_loader)
#         metrics.loss /= len(self.val_loader)
#         metrics.kl_loss /= len(self.val_loader)
#         avg_iter_speed = total_iters / total_time

#         return metrics, avg_iter_speed

#     # @torch.no_grad()
#     # def evaluate_test(self):
#     #     self.test_log_path.write_text("")
#     #     self.model.load_state_dict(torch.load(self.model_save_path))
#     #     self.model.eval()
#     #     epsilon = 1e-10
#     #     metrics = PerformanceMetrics()
#     #     for x, y in tqdm(self.test_loader, desc="Testing", disable=not self.args.use_tqdm):
#     #         if self.args.loss_type == 'mse':
#     #             predictions , z_seq, q = self.model(x, task="test")

#     #             weight = q ** 2 / (q.sum(0) + epsilon)
#     #             p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)

#     #             kl_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')
#     #             mse_loss = self.mse_criterion(predictions, y)
#     #             mae_loss = self.mae_criterion(predictions, y)
#     #             loss = mse_loss + 10*kl_loss

#     #         metrics.mse_loss += mse_loss.item()
#     #         metrics.mae_loss += mae_loss.item()
#     #         metrics.loss += loss.item()
#     #         metrics.kl_loss += kl_loss.item()

#     #     metrics.mse_loss /= len(self.test_loader)
#     #     metrics.mae_loss /= len(self.test_loader)
#     #     metrics.loss /= len(self.test_loader)
#     #     metrics.kl_loss /= len(self.test_loader)

#     #     with self.test_log_path.open("w") as log_file:
#     #         log_file.write(f"{metrics}\n")

#     #     return metrics


#     @torch.no_grad()
#     def evaluate_test(self):
#         # 清空之前的测试日志
#         self.test_log_path.write_text("")

#         # —— BEGIN: 安全加载模型权重 —— 
#         # 可能 checkpoint 中包含 extra fields（如 total_ops、total_params 等），
#         # 先读取并取出真正的 state_dict
#         raw_ckpt = torch.load(self.model_save_path, map_location=self.device)
#         state = raw_ckpt.get("state_dict", raw_ckpt)

#         # 过滤掉那些不在 model.state_dict() 中的多余 key
#         model_state = self.model.state_dict()
#         filtered = {k: v for k, v in state.items() if k in model_state}

#         # 加载权重，strict=False 会忽略掉缺失或多余的键
#         missing_keys, unexpected_keys = self.model.load_state_dict(filtered, strict=False)
#         if self.verbose:
#             print(">>> Loaded checkpoint for testing")
#             print("    Missing keys:", missing_keys)
#             print("    Unexpected keys (dropped):", unexpected_keys)
#         # —— END: 安全加载模型权重 —— 

#         # 进入评估模式
#         self.model.eval()
#         epsilon = 1e-10
#         metrics = PerformanceMetrics()

#         # 遍历测试集
#         for x, y in tqdm(self.test_loader, desc="Testing", disable=not self.args.use_tqdm):
#             # 前向，得到预测、子空间特征和软聚类分配
#             predictions, z_seq, q = self.model(x, task="test")

#             # 重新计算 P 分布
#             weight = q ** 2 / (q.sum(0) + epsilon)
#             p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)

#             # 计算聚类 KL 损失 + MSE
#             kl_loss  = F.kl_div((q + epsilon).log(), p, reduction='batchmean')
#             mse_loss = self.mse_criterion(predictions, y)
#             mae_loss = self.mae_criterion(predictions, y)
#             # 测试阶段把 KL 损失放大 10 倍
#             loss = mse_loss + 10 * kl_loss

#             # 累加指标
#             metrics.mse_loss += mse_loss.item()
#             metrics.mae_loss += mae_loss.item()
#             metrics.loss     += loss.item()
#             metrics.kl_loss  += kl_loss.item()

#         # 取平均
#         metrics.mse_loss /= len(self.test_loader)
#         metrics.mae_loss /= len(self.test_loader)
#         metrics.loss     /= len(self.test_loader)
#         metrics.kl_loss  /= len(self.test_loader)

#         # 写入测试日志
#         with self.test_log_path.open("w") as log_file:
#             log_file.write(f"{metrics}\n")

#         return metrics



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from utils.tools import EarlyStopping, EpochTimer
import time
from thop import profile
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

@dataclass
class PerformanceMetrics:
    mse_loss: float = 0.0
    mae_loss: float = 0.0
    loss: float = 0.0
   

    def __repr__(self):
        return (f"MSE: {self.mse_loss:.6f}, MAE: {self.mae_loss:.6f}, "
                f"Loss: {self.loss:.6f}")

class ModelTrainer:
    def __init__(
        self,
        args: Namespace,
        model: nn.Module,
        device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        test_dataset: Namespace
    ):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset

        self.kl_weight_start = 0.0001
        self.kl_weight_end = 1.0

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.lr_decay)

        self.num_epochs = args.num_epochs
        self.eval_freq = args.eval_frequency

        self.save_dir = Path(args.save_dir)
        self.train_log_path = self.save_dir / "train_log.txt"
        self.val_log_path   = self.save_dir / "val_log.txt"
        self.test_log_path  = self.save_dir / "test_log.txt"
        self.model_save_path= self.save_dir / "model_checkpoint.pth"

        self.early_stopping = EarlyStopping(args)
        self.epoch_timer = EpochTimer()

    def train(self):
        self.train_log_path.write_text("")
        self.val_log_path.write_text("")

        for epoch in range(self.num_epochs):
            kl_w = (self.kl_weight_start +
                    (self.kl_weight_end - self.kl_weight_start) *
                    (epoch / self.num_epochs))
            start = time.time()
            train_metrics, train_speed = self._train_one_epoch(kl_w)
            duration = time.time() - start

            with self.train_log_path.open("a") as f:
                f.write(f"Epoch {epoch+1}: {train_metrics} | {duration:.1f}s\n")
            if self.verbose:
                print(f"Epoch {epoch+1}: {train_metrics} | {duration:.1f}s")

            if (epoch+1) % self.eval_freq == 0:
                val_metrics, _ = self._validate_one_epoch(kl_w)
                with self.val_log_path.open("a") as f:
                    f.write(f"Epoch {epoch+1}: {val_metrics}\n")
                if self.verbose:
                    print(f"Validation {epoch+1}: {val_metrics}")

                self.early_stopping(val_metrics.mse_loss,
                                    self.model, self.model_save_path)
                if self.early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping.")
                    break

    def _train_one_epoch(self, kl_weight):
        self.model.train()
        metrics = PerformanceMetrics()
        epsilon = 1e-10

        for x, y in tqdm(self.train_loader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # forward 多返回 sep_loss 和 contrastive_loss
            preds, z_seq,  contrastive_loss = self.model(x, task="train")

            # KL loss
            # weight = q ** 2 / (q.sum(0) + epsilon)
            # p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)
            # kl_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')

            mse_loss = self.mse_criterion(preds, y)
            mae_loss= self.mae_criterion(preds, y)

            # 总 loss = MSE + kl_weight * KL + α * sep_loss + β * contrastive_loss
            alpha = 0.1
            beta = 0.1
            loss = mse_loss  + beta * contrastive_loss

            loss.backward()
            self.optimizer.step()

            # accumulate
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()
            metrics.loss     += loss.item()
            # metrics.kl_loss  += kl_loss.item()
            # metrics.sep_loss += sep_loss.item()

        # 平均
        n = len(self.train_loader)
        metrics.mse_loss /= n
        metrics.mae_loss /= n
        metrics.loss     /= n
        # metrics.kl_loss  /= n
        # metrics.sep_loss /= n

        self.scheduler.step()
        return metrics, None

    @torch.no_grad()
    def _validate_one_epoch(self, kl_weight):
        self.model.eval()
        metrics = PerformanceMetrics()
        epsilon = 1e-10

        for x, y in tqdm(self.val_loader, desc="Validate"):
            x, y = x.to(self.device), y.to(self.device)
            preds, z_seq, contrastive_loss = self.model(x, task="valid")

            # weight = q ** 2 / (q.sum(0) + epsilon)
            # p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)
            # kl_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')

            mse_loss = self.mse_criterion(preds, y)
            mae_loss= self.mae_criterion(preds, y)
            alpha=0.1
            beta=0.1
            loss = mse_loss  + beta * contrastive_loss

            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()
            metrics.loss     += loss.item()
            # metrics.kl_loss  += kl_loss.item()
            # metrics.sep_loss += sep_loss.item()

        n = len(self.val_loader)
        metrics.mse_loss /= n
        metrics.mae_loss /= n
        metrics.loss     /= n
        # metrics.kl_loss  /= n
        # metrics.sep_loss /= n

        return metrics, None

    @torch.no_grad()
    def evaluate_test(self):
        # 安全加载 checkpoint（同前示例）
        raw = torch.load(self.model_save_path, map_location=self.device)
        state = raw.get("state_dict", raw)
        model_state = self.model.state_dict()
        filtered = {k: v for k,v in state.items() if k in model_state}
        self.model.load_state_dict(filtered, strict=False)

        self.model.eval()
        metrics = PerformanceMetrics()
        epsilon = 1e-10

        for x, y in tqdm(self.test_loader, desc="Testing"):
            x, y = x.to(self.device), y.to(self.device)
            preds, z_seq, contrastive_loss = self.model(x, task="test")

            # weight = q ** 2 / (q.sum(0) + epsilon)
            # p = weight / (weight.sum(dim=1, keepdim=True) + epsilon)
            # kl_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')

            mse_loss = self.mse_criterion(preds, y)
            mae_loss= self.mae_criterion(preds, y)
            # 在测试阶段通常不加 sep_loss 和 contrastive_loss，但你可按需添加
            beta=0.1
            loss = mse_loss  + beta * contrastive_loss

            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()
            metrics.loss     += loss.item()

            # metrics.mse_loss += mse_loss.item()
            # metrics.mae_loss += mae_loss.item()
            # metrics.loss     += loss.item()
            # metrics.kl_loss  += kl_loss.item()
            # metrics.sep_loss += sep_loss.item()

        n = len(self.test_loader)
        metrics.mse_loss /= n
        metrics.mae_loss /= n
        # metrics.loss     /= n
        # metrics.kl_loss  /= n
        # metrics.sep_loss /= n

        with self.test_log_path.open("w") as f:
            f.write(f"{metrics}\n")

        return metrics
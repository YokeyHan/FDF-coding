# import torch
# import torch.nn as nn
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.api import SARIMAX
# from argparse import Namespace
# from models.generate_adj import *
# from models.fdf_backbone import (
#     Diffusion,
#     series_decomposition,
#     ST_decomposition,
#     MultiLinearModel,
#     ARIMAModel,
#     ARIMAXModel
# )
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


# class FDF(nn.Module):
#     def __init__(self, args: Namespace):
#         super(FDF, self).__init__()

#         #self.decom = series_decomposition(kernel_size = 5)channels,c_hidden_dim,z_dim,num_clusters

#         self.input_len = args.input_len
#         self.device = args.device
#         self.pred_len = args.pred_len
#         self.time_steps = args.time_steps
#         self.num_clusters = args.num_clusters
#         self.adj = get_similarity_metrla(thr=0.1)
#         self.support = compute_support_gwn(self.adj, device=self.device)
#         self.is_adp = True
#         if self.is_adp:
#             node_num = self.adj.shape[0]
#             self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
#             self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
#             self.support.append([self.nodevec1, self.nodevec2])

#         self.decom = ST_decomposition(channels=args.ST_channels, z_dim=args.z_dim, num_clusters=args.num_clusters,
#                                       adj=self.support, device=self.device)
#         self.diffusion = Diffusion(
#             time_steps=args.time_steps,
#             feature_dim=args.feature_dim,
#             seq_len=args.input_len,
#             pred_len=args.pred_len,
#             MLP_hidden_dim=args.MLP_hidden_dim,
#             emb_dim=args.z_dim,    #args.z_dim  !!!!
#             adj=self.support,
#             channels=args.ST_channels,
#             z_dim=args.z_dim,
#             device=self.device,
#             beta_scheduler=args.scheduler,

#             patch_size=args.patch_size
#         )
#         self.eta = 0
        
#         self.seq_len = args.input_len 
#         #self.trend_linear = MultiLinearModel(seq_len = args.input_len, pred_len = args.pred_len)
#         #self.trend_linear =  ARIMAModel(seq_len=args.input_len, pred_len=args.pred_len)
#         self.trend_linear =  ARIMAXModel(seq_len=args.input_len, pred_len=args.pred_len,ar_order=1, diff_order=1,
#                    ma_order=1, exog_dim=1)
#         # 定义投影层
#         self.projection = nn.Linear(args.z_dim, 1)  # 将z_dim映射到1维
#     # 训练
#     def pred(self, x):
#         batch_size, input_len, num_features = x.size()
        
#         x_seq = x[:, :self.seq_len, :]
#         x_means = x_seq.mean(1, keepdim=True).detach()
#         x_enc = x_seq - x_means
#         x_stdev = torch.sqrt(
#             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
#         x_norm = x - x_means
#         x_norm /= x_stdev
#         #[B,L,C]
#         x_seq_input = x_norm[:, :self.seq_len, :]       #[B,seq_len,N]
#         season_seq, trend_seq, cluster_ids_seq, z_seq, logits_seq, q_seq = self.decom(x_seq_input)     #[B,seq_len,N]    # season [B, L, N, z_dim]; trend ->[B,L,num_clusters*self.z_dim]; id [N,]
#         B, L, N, z_dim = season_seq.size()
#         x_pred = x_norm[:, -self.pred_len:, :]         #[B,Pre_len,N]
#         season_pred, trend_pred, cluster_ids_pred, z_pred, logits_pred, q_pred= self.decom(x_pred)      #[B,Pre_len,N]

#         base_shape = season_pred.shape
#         trend_pred = self.trend_linear(trend_seq)       #[B,Pre_len,N]   [B,L,num_clusters*self.z_dim]
        
#         # Noising Diffusion
#         t = torch.randint(0, self.time_steps, (batch_size,), device=self.device)
#         noise_season = self.diffusion(season_pred.reshape(B,-1,N*z_dim),t)   # season [B, L, N, z_dim]-->[B,L,N*z*dim];  [B,L,N]
#         season_pred = self.diffusion.pred(noise_season, t, season_seq.reshape(B,L,-1))  #[B,Pre_len,N]  [B,L,N*z_dim]

#         #[B, L, num_clusters * self.z_dim]-->[num_clusters,B,L,self.z_dim]
#         # 选择对应中心
#         # 扩展聚类标签 [N,] -> [B, L, N]
#         trend_pred = trend_pred.reshape(B, self.pred_len, -1, z_dim)
#         # 假设所有batch和timestep使用相同的空间聚类结构
#         cluster_ids = cluster_ids_pred.view(1, 1, N).expand(B, self.pred_len, -1)
#         # 为每个时空点选择对应的聚类中心 [B, L, N, z_dim]
#         selected_centers = torch.gather(
#             trend_pred,
#             dim=2,  # 在num_clusters维度上gather
#             index=cluster_ids.unsqueeze(-1).expand(-1, -1, -1, z_dim)
#         )

#         # 调整selected_centers形状 [B, L, N, z_dim] -> [B, L, N*z_dim]
#         selected_centers = selected_centers.reshape(B, self.pred_len, N * z_dim)

#         # 5将残差加回聚类中心
#         reconstructed = season_pred + selected_centers

#         predict_x = self.projection(reconstructed.reshape(B,self.pred_len,N,z_dim)).squeeze(-1)  #[B, L, N*z_dim]-->[B, L, N]
#         #predict_x = trend_pred + season_pred

#         dec_out = predict_x * \
#                   (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         dec_out = dec_out + \
#                   (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
#         return dec_out, z_seq, q_seq

#     # 验证或测试
#     def forecast(self, input_x):
#         x = input_x[:, :self.seq_len, :]   #[B, seq_len, N]
#         #b, _, dim = x.shape
#         x_means = x.mean(1, keepdim=True).detach()
#         x_enc = x - x_means
#         x_stdev = torch.sqrt(
#             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= x_stdev
#         #
#         season, trend, cluster_ids_seq, z_seq, logits_seq, q_seq= self.decom(x_enc)    # season [B, L, N, z_dim]; trend ->[B,L,num_clusters*self.z_dim]; id [N,]
#         B, L, N, z_dim = season.size()
#         trend_pred_part = trend
#         trend_pred = self.trend_linear(trend_pred_part)
#         shape = torch.zeros((B, self.pred_len, N*z_dim), dtype=torch.int, device=self.device)  # [B, pred_len, N]
#         #predict_x = self.projection(reconstructed.reshape(B, L, N, z_dim)).squeeze(-1)  # [B, L, N*z_dim]-->[B, L, N]
#         season_pred = self.diffusion.sample_infill(shape, self.time_steps, season.reshape(B,L,-1))  #[B, pred_len, N]  #[B,L,N*dim]
#         # 扩展聚类标签 [N,] -> [B, L, N]
#         trend_pred = trend_pred.reshape(B, self.pred_len, -1, z_dim)
#         # 假设所有batch和timestep使用相同的空间聚类结构
#         cluster_ids = cluster_ids_seq.view(1, 1, N).expand(B, self.pred_len, -1)
#         # 为每个时空点选择对应的聚类中心 [B, L, N, z_dim]
#         selected_centers = torch.gather(
#            trend_pred,
#            dim=2,  # 在num_clusters维度上gather
#            index=cluster_ids.unsqueeze(-1).expand(-1, -1, -1, z_dim)
#         )
#         # 调整selected_centers形状 [B, L, N, z_dim] -> [B, L, N*z_dim]
#         selected_centers = selected_centers.reshape(B, self.pred_len, N * z_dim)
#         # 将残差加回聚类中心
#         reconstructed = season_pred + selected_centers
#         predict_x = self.projection(reconstructed.reshape(B, self.pred_len, N, z_dim)).squeeze(-1)  # [B, L, N*z_dim]-->[B, L, N]

#         #predict_x = trend_pred + season_pred
#         dec_out = predict_x * \
#                   (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         dec_out = dec_out + \
#                   (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
#         #x_pred = input_x[:, -self.pred_len:, :]
#         #season_pred_part, trend_pred_part, cluster_id= self.decom(x_pred)
#         return dec_out, z_seq, q_seq

#     def forward(self, x, task):
#         if task == "train":
#             return self.pred(x)  
#         elif task == 'valid' or task == "test":
#             return self.forecast(x)  
#         else:
#             raise ValueError(f"Invalid task: {task=}")



import torch
import torch.nn as nn
from argparse import Namespace

from models.generate_adj import get_similarity_metrla, compute_support_gwn,load_pickle
from models.fdf_backbone2 import Diffusion, ST_decomposition, ARIMAXModel
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


class FDF(nn.Module):
    def __init__(self, args: Namespace):
        super(FDF, self).__init__()
        self.input_len    = args.input_len
        self.pred_len     = args.pred_len
        self.time_steps   = args.time_steps
        self.device       = args.device
        self.z_dim        = args.z_dim
        self.num_clusters = args.num_clusters

        _,_,adj_mat=load_pickle('/home/user/plh/Prediction/myFDF-v5/FDF-main/datasets/METR-LA/adj_mx_metr_la.pkl')
        print(adj_mat.shape)
        # 构建图结构
        # adj_mat = get_similarity_metrla(thr=0.1)
        support = compute_support_gwn(adj_mat, device=self.device)
        node_num = adj_mat.shape[0]
        nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True).to(self.device)
        nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True).to(self.device)
        support.append([nodevec1, nodevec2])
        self.support = support

        # ST 分解（带 Sinkhorn 软聚类 + margin loss + 对比学习损失）
        self.decom = ST_decomposition(
            channels     = args.ST_channels,
            z_dim        = args.z_dim,
            num_clusters = args.num_clusters,
            adj          = self.support[0],
            device       = self.device,
            batch_size   = args.batch_size,
            tau          = args.cluster_tau,
            sep_margin   = args.sep_margin,
            contrastive_loss_weight = args.contrastive_loss_weight , # 新增参数
            input_len=args.input_len
        ).to(self.device)

        # 扩散模块
        self.diffusion = Diffusion(
            time_steps     = args.time_steps,
            feature_dim    = args.feature_dim,
            seq_len        = args.input_len,
            pred_len       = args.pred_len,
            MLP_hidden_dim = args.MLP_hidden_dim,
            emb_dim        = args.z_dim,
            adj            = self.support[0],
            channels       = args.ST_channels,
            z_dim          = args.z_dim,
            device         = self.device,
            beta_scheduler = args.scheduler,
            patch_size     = args.patch_size
        )

        # 趋势预测（ARIMAX）
        self.trend_linear = ARIMAXModel(
            seq_len    = args.input_len,
            pred_len   = args.pred_len,
            ar_order   = 1,
            diff_order = 1,
            ma_order   = 1,
            exog_dim   = 1
        )

        # 最终投影
        self.projection = nn.Linear(args.z_dim, 1)

    def pred(self, x):
        """
        x: [B, input_len + pred_len, N]
        return:
          final_output: [B, pred_len, N]
          z_seq:        [B, input_len, N, z_dim]
          q_seq:        [B*N*input_len, num_clusters]
          sep_loss:     scalar
          contrastive_loss: scalar
        """
        B, total_len, N = x.size()

        # 1) 标准化
        x_seq = x[:, :self.input_len, :]                # [B, input_len, N]
        x_mean = x_seq.mean(1, keepdim=True)             # [B, 1, N]
        x_enc  = x_seq - x_mean                          # [B, input_len, N]
        x_std  = torch.sqrt(x_enc.var(1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,N]
        x_norm = (x - x_mean) / x_std                    # [B, total_len, N]

        # 2) ST 分解 —— 得到 season、centers、q、z、sep_loss、contrastive_loss
        season_seq, _,  z_seq, contrastive_loss = \
            self.decom(x_norm[:, :self.input_len, :])   # [B,L,N,C], 6-outputs
        season_pred, _,  _ ,  _contrastive_loss2 = \
            self.decom(x_norm[:, -self.pred_len:, :])  # [B,pre,N,C], 6-outputs

        # 3) 趋势预测：对每个节点分别做 ARIMAX
        z_flat = z_seq.reshape(B*N, self.input_len, self.z_dim)
        z_flat_1d = z_flat.mean(dim=-1, keepdim=True)      # [B*N, L, 1]
        trend_flat = self.trend_linear(z_flat_1d)           # [B*N, pred_len, 1]
        trend_pred = trend_flat.reshape(B, N, self.pred_len).permute(0,2,1)  # [B, pred_len, N]

        # 4) Diffusion 噪声 & 去噪
        t = torch.randint(0, self.time_steps, (B,), device=self.device)
        season_flat = season_pred.reshape(B, self.pred_len, -1)  # [B, pred_len, N*C]
        noise = self.diffusion(season_flat, t)                   # [B, pred_len, N*C]
        denoised = self.diffusion.pred(noise, t, season_flat)    # [B, pred_len, N*C]

        # 5) 聚类中心重构 & 合并
        denoised4 = denoised.view(B, self.pred_len, N, self.z_dim)  # [B, pred_len, N, C]
        trend4    = trend_pred.unsqueeze(-1)                        # [B, pred_len, N, 1]
        recon     = denoised4 + trend4                              # [B, pred_len, N, C]

        # 6) 投影 & 反标准化
        proj    = self.projection(recon).squeeze(-1)                 # [B, pred_len, N]
        dec_out = proj * x_std[:, :, :] + x_mean[:, :, :]           # [B, pred_len, N]

        final_output = dec_out

        c_loss=contrastive_loss + _contrastive_loss2
        return final_output, z_seq, c_loss

    def forward(self, x, task):
        return self.pred(x)
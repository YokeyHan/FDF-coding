import math
import torch
import torch.nn as nn
from models.fdf_denoise_network import fdf_denoise_network
import torch.nn.functional as F
from functools import partial
from models.layers import *
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
class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomposition(nn.Module):
    
    def __init__(self, kernel_size):
        super(series_decomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ST_decomposition00(nn.Module):

    def __init__(self,channels, z_dim, num_clusters, adj, device,nheads=4, is_cross_t=False, is_cross_s=False, inputdim=1, v=1, init_method='kmeans++'):
        """
                时空分解模型，显式使用聚类中心计算残差

                Args:
                    input_dim: 输入特征维度
                    c_hidden_dim: 隐藏层维度
                    z_dim: 子空间维度（也是聚类中心维度）
                    num_clusters: 聚类中心数量
                """
        super(ST_decomposition, self).__init__()
        self.adj = adj
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.channels = channels
        self.device = device
        #self.moving_avg = moving_avg(kernel_size, stride=1)
        self.input_projection = Conv1d_with_init(inputdim, channels, 1)
        self.pre_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)     #[B,seq_len,N]
        self.pre_feature = FeatureLearning0(channels=channels, nheads=nheads, is_cross=is_cross_s)
        #self.pre_feature = FeatureLearning(channels=channels, nheads=nheads, target_dim=36,
        #                                       order=2, include_self=True, device=device, is_adp=True,
        #                                       adj_file=adj, proj_t=64,is_cross=is_cross_s)



    def forward(self, x):   #[B,seq_len,N]--->[B,seq_len,N]
        #Spatiotemporal encoding
        B, L, N = x.shape
        x = self.input_projection(x.unsqueeze(dim=-1).permute(0, 3, 2,1).reshape(B,1,N*L)).reshape(B,self.channels,N,L) #[B, L, N, 1] ->[B,1,L,N]->[B,channels,L*N]
        base_shape = x.shape
        pre_time = self.pre_time(x,base_shape)   #[B,channel,N * L]
        pre_feature = self.pre_feature(pre_time,base_shape, self.adj)   #[B,channel,N * L]

        ###### 聚类
       

        return residuals, centers, z, q


class ST_decomposition(nn.Module):

    def __init__(self,channels, z_dim, num_clusters, adj, device,nheads=4, is_cross_t=False, is_cross_s=False, inputdim=1, v=1, init_method='kmeans++'):
        """
                时空分解模型，显式使用聚类中心计算残差

                Args:
                    input_dim: 输入特征维度
                    c_hidden_dim: 隐藏层维度
                    z_dim: 子空间维度（也是聚类中心维度）
                    num_clusters: 聚类中心数量
                """
        super(ST_decomposition, self).__init__()
        self.adj = adj
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.channels = channels
        self.device = device
        #self.moving_avg = moving_avg(kernel_size, stride=1)
        self.input_projection = Conv1d_with_init(inputdim, channels, 1)
        self.pre_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)     #[B,seq_len,N]
        self.pre_feature = FeatureLearning0(channels=channels, nheads=nheads, is_cross=is_cross_s)
        #self.pre_feature = FeatureLearning(channels=channels, nheads=nheads, target_dim=36,
        #                                       order=2, include_self=True, device=device, is_adp=True,
        #                                       adj_file=adj, proj_t=64,is_cross=is_cross_s)
        # Initialize cluster centers
        self.cluster_centers = None

        self.subspace = nn.Sequential(
            nn.Linear(channels, z_dim)
        )
        self.cluster = nn.Sequential(nn.Linear(channels, z_dim)
        )
        # 显式定义聚类中心 [n_clusters, z_dim]
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, z_dim))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
        # degree
        self.v = v
        # 标记聚类中心是否已初始化
        self.centers_initialized = False
        self.init_method = init_method

    def initialize_cluster_centers(self, data):
        """使用K-means++方法初始化聚类中心"""
        if not self.centers_initialized:
            # 准备用于初始化的数据
            B, L, N = data.shape
            x = self.input_projection(data.unsqueeze(dim=-1).permute(0, 3, 2, 1).reshape(B, 1, N * L)).reshape(B,
                                                                                                               self.channels,
                                                                                                               N, L)
            base_shape = x.shape
            pre_time = self.pre_time(x, base_shape)
            pre_feature = self.pre_feature(pre_time, base_shape, self.adj)
            features = pre_feature.permute(0, 2, 1).reshape(-1, self.channels)

            # 应用子空间投影
            Z = self.subspace(features)  # [B*N*L, z_dim]
            Z = F.normalize(Z, p=2, dim=1)  # L2归一化

            # 使用K-means++初始化聚类中心
            init_centers = kmeans_plus_plus_init(Z, self.num_clusters, self.device)
 
            # 赋值给模型参数
            self.cluster_centers.data.copy_(init_centers)
            self.centers_initialized = True
            print("聚类中心初始化完成")

    def forward(self, x):   #[B,seq_len,N]--->[B,seq_len,N]
        # 如果聚类中心尚未初始化，先进行初始化
        if not self.centers_initialized and self.training:
            self.initialize_cluster_centers(x)

        B, L, N = x.shape
        x = self.input_projection(x.unsqueeze(dim=-1).permute(0, 3, 2,1).reshape(B,1,N*L)).reshape(B,self.channels,N,L) #[B, L, N, 1] ->[B,1,L,N]->[B,channels,L*N]
        base_shape = x.shape
        pre_time = self.pre_time(x,base_shape)   #[B,channel,N * L]
        pre_feature = self.pre_feature(pre_time,base_shape, self.adj)   #[B,channel,N * L]
        # Reshape features for clustering [B*N*L, channels]
        features = pre_feature.permute(0, 2, 1).reshape(-1, self.channels)
        ###### 聚类
        # 3. 子空间投影和归一化
        Z = self.subspace(features)  # [B*N*L, z_dim]
        Z = F.normalize(Z, p=2, dim=1)  # L2归一化
        #[B,N,L,self.z_dim]-->[N,B,L,self.z_dim]-->[N,B*L*self.z_dim]
        # 4. 计算聚类分配（基于聚类中心距离）[N,B*L*self.z_dim]  [num_clusters,B*L*self.z_dim]
        logits = self.cluster(features).float()
        logits = F.normalize(logits, p=2, dim=1)

        # 3. 动态扩展聚类中心到当前序列长度 L [num_clusters, B, L, z_dim]
        centers = self.cluster_centers.unsqueeze(1).unsqueeze(1)  # [num_clusters, 1, 1, z_dim]
        centers = centers.expand(-1, B, L, -1)  # [num_clusters, B, L, z_dim]
        distances = torch.cdist(Z.reshape(B,N,L,-1).permute(1, 0, 2, 3).reshape(N,B*L*self.z_dim), centers.reshape(self.num_clusters,-1))
        cluster_ids = torch.argmin(distances, dim=1)  #[B,N]  [B*L*z_dim]

        # 5. 计算残差：样本 - 所属聚类中心  [B*N, L*z_dim] [num_clusters, B*L*z_dim]
        residuals = Z.reshape(B,N,L,self.z_dim).permute(1, 0, 2, 3).reshape(N,B*L*self.z_dim) - centers[cluster_ids].reshape(N,B*L*self.z_dim)  # [N, z_dim]
        # 6. 重塑残差为原始时空形状 [B, L, N, z_dim]    [B*N, L*z_dim]
        residuals = residuals.view(N, B, L, -1).permute(1, 2, 0, 3)  # [B, L, N, z_dim]

        centers = centers.reshape(-1, B, L, self.z_dim).permute(1, 2, 0, 3).reshape(B, L,-1)  # [cluster_num,L,z_dim]
        z = Z.reshape(B, N,L,self.z_dim)
        logit = logits.reshape(B, N,L,self.z_dim)

        diff = residuals  # [1, num_clusters, 1, z_dim]  # [B, num_clusters, N, L, z_dim]
        distance = torch.sum(diff.pow(2), dim=-1)  # [B, num_clusters, N, L]
        # 添加小的epsilon值提高数值稳定性
        epsilon = 1e-10
        q = 1.0 / (1.0 + (distance + epsilon) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + epsilon) # 归一化
        # 为了提高稳定性，在训练期间添加小的随机噪声

        noise = torch.rand_like(q) * 0.01
        q = (q + noise) / (q + noise).sum(dim=1, keepdim=True)

        return residuals, centers, cluster_ids, z, logit, q


    def get_cluster_assignments(self, x):
        """获取样本的聚类分配"""
        with torch.no_grad():
            pre_feature = self.pre_feature(x)
            logits = self.cluster(pre_feature)
            return torch.argmax(logits, dim=1)



class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        feature_dim : int,
        seq_len : int,
        pred_len : int,
        MLP_hidden_dim : int,
        emb_dim : int,
        patch_size : int,
        adj : None,
        channels: int,
        z_dim: int,
        device: torch.device,
        beta_scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps
        self.seq_length = seq_len
        self.pred_length = pred_len
        self.adj = adj

        if beta_scheduler == 'cosine':
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif beta_scheduler == 'linear':
            self.betas = self._linear_beta_schedule().to(self.device)
        elif beta_scheduler == 'exponential':
            self.betas = self._exponential_beta_schedule().to(self.device)
        elif beta_scheduler == 'inverse_sqrt':
            self.betas = self._inverse_sqrt_beta_schedule().to(self.device)
        elif beta_scheduler == 'piecewise':
            self.betas = self._piecewise_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Unknown schedule type: {scheduler}")
        
        self.eta = 0
        self.alpha = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

        self.denoise_net = fdf_denoise_network(feature_dim, seq_len, pred_len, device, self.adj, channels,
                                               z_dim, MLP_hidden_dim, emb_dim, patch_size)
    
    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alpha_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _exponential_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        steps = self.time_steps
        return beta_start * ((beta_end / beta_start) ** (torch.linspace(0, 1, steps)))

    def _inverse_sqrt_beta_schedule(self, beta_start=1e-4):
        steps = self.time_steps
        x = torch.arange(1, steps + 1)
        return torch.clip(beta_start / torch.sqrt(x), 0, 0.999)

    def _piecewise_beta_schedule(self, beta_values=[1e-4, 0.01, 0.02], segment_steps=[100, 200, 300]):
        assert len(beta_values) == len(segment_steps), "beta_values and segment_steps length mismatch"
        betas = [torch.full((steps,), beta) for beta, steps in zip(beta_values, segment_steps)]
        return torch.cat(betas)[:self.time_steps]
    
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)  
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1).unsqueeze(-1)  
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x, t):
        noisy_x, _ = self.noise(x, t)
        return noisy_x
    
    def pred(self, x, t, cond):
        if t == None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        return self.denoise_net(x, t, cond)
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t, cond, clip_x_start=False, padding_masks=None):

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.denoise_net(x, t, cond)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
        
    @torch.no_grad()
    def sample_infill(self, shape, sampling_timesteps, cond, clip_denoised=True):
        batch_size, _, _ = shape.shape     ##[B, pred_len, N]     #[B,L,N*dim]
        batch, device, total_timesteps, eta = shape[0], self.device, self.time_steps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        shape = shape #torch.zeros((batch_size, shape[1], cond.shape[2]), dtype=torch.int, device=self.device)
           #[B,pre_len,N*z_dim]
        denoise_series = torch.randn(shape.shape, device=device)   #[B, pred_len, N]  [B,L,N*dim]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(denoise_series, time_cond, cond, clip_x_start=clip_denoised)

            if time_next < 0:
                denoise_series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(denoise_series)

            denoise_series = pred_mean + sigma * noise

        return denoise_series

class MultiLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_loops=2):
        super(MultiLinearModel, self).__init__()
        # 定义线性投影层：将seq_len长度的序列映射为pred_len长度
        # 输入形状：[batch, num_loops, feature_dim, seq_len]
        # 输出形状：[batch, num_loops, feature_dim, pred_len]
        self.linear_projection = nn.Linear(seq_len, pred_len, bias=True)
        # 定义加权线性层：将num_loops个变换结果融合为1个输出
        # 输入形状：[batch, pred_len, feature_dim, num_loops]
        # 输出形状：[batch, pred_len, feature_dim, 1]
        self.weighted_linear = nn.Linear(num_loops, 1, bias=True)
        # 保存变换次数（至少包含原始数据和1次变换）
        self.num_loops = num_loops

    def forward(self, input_data):
        # 初始化变换数据列表，加入原始数据（添加最后一个维度）
        # transformed_data[0]形状：[batch, seq_len, feature_dim, 1]
        transformed_data = [input_data.unsqueeze(-1)]
        # 生成多阶非线性变换数据
        for i in range(2, self.num_loops + 1):
            # 克隆输入数据避免修改原始值
            transformed = input_data.clone()

            # 对第2个特征维度（下标1）进行非线性变换：
            # sign(x) * |x|^(1/i)  （i=2时为平方根变换，i=3为立方根变换等）
            transformed[:, 1, :] = torch.sign(input_data[:, 1, :]) * (torch.abs(input_data[:, 1, :]) ** (1 / i))
            # 将变换结果加入列表（添加最后一个维度）
            transformed_data.append(transformed.unsqueeze(-1))
        # 沿最后一个维度拼接所有变换结果
        # 输出形状：[batch, seq_len, feature_dim, num_loops]
        concatenated_data = torch.cat(transformed_data, dim=-1)
        # 序列长度投影：
        # 1. 先置换维度为 [batch, num_loops, feature_dim, seq_len]
        # 2. 线性投影到 [batch, num_loops, feature_dim, pred_len]
        # 3. 置换回 [batch, pred_len, feature_dim, num_loops]
        sequence_output = self.linear_projection(concatenated_data.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # 多视角加权融合：
        # 将num_loops个变换结果加权合并为1个输出
        # 输出形状：[batch, pred_len, feature_dim]
        output = self.weighted_linear(sequence_output).squeeze(-1)
        
        return output



class ARIMAXModel(nn.Module):
    def __init__(self, seq_len, pred_len, ar_order=1, diff_order=1, ma_order=1, exog_dim=0):

        """
                ARIMAX模型（带外部特征的ARIMA）

                Args:
                    seq_len: 输入序列长度
                    pred_len: 预测序列长度
                    ar_order: 自回归阶数 (p)
                    diff_order: 差分阶数 (d)
                    ma_order: 移动平均阶数 (q)
                    exog_dim: 外部特征维度 (0表示无外部特征)
                """
        super(ARIMAXModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ar_order = ar_order
        self.diff_order = diff_order
        self.ma_order = ma_order
        self.exog_dim = exog_dim
        # ARIMA核心组件
        self.ar_weights = nn.Parameter(torch.randn(ar_order)* 0.1)
        self.ma_weights = nn.Parameter(torch.randn(ma_order)* 0.1)

        # 外部特征处理层
        if exog_dim > 0:
            self.exog_projection = nn.Sequential(
                nn.Linear(exog_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1))

        # 差分后序列长度
        self.diff_seq_len = seq_len - diff_order

        # 最终投影层（处理组合特征）
        self.linear_projection = nn.Linear(self.diff_seq_len, pred_len)

        # 残差缓冲区（用于MA计算）
        self.register_buffer('residuals', torch.zeros(ma_order, dtype=torch.float32))

    def diff(self, x, order):
        """
        Differencing operation
        """
        for _ in range(order):
            x = x[:, 1:] - x[:, :-1]
        return x

    def forward(self, x, exog=None):
        """
                输入：
                    x: 主序列 [batch, seq_len, 1]
                    exog: 外部特征 [batch, seq_len, exog_dim] (可选)
                输出：
                    [batch, pred_len, 1]
                """

        batch_size, seq_len, feature_dim = x.shape
        # 差分处理
        x_diff = self.diff(x, self.diff_order) if self.diff_order > 0 else x

        # 自回归项（修复切片错误）
        ar_terms = torch.zeros_like(x_diff)
        for i in range(self.ar_order):
            slice_end = -i if i > 0 else None
            ar_terms[:, i:] += self.ar_weights[i] * x_diff[:, :slice_end]

            # 移动平均项（使用历史残差）
        ma_terms = torch.zeros_like(x_diff)
        if self.ma_order > 0:
            current_residual = (x_diff - ar_terms)[:, -1, 0]  # 最新残差
            self._update_residuals(current_residual)
            for i in range(self.ma_order):
                ma_terms[:, i:] += self.ma_weights[i] * self.residuals[i]



        # 外部特征处理
        #exog_effect = 0
        if self.exog_dim > 0 and exog is not None:
            exog_effect = self.exog_projection(exog).squeeze(-1)
            exog_effect = exog_effect[:, self.diff_order:].unsqueeze(-1)  # 对齐差分后长度
        else:
            exog_effect = torch.zeros_like(x_diff)  # 保持张量形式
        # 组合所有成分
        combined = x_diff + ar_terms + ma_terms + exog_effect

        # 维度调整并投影
        output = self.linear_projection(combined.permute(0, 2, 1))
        return output.permute(0, 2, 1)

    def _update_residuals(self, batch_residuals):
        """处理batch维度的残差更新"""
        if self.ma_order == 0:
            return

        # 方法1：取batch平均
        mean_residual = batch_residuals.mean().detach()

        # 方法2：取最后一个样本（任选一种）
        # mean_residual = batch_residuals[-1].detach()

        # 安全更新
        with torch.no_grad():
            new_buffer = torch.roll(self.residuals, 1)
            new_buffer[0] = mean_residual
            self.residuals.copy_(new_buffer)

    def reset_residuals(self):
        """重置残差缓冲区"""
        if self.ma_order > 0:
            self.residuals.zero_()

class ARIMAModel(nn.Module):
    def __init__(self, seq_len, pred_len, ar_order=1, diff_order=1, ma_order=1):
        """
        ARIMA-like model implemented in PyTorch

        Args:
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
            ar_order: Autoregressive order (p)
            diff_order: Differencing order (d)
            ma_order: Moving average order (q)
        """
        super(ARIMAModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ar_order = ar_order
        self.diff_order = diff_order
        self.ma_order = ma_order

        # Autoregressive component
        self.ar_weights = nn.Parameter(torch.randn(ar_order))

        # Moving average component
        self.ma_weights = nn.Parameter(torch.randn(ma_order))

        # Differencing is implemented as a fixed operation
        # Final projection to prediction length
        self.linear_projection = nn.Linear(seq_len - diff_order, pred_len)

        # 修改初始化部分
        self.ar_weights = nn.Parameter(torch.randn(ar_order) * 0.1)
        # 小随机值
        self.ma_weights = nn.Parameter(torch.randn(ma_order) * 0.1)
        # 小随机值

        # 删除 xavier 初始化代码（不需要）
        # nn.init.xavier_uniform_(self.ar_weights)  # 删除
        # nn.init.xavier_uniform_(self.ma_weights)  # 删除

    def diff(self, x, order):
        """
        Differencing operation
        """
        for _ in range(order):
            x = x[:, 1:] - x[:, :-1]
        return x

    def forward(self, x):
        """
        x shape: [batch, seq_len, feature_dim]
        We'll operate on each feature dimension independently
        """
        batch_size, seq_len, feature_dim = x.shape
        # 差分处理
        x_diff = self.diff(x, self.diff_order) if self.diff_order > 0 else x

        # 自回归项（修复切片错误）
        ar_terms = torch.zeros_like(x_diff)
        for i in range(self.ar_order):
            # 安全切片方式
            start_idx = 0
            end_idx = -i if i > 0 else None
            ar_terms[:, i:] += self.ar_weights[i] * x_diff[:, start_idx:end_idx]

        # 移动平均项（同样修复）
        ma_terms = torch.zeros_like(x_diff)
        for i in range(self.ma_order):
            start_idx = 0
            end_idx = -i if i > 0 else None
            ma_terms[:, i:] += self.ma_weights[i] * x_diff[:, start_idx:end_idx]

        # 组合并投影
        combined = x_diff + ar_terms + ma_terms
        output = self.linear_projection(combined.permute(0, 2, 1))

        output = output.permute(0, 2, 1)
        return output

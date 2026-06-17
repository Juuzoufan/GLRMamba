"""
MambaFormer ST-Mamba-Res: 残差学习版时空混合模型

关键升级：
1. 归一化残差连接 - 模型只预测增量，不再死记硬背基数
2. 静态图偏置注意力 - 可学习的邻接矩阵，防止过拟合
3. 独立解码头 - 12个时间步有独立的预测参数
4. RevIN - 可逆实例归一化，解决非平稳性问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from einops import rearrange
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using fallback")


# ============================================================================
# 1. RevIN (保持不变)
# ============================================================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_dim=None):
        if self.affine:
            w = self.affine_weight[:target_dim] if target_dim else self.affine_weight
            b = self.affine_bias[:target_dim] if target_dim else self.affine_bias
            x = x - b
            x = x / (w + self.eps * self.eps)
        mean = self.mean[..., :target_dim] if target_dim else self.mean
        std = self.stdev[..., :target_dim] if target_dim else self.stdev
        x = x * std
        x = x + mean
        return x

    def forward(self, x, mode: str, target_dim=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_dim)
        return x

# ============================================================================
# 2. 基础模块 (Mamba & Spatial Bias Attention)
# ============================================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.conv = nn.Conv1d(d_model, d_model * 2, kernel_size=d_conv, padding=d_conv-1, groups=d_model)
            self.proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            x_t = rearrange(x, 'b l d -> b d l')
            x_t = self.conv(x_t)[..., :x.shape[1]]
            gate, val = x_t.chunk(2, dim=1)
            x = self.proj(rearrange(F.silu(gate) * val, 'b d l -> b l d'))
        return residual + self.dropout(x)

class SpatialAttentionWithBias(nn.Module):
    """[升级] 带门控静态图偏置的空间注意力"""
    def __init__(self, d_model, n_heads, num_nodes, dropout=0.1, use_graph_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.use_graph_bias = use_graph_bias
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        if self.use_graph_bias:
            # [核心] 可学习的静态图偏置 (Adjacency Bias)
            self.graph_bias = nn.Parameter(torch.zeros(1, n_heads, num_nodes, num_nodes))
            nn.init.xavier_uniform_(self.graph_bias)
            # [新增] 门控系数：从很小开始 sigmoid(-3)≈0.047，减少过拟合
            self.graph_gate = nn.Parameter(torch.tensor(-3.0))
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x).view(B, N, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, -1).transpose(1, 2)
        
        # 动态注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # [消融开关] 门控静态图偏置
        if self.use_graph_bias:
            gate = torch.sigmoid(self.graph_gate)
            attn = attn + gate * self.graph_bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.o_proj(out)
        return residual + self.dropout(out)


class TemporalSelfAttentionBlock(nn.Module):
    """[消融] 标准多头自注意力时序建模（替代 MambaBlock）"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + self.dropout(x)

# ============================================================================
# 3. 串行编码器
# ============================================================================
class SerialSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_nodes, d_state=32, dropout=0.1,
                 temporal_module='mamba', use_graph_bias=True):
        super().__init__()
        # [消融开关] 时序建模模块选择
        if temporal_module == 'self_attention':
            self.temporal_block = TemporalSelfAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
        else:
            self.temporal_block = MambaBlock(d_model, d_state=d_state, dropout=dropout)
        self.spatial_attn = SpatialAttentionWithBias(
            d_model, n_heads, num_nodes, dropout=dropout, use_graph_bias=use_graph_bias)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, N, D = x.shape
        
        # Time Phase
        x_t = rearrange(x, 'b t n d -> (b n) t d')
        x_t = self.temporal_block(x_t)
        x = rearrange(x_t, '(b n) t d -> b t n d', b=B)
        
        # Space Phase
        x_s = rearrange(x, 'b t n d -> (b t) n d')
        x_s = self.spatial_attn(x_s)
        x = rearrange(x_s, '(b t) n d -> b t n d', b=B)
        
        # FFN
        x = x + self.ffn(x)
        return x

# ============================================================================
# 4. 独立解码器 (Parallel Decoding) & 共享解码器 (消融用)
# ============================================================================
class ParallelPredictor(nn.Module):
    def __init__(self, d_model, input_len, output_len, output_dim, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        
        # 共享特征提取
        self.shared_proj = nn.Sequential(
            nn.Linear(input_len * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # [核心] 独立预测头：每个时间步有专属的参数
        self.heads = nn.ModuleList([
            nn.Linear(d_model * 2, output_dim)
            for _ in range(output_len)
        ])

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        x_flat = rearrange(x, 'b t n d -> b n (t d)')
        feat = self.shared_proj(x_flat)
        
        outputs = []
        for head in self.heads:
            outputs.append(head(feat))
            
        return torch.stack(outputs, dim=1) # (B, T_out, N, C_out)


class SharedPredictor(nn.Module):
    """[消融] 共享解码头（替代 ParallelPredictor）"""
    def __init__(self, d_model, input_len, output_len, output_dim, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        
        self.shared_proj = nn.Sequential(
            nn.Linear(input_len * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 所有时间步共享同一个线性层
        self.shared_head = nn.Linear(d_model * 2, output_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = rearrange(x, 'b t n d -> b n (t d)')
        feat = self.shared_proj(x_flat)  # (B, N, d_model*2)
        out = self.shared_head(feat)     # (B, N, output_dim)
        # 复制到所有时间步
        return out.unsqueeze(1).expand(-1, self.output_len, -1, -1)

# ============================================================================
# 5. Learnable Fourier Position Encoding
# ============================================================================
class LearnableFourierEncoding(nn.Module):
    """可学习傅里叶位置编码 - 捕获多周期模式(日/周/月)"""
    def __init__(self, d_model, max_len=288*7, n_freqs=32, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        
        # 可学习频率和相位
        self.freqs = nn.Parameter(torch.randn(n_freqs) * 0.01)
        self.phases = nn.Parameter(torch.randn(n_freqs) * 0.01)
        
        # 投影到模型维度
        self.proj = nn.Linear(n_freqs * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化一些典型周期
        with torch.no_grad():
            self.freqs[0] = 1.0 / 288  # 日周期
            self.freqs[1] = 1.0 / (288 * 7)  # 周周期
            self.freqs[2] = 1.0 / 144  # 半天周期
    
    def forward(self, t_indices):
        t = t_indices.float().unsqueeze(-1)
        freqs = self.freqs.to(t.device)
        phases = self.phases.to(t.device)
        angles = 2 * np.pi * freqs * t + phases
        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)
        fourier_feats = torch.cat([sin_feats, cos_feats], dim=-1)
        pos_emb = self.proj(fourier_feats)
        pos_emb = self.dropout(pos_emb)
        return pos_emb

# ============================================================================
# 6. 主模型 (ST-Mamba-Res)
# ============================================================================
class MambaFormerModel(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, input_dim, output_dim,
                 d_model=96, n_heads=4, n_layers=3, d_state=32, 
                 n_freqs=32, scales=None, dropout=0.05,
                 use_revin=True, use_graph_bias=True, temporal_module='mamba',
                 use_parallel_predictor=True, use_fourier_enc=True,
                 use_residual_mix=True):
        super().__init__()
        self.output_dim = output_dim
        self.output_len = output_len
        self.num_nodes = num_nodes
        
        # [消融开关]
        self.use_revin = use_revin
        self.use_fourier_enc = use_fourier_enc
        self.use_residual_mix = use_residual_mix
        
        # 1. RevIN (只对交通特征做归一化)
        if self.use_revin:
            self.revin = RevIN(input_dim, affine=True)
        
        # 2. Components
        self.input_proj = nn.Linear(input_dim, d_model)
        if self.use_fourier_enc:
            self.fourier_enc = LearnableFourierEncoding(d_model, n_freqs=n_freqs, dropout=dropout)
        self.spatial_emb = nn.Embedding(num_nodes, d_model)
        
        # 3. Day-of-week 周期性编码 (sin/cos 捕捉周期性)
        self.day_proj = nn.Linear(2, d_model)
        self.day_dropout = nn.Dropout(dropout)
        
        self.encoders = nn.ModuleList([
            SerialSTEncoder(d_model, n_heads, num_nodes, d_state, dropout,
                            temporal_module=temporal_module, use_graph_bias=use_graph_bias)
            for _ in range(n_layers)
        ])
        
        # [消融开关] 解码器选择
        if use_parallel_predictor:
            self.predictor = ParallelPredictor(d_model, input_len, output_len, output_dim, dropout)
        else:
            self.predictor = SharedPredictor(d_model, input_len, output_len, output_dim, dropout)

        # Residual mixing gate: alpha in (0,1)
        if self.use_residual_mix:
            self.residual_mix = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x, t_indices=None, day_indices=None):
        # x: (B, T, N, C) - 交通特征
        # t_indices: (B, T) - time_in_day 索引
        # day_indices: (B, T) - day_of_week 索引 (0-6)
        
        # [Step 1] 归一化
        if self.use_revin:
            x_norm = self.revin(x, 'norm')
        else:
            x_norm = x
        
        # [Step 2] Embedding
        x_emb = self.input_proj(x_norm)
        
        # 时序编码: Fourier(time_in_day) + Cyclical(day_of_week)
        if t_indices is not None and self.use_fourier_enc:
            temporal_enc = self.fourier_enc(t_indices)  # (B, T, d_model)
            if day_indices is not None:
                day_rad = day_indices.float() * (2 * 3.14159265 / 7)
                day_sin = torch.sin(day_rad)
                day_cos = torch.cos(day_rad)
                day_feat = torch.stack([day_sin, day_cos], dim=-1)
                day_enc = self.day_dropout(self.day_proj(day_feat))
                temporal_enc = temporal_enc + day_enc
            x_emb = x_emb + temporal_enc.unsqueeze(2)
        elif t_indices is not None and day_indices is not None:
            # 即使关闭 Fourier，day encoding 仍然保留
            day_rad = day_indices.float() * (2 * 3.14159265 / 7)
            day_sin = torch.sin(day_rad)
            day_cos = torch.cos(day_rad)
            day_feat = torch.stack([day_sin, day_cos], dim=-1)
            day_enc = self.day_dropout(self.day_proj(day_feat))
            x_emb = x_emb + day_enc.unsqueeze(2)
        
        # 空间编码
        node_idx = torch.arange(self.num_nodes, device=x.device)
        x_emb = x_emb + self.spatial_emb(node_idx).unsqueeze(0).unsqueeze(0)
        
        # [Step 3] Encoding
        z = x_emb
        for encoder in self.encoders:
            z = encoder(z)
            
        # [Step 4] Decoding (预测增量)
        out = self.predictor(z)
        
        # [Step 5] 残差连接
        last_step = x_norm[:, -1:, :, :self.output_dim]
        delta = out
        if self.use_residual_mix:
            # 可学习的混合残差连接
            out_indep = last_step + delta
            out_rollout = last_step + torch.cumsum(delta, dim=1)
            alpha = torch.sigmoid(self.residual_mix)
            out = (1.0 - alpha) * out_indep + alpha * out_rollout
        else:
            # 固定独立残差 (alpha=0)
            out = last_step + delta
        
        # [Step 6] 反归一化
        if self.use_revin:
            out = self.revin(out, 'denorm', target_dim=self.output_dim)
        
        return out


# ============================================================================
# 7. LibCity Wrapper (LibCity框架适配器)
# ============================================================================
class MambaFormer(AbstractTrafficStateModel):
    """MambaFormer的LibCity框架适配器 (GLRMamba)"""
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        self._logger = getLogger(__name__)
        self._scaler = self.data_feature.get('scaler')
        
        # 数据参数
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.input_dim = config.get('input_dim', 1)
        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        
        # 模型参数
        self.d_model = config.get('d_model', 96)
        self.n_heads = config.get('n_heads', 4)
        self.n_layers = config.get('n_layers', 3)
        self.d_state = config.get('d_state', 32)
        self.n_freqs = config.get('n_freqs', 32)
        self.scales = config.get('scales', [3, 5, 7])
        self.dropout = config.get('dropout', 0.1)
        
        # 损失函数
        self.set_loss = config.get('set_loss', 'masked_mae')
        self.device = config.get('device', torch.device('cpu'))
        
        # 时间参数
        self.slice_size_per_day = config.get('slice_size_per_day', 288)
        self.x_feature_dim = config.get('x_feature_dim', None)
        
        # ======= 消融实验开关 (默认值保持完整模型行为) =======
        self.use_revin = config.get('use_revin', True)
        self.use_graph_bias = config.get('use_graph_bias', True)
        self.temporal_module = config.get('temporal_module', 'mamba')
        self.use_parallel_predictor = config.get('use_parallel_predictor', True)
        self.use_fourier_enc = config.get('use_fourier_enc', True)
        self.use_residual_mix = config.get('use_residual_mix', True)
        
        self._build_model()
        
        self._logger.info('✨ GLRMamba 模型初始化完成')
        self._logger.info(f'   - 节点数: {self.num_nodes}')
        self._logger.info(f'   - 输入/输出窗口: {self.input_window}/{self.output_window}')
        self._logger.info(f'   - 模型维度: {self.d_model}, 注意力头: {self.n_heads}')
        self._logger.info(f'   - 编码器层数: {self.n_layers}')
        self._logger.info(f'   - Mamba可用: {MAMBA_AVAILABLE}')
        self._logger.info(f'   - RevIN: {self.use_revin}')
        self._logger.info(f'   - 图偏置注意力: {self.use_graph_bias}')
        self._logger.info(f'   - 时序模块: {self.temporal_module}')
        self._logger.info(f'   - 独立解码头: {self.use_parallel_predictor}')
        self._logger.info(f'   - 傅里叶编码: {self.use_fourier_enc}')
        self._logger.info(f'   - 可学习残差混合: {self.use_residual_mix}')
    
    def _build_model(self):
        model_input_dim = self.x_feature_dim if self.x_feature_dim else self.input_dim
        
        self.model = MambaFormerModel(
            num_nodes=self.num_nodes,
            input_len=self.input_window,
            output_len=self.output_window,
            input_dim=model_input_dim,
            output_dim=self.output_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_state=self.d_state,
            n_freqs=self.n_freqs,
            scales=self.scales,
            dropout=self.dropout,
            use_revin=self.use_revin,
            use_graph_bias=self.use_graph_bias,
            temporal_module=self.temporal_module,
            use_parallel_predictor=self.use_parallel_predictor,
            use_fourier_enc=self.use_fourier_enc,
            use_residual_mix=self.use_residual_mix,
        )
    
    def forward(self, batch):
        x = batch['X']  # (B, T, N, F) F可能是3或11
        B, T, N, F = x.shape
        device = x.device
        
        # 提取时间索引 (用于傅里叶编码)
        t_indices = self._extract_time_indices(batch, B, T)
        
        # 提取 day_of_week 索引 (用于 day_embedding)
        if F >= 11:
            # 完整特征: [traffic(3), time_in_day(1), day_of_week(7)]
            day_onehot = x[..., 4:11]  # (B,T,N,7)
            day_indices = day_onehot[:, :, 0, :].argmax(dim=-1)  # (B,T)
        else:
            # 只有交通特征，用默认值 0 (周一)
            day_indices = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # 只用交通特征 (前3维: traffic_flow, occupancy, speed)
        x_traffic = x[..., :3]  # (B,T,N,3)
        
        # 前向传播 (RevIN + 残差学习 已集成在模型中)
        return self.model(x_traffic, t_indices, day_indices)
    
    def _extract_time_indices(self, batch, B, T):
        """提取时间索引用于傅里叶编码"""
        x = batch['X']  # (B, T, N, 11)
        device = x.device
        
        # time_in_day 始终在维度 3 (原始数据: 0-2交通, 3时间, 4-10星期)
        if x.shape[-1] > 3:
            time_feat = x[..., 3]  # (B, T, N) time_in_day
            t_indices = time_feat[:, :, 0]  # (B, T) 取第一个节点
            if t_indices.max() <= 1.0:
                t_indices = (t_indices * self.slice_size_per_day)
            t_indices = t_indices.long()
        else:
            t_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        return t_indices
    
    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_pred = self.forward(batch)
        
        # 反归一化
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        
        # 计算损失
        if self.set_loss == 'masked_mae':
            return loss.masked_mae_torch(y_pred, y_true, null_val=np.nan)
        elif self.set_loss == 'mse':
            return loss.masked_mse_torch(y_pred, y_true)
        elif self.set_loss == 'huber':
            mask = ~torch.isnan(y_true)
            delta = 1.0
            residual = torch.abs(y_pred - y_true)
            huber = torch.where(residual < delta, 
                               0.5 * residual ** 2, 
                               delta * (residual - 0.5 * delta))
            return huber[mask].mean()
        else:
            return loss.masked_mae_torch(y_pred, y_true)
    
    def predict(self, batch):
        return self.forward(batch)

"""Core GLRMamba model.

This file contains the main architecture only. It is independent from LibCity
and can be imported directly in PyTorch projects.

Input:
    x: Tensor with shape (batch, input_len, num_nodes, input_dim)
    time_indices: optional Tensor with shape (batch, input_len) or (input_len,)
    day_indices: optional Tensor with shape (batch, input_len) or (input_len,)

Output:
    Tensor with shape (batch, output_len, num_nodes, output_dim)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False


class RevIN(nn.Module):
    """Reversible instance normalization for non-stationary time series."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(
        self, x: torch.Tensor, target_dim: Optional[int] = None
    ) -> torch.Tensor:
        if self.affine:
            weight = self.affine_weight[:target_dim] if target_dim else self.affine_weight
            bias = self.affine_bias[:target_dim] if target_dim else self.affine_bias
            x = (x - bias) / (weight + self.eps * self.eps)

        mean = self.mean[..., :target_dim] if target_dim else self.mean
        stdev = self.stdev[..., :target_dim] if target_dim else self.stdev
        return x * stdev + mean

    def forward(
        self, x: torch.Tensor, mode: str, target_dim: Optional[int] = None
    ) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x, target_dim)
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class MambaBlock(nn.Module):
    """Temporal sequence block with an optional Conv1d fallback."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.conv = nn.Conv1d(
                d_model,
                d_model * 2,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=d_model,
            )
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            x_t = rearrange(x, "b t d -> b d t")
            x_t = self.conv(x_t)[..., : x.shape[1]]
            gate, value = x_t.chunk(2, dim=1)
            x = self.proj(rearrange(F.silu(gate) * value, "b d t -> b t d"))

        return residual + self.dropout(x)


class SpatialAttentionWithBias(nn.Module):
    """Graph-guided spatial attention with a learnable bias gate."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_nodes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.graph_bias = nn.Parameter(torch.zeros(1, n_heads, num_nodes, num_nodes))
        self.graph_gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.xavier_uniform_(self.graph_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, d_model = x.shape
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, num_nodes, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + torch.sigmoid(self.graph_gate) * self.graph_bias
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(batch, num_nodes, d_model)
        out = self.o_proj(out)
        return residual + self.dropout(out)


class SerialSTEncoder(nn.Module):
    """Serial temporal Mamba block followed by graph-guided spatial attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_nodes: int,
        d_state: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
        )
        self.spatial_attn = SpatialAttentionWithBias(
            d_model=d_model,
            n_heads=n_heads,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, input_len, num_nodes, d_model = x.shape

        x_t = rearrange(x, "b t n d -> (b n) t d")
        x_t = self.temporal_block(x_t)
        x = rearrange(x_t, "(b n) t d -> b t n d", b=batch, n=num_nodes)

        x_s = rearrange(x, "b t n d -> (b t) n d")
        x_s = self.spatial_attn(x_s)
        x = rearrange(x_s, "(b t) n d -> b t n d", b=batch, t=input_len)

        return x + self.ffn(x)


class ParallelPredictor(nn.Module):
    """Independent prediction head for each forecasting horizon."""

    def __init__(
        self,
        d_model: int,
        input_len: int,
        output_len: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(input_len * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(d_model * 2, output_dim) for _ in range(output_len)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t n d -> b n (t d)")
        feat = self.shared_proj(x)
        return torch.stack([head(feat) for head in self.heads], dim=1)


class LearnableFourierEncoding(nn.Module):
    """Learnable Fourier temporal encoding for daily and weekly periodicity."""

    def __init__(
        self,
        d_model: int,
        n_freqs: int = 32,
        dropout: float = 0.1,
        slices_per_day: int = 288,
    ):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_freqs) * 0.01)
        self.phases = nn.Parameter(torch.randn(n_freqs) * 0.01)
        self.proj = nn.Linear(n_freqs * 2, d_model)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            if n_freqs > 0:
                self.freqs[0] = 1.0 / slices_per_day
            if n_freqs > 1:
                self.freqs[1] = 1.0 / (slices_per_day * 7)
            if n_freqs > 2:
                self.freqs[2] = 1.0 / (slices_per_day // 2)

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        t = time_indices.float().unsqueeze(-1)
        angles = 2.0 * math.pi * self.freqs.to(t.device) * t + self.phases.to(t.device)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.dropout(self.proj(fourier))


class GLRMamba(nn.Module):
    """Normalized graph-guided state space model for traffic forecasting."""

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        input_dim: int = 3,
        output_dim: int = 1,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        d_state: int = 32,
        n_freqs: int = 32,
        dropout: float = 0.1,
        slices_per_day: int = 288,
        use_revin: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(input_dim, affine=True)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.fourier_enc = LearnableFourierEncoding(
            d_model=d_model,
            n_freqs=n_freqs,
            dropout=dropout,
            slices_per_day=slices_per_day,
        )
        self.day_proj = nn.Linear(2, d_model)
        self.day_dropout = nn.Dropout(dropout)
        self.node_emb = nn.Embedding(num_nodes, d_model)

        self.encoders = nn.ModuleList(
            [
                SerialSTEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    num_nodes=num_nodes,
                    d_state=d_state,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.predictor = ParallelPredictor(
            d_model=d_model,
            input_len=input_len,
            output_len=output_len,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.residual_mix = nn.Parameter(torch.tensor(-3.0))

    def _prepare_indices(
        self,
        values: Optional[torch.Tensor],
        batch: int,
        input_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if values is None:
            return None
        values = values.to(device)
        if values.ndim == 1:
            values = values.unsqueeze(0).expand(batch, -1)
        if values.shape != (batch, input_len):
            raise ValueError(
                f"Expected index shape {(batch, input_len)}, got {tuple(values.shape)}"
            )
        return values

    def forward(
        self,
        x: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None,
        day_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, input_len, num_nodes, input_dim = x.shape
        if input_len != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {input_len}")
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {num_nodes}")
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        time_indices = self._prepare_indices(time_indices, batch, input_len, x.device)
        day_indices = self._prepare_indices(day_indices, batch, input_len, x.device)

        if self.use_revin:
            x_norm = self.revin(x, "norm")
        else:
            x_norm = x

        z = self.input_proj(x_norm)

        if time_indices is not None:
            z = z + self.fourier_enc(time_indices).unsqueeze(2)

        if day_indices is not None:
            day_rad = day_indices.float() * (2.0 * math.pi / 7.0)
            day_feat = torch.stack([torch.sin(day_rad), torch.cos(day_rad)], dim=-1)
            z = z + self.day_dropout(self.day_proj(day_feat)).unsqueeze(2)

        node_idx = torch.arange(self.num_nodes, device=x.device)
        z = z + self.node_emb(node_idx).view(1, 1, self.num_nodes, -1)

        for encoder in self.encoders:
            z = encoder(z)

        delta = self.predictor(z)
        last_value = x_norm[:, -1:, :, : self.output_dim]
        independent = last_value + delta
        rollout = last_value + torch.cumsum(delta, dim=1)
        alpha = torch.sigmoid(self.residual_mix)
        out = (1.0 - alpha) * independent + alpha * rollout

        if self.use_revin:
            out = self.revin(out, "denorm", target_dim=self.output_dim)

        return out


if __name__ == "__main__":
    model = GLRMamba(num_nodes=170, input_len=12, output_len=12)
    x = torch.randn(2, 12, 170, 3)
    time = torch.arange(12)
    day = torch.zeros(12, dtype=torch.long)
    y = model(x, time_indices=time, day_indices=day)
    print(f"output shape: {tuple(y.shape)}")

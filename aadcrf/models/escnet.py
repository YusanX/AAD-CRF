"""ESCNet neural tracker for auditory attention decoding."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ESCNetConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 64
    feature_dim: int = 32
    eeg_hidden_dim: int = 64
    env_hidden_dim: int = 32
    temperature: float = 1.0
    eeg_temporal_kernel: int = 33
    env_conv1_kernel: int = 25
    env_conv2_kernel: int = 33
    causal_temporal: bool = False
    dropout: float = 0.0
    use_multiscale: bool = False
    eeg_ms_kernels: list[int] = field(default_factory=list)
    env_ms_kernels: list[int] = field(default_factory=list)

def _auto_ms_kernels(base_kernel: int) -> list[int]:
    short = max(3, base_kernel // 4 * 2 + 1)
    medium = base_kernel
    long_ = base_kernel * 2 - 1
    return [short, medium, long_]

class MultiScaleTemporalBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernels: list[int], causal: bool=False) -> None:
        super().__init__()
        self.causal = causal
        if causal:
            self.branches = nn.ModuleList([nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=0, groups=in_channels, bias=False) for k in kernels])
            self._causal_pads = [k - 1 for k in kernels]
        else:
            self.branches = nn.ModuleList([nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels, bias=False) for k in kernels])
            self._causal_pads = [0] * len(kernels)
        self.project = nn.Conv1d(len(kernels) * in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            branches = [branch(F.pad(x, (pad, 0))) for branch, pad in zip(self.branches, self._causal_pads)]
        else:
            branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)
        return F.gelu(self.norm(self.project(x)))

class EEGEncoder(nn.Module):

    def __init__(self, n_channels: int, hidden_dim: int, feature_dim: int, temporal_kernel: int=33, dropout: float=0.0, causal: bool=False) -> None:
        super().__init__()
        self.causal = causal
        self.spatial_conv = nn.Conv2d(1, hidden_dim, kernel_size=(n_channels, 1), bias=False)
        self.spatial_ln = nn.GroupNorm(1, hidden_dim)
        if causal:
            self._causal_pad = temporal_kernel - 1
            self.temporal_dw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_kernel, padding=0, groups=hidden_dim, bias=False)
        else:
            self._causal_pad = 0
            _pad = (temporal_kernel - 1) // 2
            self.temporal_dw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_kernel, padding=_pad, groups=hidden_dim, bias=False)
        self.temporal_pw = nn.Conv1d(hidden_dim, feature_dim, kernel_size=1, bias=False)
        self.temporal_ln = nn.GroupNorm(1, feature_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.gelu(self.spatial_ln(self.spatial_conv(x)))
        x = x.squeeze(2)
        if self.causal:
            x = F.pad(x, (self._causal_pad, 0))
        x = F.gelu(self.temporal_ln(self.temporal_pw(self.temporal_dw(x))))
        return self.dropout(x)

class EnvEncoder(nn.Module):

    def __init__(self, hidden_dim: int, feature_dim: int, conv1_kernel: int=25, conv2_kernel: int=33, dropout: float=0.0, causal: bool=False) -> None:
        super().__init__()
        self.causal = causal
        if causal:
            self._pad1 = conv1_kernel - 1
            self._pad2 = conv2_kernel - 1
            self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=conv1_kernel, padding=0, bias=False)
            self.conv2 = nn.Conv1d(hidden_dim, feature_dim, kernel_size=conv2_kernel, padding=0, bias=False)
        else:
            self._pad1 = 0
            self._pad2 = 0
            self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=conv1_kernel, padding=(conv1_kernel - 1) // 2, bias=False)
            self.conv2 = nn.Conv1d(hidden_dim, feature_dim, kernel_size=conv2_kernel, padding=(conv2_kernel - 1) // 2, bias=False)
        self.ln1 = nn.GroupNorm(1, hidden_dim)
        self.ln2 = nn.GroupNorm(1, feature_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self._pad1, 0))
        x = F.gelu(self.ln1(self.conv1(x)))
        if self.causal:
            x = F.pad(x, (self._pad2, 0))
        x = F.gelu(self.ln2(self.conv2(x)))
        return self.dropout(x)

class MultiScaleEEGEncoder(nn.Module):

    def __init__(self, n_channels: int, hidden_dim: int, feature_dim: int, ms_kernels: list[int], dropout: float=0.0) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, hidden_dim, kernel_size=(n_channels, 1), bias=False)
        self.spatial_ln = nn.GroupNorm(1, hidden_dim)
        self.ms_temporal = MultiScaleTemporalBlock(in_channels=hidden_dim, out_channels=feature_dim, kernels=ms_kernels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.gelu(self.spatial_ln(self.spatial_conv(x)))
        x = x.squeeze(2)
        x = self.ms_temporal(x)
        return self.dropout(x)

class MultiScaleEnvEncoder(nn.Module):

    def __init__(self, hidden_dim: int, feature_dim: int, conv1_kernel: int, ms_kernels: list[int], dropout: float=0.0) -> None:
        super().__init__()
        _pad1 = (conv1_kernel - 1) // 2
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=conv1_kernel, padding=_pad1, bias=False)
        self.ln1 = nn.GroupNorm(1, hidden_dim)
        self.ms_temporal = MultiScaleTemporalBlock(in_channels=hidden_dim, out_channels=feature_dim, kernels=ms_kernels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.conv1(x)))
        x = self.ms_temporal(x)
        return self.dropout(x)

class ESCNet(nn.Module):

    def __init__(self, cfg: ESCNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.use_multiscale:
            eeg_ms_k = cfg.eeg_ms_kernels or _auto_ms_kernels(cfg.eeg_temporal_kernel)
            env_ms_k = cfg.env_ms_kernels or _auto_ms_kernels(cfg.env_conv2_kernel)
            self.eeg_encoder: nn.Module = MultiScaleEEGEncoder(n_channels=cfg.n_eeg_channels, hidden_dim=cfg.eeg_hidden_dim, feature_dim=cfg.feature_dim, ms_kernels=eeg_ms_k, dropout=cfg.dropout)
            self.env_encoder: nn.Module = MultiScaleEnvEncoder(hidden_dim=cfg.env_hidden_dim, feature_dim=cfg.feature_dim, conv1_kernel=cfg.env_conv1_kernel, ms_kernels=env_ms_k, dropout=cfg.dropout)
        else:
            self.eeg_encoder = EEGEncoder(n_channels=cfg.n_eeg_channels, hidden_dim=cfg.eeg_hidden_dim, feature_dim=cfg.feature_dim, temporal_kernel=cfg.eeg_temporal_kernel, dropout=cfg.dropout)
            self.env_encoder = EnvEncoder(hidden_dim=cfg.env_hidden_dim, feature_dim=cfg.feature_dim, conv1_kernel=cfg.env_conv1_kernel, conv2_kernel=cfg.env_conv2_kernel, dropout=cfg.dropout)
        self.log_temperature = nn.Parameter(torch.tensor(cfg.temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.001, max=10.0)

    @staticmethod
    def _temporal_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x - x.mean(dim=-1, keepdim=True)
        y_c = y - y.mean(dim=-1, keepdim=True)
        num = (x_c * y_c).sum(dim=-1)
        den = x_c.norm(dim=-1) * y_c.norm(dim=-1)
        r = num / (den + 1e-08)
        return r.mean(dim=-1)

    def encode(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg_feat = self.eeg_encoder(eeg)
        left_feat = self.env_encoder(left_env.unsqueeze(1))
        right_feat = self.env_encoder(right_env.unsqueeze(1))
        return (eeg_feat, left_feat, right_feat)

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        eeg_feat, left_feat, right_feat = self.encode(eeg, left_env, right_env)
        sim_left = self._temporal_pearson(eeg_feat, left_feat)
        sim_right = self._temporal_pearson(eeg_feat, right_feat)
        logit_diff = (sim_left - sim_right) / self.temperature
        return torch.stack([logit_diff, -logit_diff], dim=-1)

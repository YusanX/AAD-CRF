"""PyTorch ports of the ICASSP 2023 AuditoryEEG challenge models.

Original TensorFlow implementations from:
  ICASSP2023SPGC_AuditoryEEG/experiment_models.py
  https://github.com/exporl/auditory-eeg-challenge-2023-code

References
----------
Accou, B. et al. Modeling the relationship between acoustic stimulus and EEG
with a dilated convolutional neural network. EUSIPCO 2020.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

def _same_pad(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation // 2

def _temporal_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    num = (x_c * y_c).sum(dim=-1)
    den = x_c.norm(dim=-1) * y_c.norm(dim=-1)
    return (num / (den + 1e-08)).mean(dim=-1)

class _BilinearHead(nn.Module):

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim * feature_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, eeg_feat: torch.Tensor, env_feat: torch.Tensor) -> torch.Tensor:
        eeg_n = F.normalize(eeg_feat, p=2, dim=2)
        env_n = F.normalize(env_feat, p=2, dim=2)
        corr = torch.bmm(eeg_n, env_n.permute(0, 2, 1))
        flat = corr.reshape(corr.shape[0], -1)
        return self.linear(flat).squeeze(-1)

class _TransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, attn_dropout: float=0.0, ffn_dropout: float=0.0) -> None:
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(attn_dropout)
        self.drop2 = nn.Dropout(ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.att(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop1(h)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

@dataclass
class DilationTrackerConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 128
    layers: int = 3
    kernel_size: int = 3
    spatial_filters: int = 8
    dilation_filters: int = 32
    dropout: float = 0.1
    use_bilinear_head: bool = True

@dataclass
class EEGMHADCSpeechDCTrackerConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 128
    layers: int = 3
    kernel_size: int = 3
    dilation_filters: int = 32
    mha_num_heads: int = 2
    mha_ff_dim: int = 32
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    dropout: float = 0.1
    use_bilinear_head: bool = True

@dataclass
class EEGMHADCSpeechGRUDCTrackerConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 128
    layers: int = 3
    kernel_size: int = 3
    dilation_filters: int = 32
    mha_num_heads: int = 2
    mha_ff_dim: int = 32
    gru_hidden: int = 32
    bidirectional: bool = True
    gru_ln: bool = True
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    dropout: float = 0.1
    use_bilinear_head: bool = True

class DilationTracker(nn.Module):

    def __init__(self, cfg: DilationTrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eeg_spatial = nn.Conv1d(cfg.n_eeg_channels, cfg.spatial_filters, kernel_size=1, bias=True)
        self.eeg_spatial_gn = nn.GroupNorm(1, cfg.spatial_filters)
        eeg_d, eeg_gn, env_d, env_gn = ([], [], [], [])
        in_eeg, in_env = (cfg.spatial_filters, 1)
        for i in range(cfg.layers):
            d = cfg.kernel_size ** i
            pad = _same_pad(cfg.kernel_size, d)
            eeg_d.append(nn.Conv1d(in_eeg, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=pad, bias=True))
            eeg_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            env_d.append(nn.Conv1d(in_env, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=pad, bias=True))
            env_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            in_eeg = in_env = cfg.dilation_filters
        self.eeg_dilated = nn.ModuleList(eeg_d)
        self.eeg_dilated_gn = nn.ModuleList(eeg_gn)
        self.env_dilated = nn.ModuleList(env_d)
        self.env_dilated_gn = nn.ModuleList(env_gn)
        self.dropout = nn.Dropout(p=cfg.dropout)
        if cfg.use_bilinear_head:
            self.head = _BilinearHead(cfg.dilation_filters)
            self._temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            self.head = None
            self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        if self.cfg.use_bilinear_head:
            return self._temperature.squeeze()
        return self.log_temperature.exp().clamp(min=0.001, max=10.0)

    def _encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.eeg_spatial_gn(self.eeg_spatial(eeg)))
        for conv, gn in zip(self.eeg_dilated, self.eeg_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def _encode_env(self, env: torch.Tensor) -> torch.Tensor:
        x = env.unsqueeze(1)
        for conv, gn in zip(self.env_dilated, self.env_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        eeg_f = self._encode_eeg(eeg)
        left_f = self._encode_env(left_env)
        right_f = self._encode_env(right_env)
        if self.cfg.use_bilinear_head:
            s_l = self.head(eeg_f, left_f)
            s_r = self.head(eeg_f, right_f)
            delta = s_l - s_r
            return torch.stack([delta, -delta], dim=-1)
        s_l = _temporal_pearson(eeg_f, left_f)
        s_r = _temporal_pearson(eeg_f, right_f)
        d = (s_l - s_r) / self.temperature
        return torch.stack([d, -d], dim=-1)

def build_dilation_tracker(cfg: DilationTrackerConfig) -> DilationTracker:
    return DilationTracker(cfg)

class EEGMHADCSpeechDCTracker(nn.Module):

    def __init__(self, cfg: EEGMHADCSpeechDCTrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = _TransformerBlock(embed_dim=cfg.n_eeg_channels, num_heads=cfg.mha_num_heads, ff_dim=cfg.mha_ff_dim, attn_dropout=cfg.attn_dropout, ffn_dropout=cfg.ffn_dropout)
        eeg_d, eeg_gn, env_d, env_gn = ([], [], [], [])
        in_eeg, in_env = (cfg.n_eeg_channels, 1)
        for i in range(cfg.layers):
            d = cfg.kernel_size ** i
            pad = _same_pad(cfg.kernel_size, d)
            eeg_d.append(nn.Conv1d(in_eeg, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=pad, bias=True))
            eeg_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            env_d.append(nn.Conv1d(in_env, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=pad, bias=True))
            env_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            in_eeg = in_env = cfg.dilation_filters
        self.eeg_dilated = nn.ModuleList(eeg_d)
        self.eeg_dilated_gn = nn.ModuleList(eeg_gn)
        self.env_dilated = nn.ModuleList(env_d)
        self.env_dilated_gn = nn.ModuleList(env_gn)
        self.dropout = nn.Dropout(p=cfg.dropout)
        if cfg.use_bilinear_head:
            self.head = _BilinearHead(cfg.dilation_filters)
            self._temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            self.head = None
            self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        if self.cfg.use_bilinear_head:
            return self._temperature.squeeze()
        return self.log_temperature.exp().clamp(min=0.001, max=10.0)

    def _encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        for conv, gn in zip(self.eeg_dilated, self.eeg_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def _encode_env(self, env: torch.Tensor) -> torch.Tensor:
        x = env.unsqueeze(1)
        for conv, gn in zip(self.env_dilated, self.env_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        eeg_f = self._encode_eeg(eeg)
        left_f = self._encode_env(left_env)
        right_f = self._encode_env(right_env)
        if self.cfg.use_bilinear_head:
            s_l = self.head(eeg_f, left_f)
            s_r = self.head(eeg_f, right_f)
            delta = s_l - s_r
            return torch.stack([delta, -delta], dim=-1)
        s_l = _temporal_pearson(eeg_f, left_f)
        s_r = _temporal_pearson(eeg_f, right_f)
        d = (s_l - s_r) / self.temperature
        return torch.stack([d, -d], dim=-1)

def build_eeg_mha_dc_speech_dc_tracker(cfg: EEGMHADCSpeechDCTrackerConfig) -> EEGMHADCSpeechDCTracker:
    return EEGMHADCSpeechDCTracker(cfg)

class EEGMHADCSpeechGRUDCTracker(nn.Module):

    def __init__(self, cfg: EEGMHADCSpeechGRUDCTrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = _TransformerBlock(embed_dim=cfg.n_eeg_channels, num_heads=cfg.mha_num_heads, ff_dim=cfg.mha_ff_dim, attn_dropout=cfg.attn_dropout, ffn_dropout=cfg.ffn_dropout)
        eeg_d, eeg_gn = ([], [])
        in_eeg = cfg.n_eeg_channels
        for i in range(cfg.layers):
            d = cfg.kernel_size ** i
            eeg_d.append(nn.Conv1d(in_eeg, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=_same_pad(cfg.kernel_size, d), bias=True))
            eeg_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            in_eeg = cfg.dilation_filters
        self.eeg_dilated = nn.ModuleList(eeg_d)
        self.eeg_dilated_gn = nn.ModuleList(eeg_gn)
        self.env_gru = nn.GRU(input_size=1, hidden_size=cfg.gru_hidden, batch_first=True, bidirectional=cfg.bidirectional)
        gru_out_dim = cfg.gru_hidden * (2 if cfg.bidirectional else 1)
        self.gru_ln = nn.LayerNorm(gru_out_dim) if cfg.gru_ln else nn.Identity()
        env_d, env_gn = ([], [])
        in_env = gru_out_dim
        for i in range(cfg.layers):
            d = cfg.kernel_size ** i
            env_d.append(nn.Conv1d(in_env, cfg.dilation_filters, kernel_size=cfg.kernel_size, dilation=d, padding=_same_pad(cfg.kernel_size, d), bias=True))
            env_gn.append(nn.GroupNorm(1, cfg.dilation_filters))
            in_env = cfg.dilation_filters
        self.env_dilated = nn.ModuleList(env_d)
        self.env_dilated_gn = nn.ModuleList(env_gn)
        self.dropout = nn.Dropout(p=cfg.dropout)
        if cfg.use_bilinear_head:
            self.head = _BilinearHead(cfg.dilation_filters)
            self._temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            self.head = None
            self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        if self.cfg.use_bilinear_head:
            return self._temperature.squeeze()
        return self.log_temperature.exp().clamp(min=0.001, max=10.0)

    def _encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        for conv, gn in zip(self.eeg_dilated, self.eeg_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def _encode_env(self, env: torch.Tensor) -> torch.Tensor:
        x = env.unsqueeze(2)
        gru_out, _ = self.env_gru(x)
        gru_out = self.gru_ln(gru_out)
        x = gru_out.permute(0, 2, 1)
        for conv, gn in zip(self.env_dilated, self.env_dilated_gn):
            x = F.gelu(gn(conv(x)))
        return self.dropout(x)

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        eeg_f = self._encode_eeg(eeg)
        left_f = self._encode_env(left_env)
        right_f = self._encode_env(right_env)
        if self.cfg.use_bilinear_head:
            s_l = self.head(eeg_f, left_f)
            s_r = self.head(eeg_f, right_f)
            delta = s_l - s_r
            return torch.stack([delta, -delta], dim=-1)
        s_l = _temporal_pearson(eeg_f, left_f)
        s_r = _temporal_pearson(eeg_f, right_f)
        d = (s_l - s_r) / self.temperature
        return torch.stack([d, -d], dim=-1)

def build_eeg_mha_dc_speech_gru_dc_tracker(cfg: EEGMHADCSpeechGRUDCTrackerConfig) -> EEGMHADCSpeechGRUDCTracker:
    return EEGMHADCSpeechGRUDCTracker(cfg)

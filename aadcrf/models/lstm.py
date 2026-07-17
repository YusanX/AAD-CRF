"""LSTM-based Neural Tracker for AAD.

Architecture from:
  Jalilpour Monesi et al. / Borsdorf et al.,
  "Auditory Attention Detection from EEG using an LSTM-based model",
  ICASSP 2020 / IEEE Open Journal of Signal Processing 2024.
  https://www.csl.uni-bremen.de/cms/images/documents/publications/Borsdorf2024OJSP.pdf

"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LSTMTrackerConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 128
    units_lstm: int = 16
    filters_cnn_eeg: int = 8
    filters_cnn_env: int = 16
    units_hidden: int = 20
    kerSize_temporal: int = 9
    kerSize_ver_eeg: int = 7
    stride_temporal: int = 3
    stride_ch: int = 2
    dropout: float = 0.0

class _EEGEncoder(nn.Module):

    def __init__(self, n_channels: int, filters_cnn_eeg: int, units_hidden: int, units_lstm: int, kerSize_temporal: int, kerSize_ver_eeg: int, stride_temporal: int, stride_ch: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, filters_cnn_eeg, kernel_size=(kerSize_ver_eeg, kerSize_temporal), stride=(stride_ch, stride_temporal))
        c_prime = (n_channels - kerSize_ver_eeg) // stride_ch + 1
        flat_dim = filters_cnn_eeg * c_prime
        self.norm1 = nn.LayerNorm(flat_dim)
        self.dense1 = nn.Linear(flat_dim, units_hidden)
        self.norm2 = nn.LayerNorm(units_hidden)
        self.dense2 = nn.Linear(units_hidden, units_lstm)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.tanh(self.conv(x))
        B, F, Cp, Tp = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B, Tp, Cp * F)
        x = torch.tanh(self.dense1(self.norm1(x)))
        x = torch.tanh(self.dense2(self.norm2(x)))
        return self.drop(x)

class _EnvEncoder(nn.Module):

    def __init__(self, filters_cnn_env: int, units_lstm: int, kerSize_temporal: int, stride_temporal: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, filters_cnn_env, kernel_size=kerSize_temporal, stride=stride_temporal)
        self.norm = nn.LayerNorm(filters_cnn_env)
        self.lstm = nn.LSTM(filters_cnn_env, units_lstm, batch_first=True, dropout=dropout if 1 > 1 else 0.0)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.tanh(self.conv(x))
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        out, _ = self.lstm(x)
        return self.drop(out)

class _Scorer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.td_linear = nn.Linear(1, 1)

    def forward(self, eeg_feat: torch.Tensor, env_feat: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(eeg_feat, env_feat, dim=-1)
        score = self.td_linear(cos.unsqueeze(-1)).squeeze(-1)
        return score.mean(dim=-1)

class LSTMTracker(nn.Module):

    def __init__(self, cfg: LSTMTrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eeg_encoder = _EEGEncoder(n_channels=cfg.n_eeg_channels, filters_cnn_eeg=cfg.filters_cnn_eeg, units_hidden=cfg.units_hidden, units_lstm=cfg.units_lstm, kerSize_temporal=cfg.kerSize_temporal, kerSize_ver_eeg=cfg.kerSize_ver_eeg, stride_temporal=cfg.stride_temporal, stride_ch=cfg.stride_ch, dropout=cfg.dropout)
        self.env_encoder = _EnvEncoder(filters_cnn_env=cfg.filters_cnn_env, units_lstm=cfg.units_lstm, kerSize_temporal=cfg.kerSize_temporal, stride_temporal=cfg.stride_temporal, dropout=cfg.dropout)
        self.scorer = _Scorer()
        self._temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature.squeeze()

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        eeg_feat = self.eeg_encoder(eeg)
        left_feat = self.env_encoder(left_env)
        right_feat = self.env_encoder(right_env)
        score_left = self.scorer(eeg_feat, left_feat)
        score_right = self.scorer(eeg_feat, right_feat)
        delta = score_left - score_right
        return torch.stack([delta, -delta], dim=-1)

def build_lstm_tracker(cfg: LSTMTrackerConfig) -> LSTMTracker:
    return LSTMTracker(cfg)

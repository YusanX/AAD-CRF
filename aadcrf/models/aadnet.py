"""AADNet adapter for the shared training pipeline."""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AADNET_ROOT = _PROJECT_ROOT / 'AADNet'
if str(_AADNET_ROOT) not in sys.path:
    sys.path.insert(0, str(_AADNET_ROOT))
try:
    from aadnet.EnvelopeAAD import AADNet
    _AADNET_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    AADNet = None
    _AADNET_IMPORT_ERROR = exc

@dataclass
class AADNetTrackerConfig:
    n_eeg_channels: int = 64
    n_time_steps: int = 128
    chns_1: list = field(default_factory=lambda: [32, [16, 8], [8, 8], [4, 8], [2, 8], 8])
    kernels_1: list = field(default_factory=lambda: [1, 37, 49, 65, 77, 5])
    act_1: str = 'relu'
    chns_1_aud: list = field(default_factory=lambda: [1, [1, 4], [1, 4], 0])
    kernels_1_aud: list = field(default_factory=lambda: [1, 129, 161, 5])
    pool_stride_1: int = 1
    hidden_size: int = 0
    dropout: float = 0.4
    feature_freeze: bool = False

class AADNetTracker(nn.Module):

    def __init__(self, cfg: AADNetTrackerConfig) -> None:
        super().__init__()
        if AADNet is None:
            raise ModuleNotFoundError(
                "AADNet experiments require <PROJECT_ROOT>/AADNet/aadnet/EnvelopeAAD.py"
            ) from _AADNET_IMPORT_ERROR
        self.cfg = cfg
        aadnet_cfg = {'in_channels': cfg.n_eeg_channels, 'chns_1': list(cfg.chns_1), 'kernels_1': list(cfg.kernels_1), 'act_1': cfg.act_1, 'chns_1_aud': list(cfg.chns_1_aud), 'kernels_1_aud': list(cfg.kernels_1_aud), 'pool_stride_1': cfg.pool_stride_1, 'hidden_size': cfg.hidden_size, 'dropout': cfg.dropout, 'feature_freeze': cfg.feature_freeze}
        channels = list(range(cfg.n_eeg_channels))
        self.model = AADNet(config=aadnet_cfg, L=cfg.n_time_steps, n_streams=2, sr=1, channels=channels)
        self._temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature.squeeze()

    def forward(self, eeg: torch.Tensor, left_env: torch.Tensor, right_env: torch.Tensor) -> torch.Tensor:
        env = torch.stack([left_env, right_env], dim=1)
        return self.model(eeg, env)

def build_aadnet_tracker(cfg: AADNetTrackerConfig) -> AADNetTracker:
    return AADNetTracker(cfg)

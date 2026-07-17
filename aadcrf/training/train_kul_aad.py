
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from aadcrf.training import train_kul_escnet as base
import aadcrf.training.train_avgc_escnet as _avgc_base
from aadcrf.models.aadnet import AADNetTrackerConfig, build_aadnet_tracker


@dataclass
class ExperimentConfig(base.ExperimentConfig):

    use_aadnet: bool = True
    aadnet: AADNetTrackerConfig = field(default_factory=AADNetTrackerConfig)


def load_kul_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_cfg = base.load_kul_config(path)
    exp_raw = raw.get("experiment", {})
    aad_raw = raw.get("aadnet", {})

    return ExperimentConfig(
        dataset_dir=base_cfg.dataset_dir,
        output_dir=base_cfg.output_dir,
        signal=base_cfg.signal,
        tracker=base_cfg.tracker,
        window=base_cfg.window,
        train=base_cfg.train,
        p_switch_init=base_cfg.p_switch_init,
        learn_p_switch=base_cfg.learn_p_switch,
        device=base_cfg.device,
        max_subjects=base_cfg.max_subjects,
        max_trials_per_subject=base_cfg.max_trials_per_subject,
        subject=base_cfg.subject,
        trial=base_cfg.trial,
        seed=base_cfg.seed,
        use_aadnet=bool(exp_raw.get("use_aadnet", True)),
        aadnet=AADNetTrackerConfig(**aad_raw),
    )


def build_model(cfg: ExperimentConfig, fs: int) -> torch.nn.Module:
    if not cfg.use_aadnet:
        return base.build_model(cfg, fs)

    aad_cfg = AADNetTrackerConfig(
        n_eeg_channels=cfg.aadnet.n_eeg_channels,
        n_time_steps=int(round(cfg.window.window_s * fs)),
        chns_1=list(cfg.aadnet.chns_1),
        kernels_1=list(cfg.aadnet.kernels_1),
        act_1=cfg.aadnet.act_1,
        chns_1_aud=list(cfg.aadnet.chns_1_aud),
        kernels_1_aud=list(cfg.aadnet.kernels_1_aud),
        pool_stride_1=cfg.aadnet.pool_stride_1,
        hidden_size=cfg.aadnet.hidden_size,
        dropout=cfg.aadnet.dropout,
        feature_freeze=cfg.aadnet.feature_freeze,
    )
    return build_aadnet_tracker(aad_cfg)


def run_kul_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    old_build_model = _avgc_base.build_model
    try:
        _avgc_base.build_model = build_model
        return base.run_kul_experiment(cfg)
    finally:
        _avgc_base.build_model = old_build_model

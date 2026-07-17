
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from aadcrf.training import train_avgc_escnet as base
from aadcrf.models.lstm import LSTMTrackerConfig, build_lstm_tracker


@dataclass
class ExperimentConfig(base.ExperimentConfig):

    use_lstm: bool = True
    lstm: LSTMTrackerConfig = field(default_factory=LSTMTrackerConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_cfg = base.load_config(path)
    exp_raw  = raw.get("experiment", {})
    lstm_raw = raw.get("lstm", {})

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
        use_lstm=bool(exp_raw.get("use_lstm", True)),
        lstm=LSTMTrackerConfig(**lstm_raw) if lstm_raw else LSTMTrackerConfig(),
    )


def build_model(cfg: ExperimentConfig, fs: int) -> torch.nn.Module:
    if not cfg.use_lstm:
        return base.build_model(cfg, fs)

    lstm_cfg = LSTMTrackerConfig(
        n_eeg_channels=cfg.lstm.n_eeg_channels,
        n_time_steps=int(round(cfg.window.window_s * fs)),
        units_lstm=cfg.lstm.units_lstm,
        filters_cnn_eeg=cfg.lstm.filters_cnn_eeg,
        filters_cnn_env=cfg.lstm.filters_cnn_env,
        units_hidden=cfg.lstm.units_hidden,
        kerSize_temporal=cfg.lstm.kerSize_temporal,
        kerSize_ver_eeg=cfg.lstm.kerSize_ver_eeg,
        stride_temporal=cfg.lstm.stride_temporal,
        stride_ch=cfg.lstm.stride_ch,
        dropout=cfg.lstm.dropout,
    )
    return build_lstm_tracker(lstm_cfg)


def run_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    old_build_model = base.build_model
    try:
        base.build_model = build_model
        return base.run_experiment(cfg)
    finally:
        base.build_model = old_build_model

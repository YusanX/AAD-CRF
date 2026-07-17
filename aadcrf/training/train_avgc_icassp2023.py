
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from aadcrf.training import train_avgc_escnet as base
from aadcrf.models.icassp2023 import (
    DilationTrackerConfig,
    EEGMHADCSpeechDCTrackerConfig,
    EEGMHADCSpeechGRUDCTrackerConfig,
    build_dilation_tracker,
    build_eeg_mha_dc_speech_dc_tracker,
    build_eeg_mha_dc_speech_gru_dc_tracker,
)

_VALID_VARIANTS = ("dilation", "mha_dc", "mha_gru_dc")


@dataclass
class ExperimentConfig(base.ExperimentConfig):

    icassp2023_variant: str = "dilation"
    dilation: DilationTrackerConfig = field(
        default_factory=DilationTrackerConfig
    )
    mha_dc: EEGMHADCSpeechDCTrackerConfig = field(
        default_factory=EEGMHADCSpeechDCTrackerConfig
    )
    mha_gru_dc: EEGMHADCSpeechGRUDCTrackerConfig = field(
        default_factory=EEGMHADCSpeechGRUDCTrackerConfig
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_cfg = base.load_config(path)
    exp_raw = raw.get("experiment", {})
    variant = str(exp_raw.get("icassp2023_variant", "dilation"))

    if variant not in _VALID_VARIANTS:
        raise ValueError(
            f"icassp2023_variant={variant!r} is not one of {_VALID_VARIANTS}"
        )

    dilation_raw  = raw.get("dilation", {})
    mha_dc_raw    = raw.get("mha_dc", {})
    mha_gru_dc_raw = raw.get("mha_gru_dc", {})

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
        icassp2023_variant=variant,
        dilation=DilationTrackerConfig(**dilation_raw) if dilation_raw else DilationTrackerConfig(),
        mha_dc=EEGMHADCSpeechDCTrackerConfig(**mha_dc_raw) if mha_dc_raw else EEGMHADCSpeechDCTrackerConfig(),
        mha_gru_dc=EEGMHADCSpeechGRUDCTrackerConfig(**mha_gru_dc_raw) if mha_gru_dc_raw else EEGMHADCSpeechGRUDCTrackerConfig(),
    )


def build_model(cfg: ExperimentConfig, fs: int) -> torch.nn.Module:
    n_time = int(round(cfg.window.window_s * fs))
    variant = cfg.icassp2023_variant

    if variant == "dilation":
        model_cfg = DilationTrackerConfig(
            n_eeg_channels=cfg.dilation.n_eeg_channels,
            n_time_steps=n_time,
            layers=cfg.dilation.layers,
            kernel_size=cfg.dilation.kernel_size,
            spatial_filters=cfg.dilation.spatial_filters,
            dilation_filters=cfg.dilation.dilation_filters,
            dropout=cfg.dilation.dropout,
            use_bilinear_head=cfg.dilation.use_bilinear_head,
        )
        return build_dilation_tracker(model_cfg)

    if variant == "mha_dc":
        model_cfg = EEGMHADCSpeechDCTrackerConfig(
            n_eeg_channels=cfg.mha_dc.n_eeg_channels,
            n_time_steps=n_time,
            layers=cfg.mha_dc.layers,
            kernel_size=cfg.mha_dc.kernel_size,
            dilation_filters=cfg.mha_dc.dilation_filters,
            mha_num_heads=cfg.mha_dc.mha_num_heads,
            mha_ff_dim=cfg.mha_dc.mha_ff_dim,
            attn_dropout=cfg.mha_dc.attn_dropout,
            ffn_dropout=cfg.mha_dc.ffn_dropout,
            dropout=cfg.mha_dc.dropout,
            use_bilinear_head=cfg.mha_dc.use_bilinear_head,
        )
        return build_eeg_mha_dc_speech_dc_tracker(model_cfg)

    if variant == "mha_gru_dc":
        model_cfg = EEGMHADCSpeechGRUDCTrackerConfig(
            n_eeg_channels=cfg.mha_gru_dc.n_eeg_channels,
            n_time_steps=n_time,
            layers=cfg.mha_gru_dc.layers,
            kernel_size=cfg.mha_gru_dc.kernel_size,
            dilation_filters=cfg.mha_gru_dc.dilation_filters,
            mha_num_heads=cfg.mha_gru_dc.mha_num_heads,
            mha_ff_dim=cfg.mha_gru_dc.mha_ff_dim,
            gru_hidden=cfg.mha_gru_dc.gru_hidden,
            bidirectional=cfg.mha_gru_dc.bidirectional,
            gru_ln=cfg.mha_gru_dc.gru_ln,
            attn_dropout=cfg.mha_gru_dc.attn_dropout,
            ffn_dropout=cfg.mha_gru_dc.ffn_dropout,
            dropout=cfg.mha_gru_dc.dropout,
            use_bilinear_head=cfg.mha_gru_dc.use_bilinear_head,
        )
        return build_eeg_mha_dc_speech_gru_dc_tracker(model_cfg)

    raise ValueError(f"Unknown variant: {variant!r}")


def run_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    old_build_model = base.build_model
    try:
        base.build_model = build_model
        return base.run_experiment(cfg)
    finally:
        base.build_model = old_build_model

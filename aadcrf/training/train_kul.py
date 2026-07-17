
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

import aadcrf.training.train_avgc_escnet as _base_avgc
import aadcrf.training.train_kul_escnet as _kul_base

_escnet_build_model = _base_avgc.build_model
from aadcrf.models.aadnet import AADNetTrackerConfig, build_aadnet_tracker
from aadcrf.models.icassp2023 import (
    DilationTrackerConfig,
    EEGMHADCSpeechDCTrackerConfig,
    EEGMHADCSpeechGRUDCTrackerConfig,
    build_dilation_tracker,
    build_eeg_mha_dc_speech_dc_tracker,
    build_eeg_mha_dc_speech_gru_dc_tracker,
)
from aadcrf.models.lstm import LSTMTrackerConfig, build_lstm_tracker

_VALID_MODELS = ("escnet", "aadnet", "lstm", "icassp2023")
_VALID_ICASSP_VARIANTS = ("dilation", "mha_dc", "mha_gru_dc")



@dataclass
class ExperimentConfig(_base_avgc.ExperimentConfig):

    model: str = "escnet"
    icassp2023_variant: str = "mha_gru_dc"
    aadnet: AADNetTrackerConfig = field(default_factory=AADNetTrackerConfig)
    lstm: LSTMTrackerConfig = field(default_factory=LSTMTrackerConfig)
    dilation: DilationTrackerConfig = field(default_factory=DilationTrackerConfig)
    mha_dc: EEGMHADCSpeechDCTrackerConfig = field(
        default_factory=EEGMHADCSpeechDCTrackerConfig
    )
    mha_gru_dc: EEGMHADCSpeechGRUDCTrackerConfig = field(
        default_factory=EEGMHADCSpeechGRUDCTrackerConfig
    )



def load_kul_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_cfg = _kul_base.load_kul_config(path)

    model = str(raw.get("model", "escnet")).lower()
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model={model!r} is not one of {_VALID_MODELS}. "
            "Set the top-level 'model' key in your YAML config."
        )

    exp_raw = raw.get("experiment", {})
    variant = str(exp_raw.get("icassp2023_variant", "mha_gru_dc"))
    if variant not in _VALID_ICASSP_VARIANTS:
        raise ValueError(
            f"icassp2023_variant={variant!r} is not one of {_VALID_ICASSP_VARIANTS}"
        )

    aadnet_raw     = raw.get("aadnet", {})
    lstm_raw       = raw.get("lstm", {})
    dilation_raw   = raw.get("dilation", {})
    mha_dc_raw     = raw.get("mha_dc", {})
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
        model=model,
        icassp2023_variant=variant,
        aadnet=AADNetTrackerConfig(**aadnet_raw) if aadnet_raw else AADNetTrackerConfig(),
        lstm=LSTMTrackerConfig(**lstm_raw) if lstm_raw else LSTMTrackerConfig(),
        dilation=(
            DilationTrackerConfig(**dilation_raw) if dilation_raw
            else DilationTrackerConfig()
        ),
        mha_dc=(
            EEGMHADCSpeechDCTrackerConfig(**mha_dc_raw) if mha_dc_raw
            else EEGMHADCSpeechDCTrackerConfig()
        ),
        mha_gru_dc=(
            EEGMHADCSpeechGRUDCTrackerConfig(**mha_gru_dc_raw) if mha_gru_dc_raw
            else EEGMHADCSpeechGRUDCTrackerConfig()
        ),
    )



def build_model(cfg: ExperimentConfig, fs: int) -> torch.nn.Module:
    n_time = int(round(cfg.window.window_s * fs))
    model = cfg.model

    if model == "escnet":
        return _escnet_build_model(cfg, fs)

    if model == "aadnet":
        aad_cfg = AADNetTrackerConfig(
            n_eeg_channels=cfg.aadnet.n_eeg_channels,
            n_time_steps=n_time,
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

    if model == "lstm":
        lstm_cfg = LSTMTrackerConfig(
            n_eeg_channels=cfg.lstm.n_eeg_channels,
            n_time_steps=n_time,
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

    if model == "icassp2023":
        variant = cfg.icassp2023_variant
        if variant == "dilation":
            dil_cfg = DilationTrackerConfig(
                n_eeg_channels=cfg.dilation.n_eeg_channels,
                n_time_steps=n_time,
                layers=cfg.dilation.layers,
                kernel_size=cfg.dilation.kernel_size,
                spatial_filters=cfg.dilation.spatial_filters,
                dilation_filters=cfg.dilation.dilation_filters,
                dropout=cfg.dilation.dropout,
                use_bilinear_head=cfg.dilation.use_bilinear_head,
            )
            return build_dilation_tracker(dil_cfg)
        if variant == "mha_dc":
            mha_cfg = EEGMHADCSpeechDCTrackerConfig(
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
            return build_eeg_mha_dc_speech_dc_tracker(mha_cfg)
        if variant == "mha_gru_dc":
            gru_cfg = EEGMHADCSpeechGRUDCTrackerConfig(
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
            return build_eeg_mha_dc_speech_gru_dc_tracker(gru_cfg)
        raise ValueError(f"Unknown icassp2023_variant: {variant!r}")

    raise ValueError(f"Unknown model: {model!r}")



def run_kul_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    old_build = _base_avgc.build_model
    try:
        _base_avgc.build_model = build_model
        return _kul_base.run_kul_experiment(cfg)
    finally:
        _base_avgc.build_model = old_build

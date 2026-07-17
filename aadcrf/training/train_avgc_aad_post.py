
from __future__ import annotations

from pathlib import Path

from aadcrf.training import train_avgc_aad as aad_base

ExperimentConfig = aad_base.ExperimentConfig


def _force_postprocessing_baseline(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg.learn_p_switch = False
    cfg.train.lambda_crf = 0.0
    cfg.train.warmup_crf_epoch = cfg.train.epochs
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = aad_base.load_config(path)
    return _force_postprocessing_baseline(cfg)


def run_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    cfg = _force_postprocessing_baseline(cfg)
    return aad_base.run_experiment(cfg)

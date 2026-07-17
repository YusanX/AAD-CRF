from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal


@dataclass
class SignalPreprocessConfig:
    low_hz: float = 1.0
    high_hz: float = 9.0
    target_fs: int = 64
    rereference: str = "common_average"
    filter_order: int = 4


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    mean_per_sample = np.mean(eeg, axis=1, keepdims=True)
    return eeg - mean_per_sample


def bandpass_filter(data: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, data, axis=0)


def resample_to_fs(data: np.ndarray, source_fs: float, target_fs: int) -> np.ndarray:
    if int(source_fs) == int(target_fs):
        return data
    ratio = Fraction(int(target_fs), int(source_fs)).limit_denominator()
    return signal.resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=0)


def align_to_shortest(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    min_len = min(arr.shape[0] for arr in arrays)
    return tuple(arr[:min_len] for arr in arrays)


def preprocess_trial_signals(
    eeg: np.ndarray,
    left_env: np.ndarray,
    right_env: np.ndarray,
    source_fs: float,
    cfg: SignalPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if cfg.rereference == "common_average":
        eeg = common_average_reference(eeg)

    eeg = bandpass_filter(eeg, fs=source_fs, low_hz=cfg.low_hz, high_hz=cfg.high_hz, order=cfg.filter_order)
    left_env = bandpass_filter(left_env[:, None], fs=source_fs, low_hz=cfg.low_hz, high_hz=cfg.high_hz, order=cfg.filter_order)[:, 0]
    right_env = bandpass_filter(right_env[:, None], fs=source_fs, low_hz=cfg.low_hz, high_hz=cfg.high_hz, order=cfg.filter_order)[:, 0]

    eeg = resample_to_fs(eeg, source_fs=source_fs, target_fs=cfg.target_fs)
    left_env = resample_to_fs(left_env[:, None], source_fs=source_fs, target_fs=cfg.target_fs)[:, 0]
    right_env = resample_to_fs(right_env[:, None], source_fs=source_fs, target_fs=cfg.target_fs)[:, 0]

    eeg, left_env, right_env = align_to_shortest(eeg, left_env[:, None], right_env[:, None])
    return eeg, left_env[:, 0], right_env[:, 0], cfg.target_fs


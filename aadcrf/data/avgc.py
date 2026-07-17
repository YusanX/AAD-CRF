from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from aadcrf.preprocess.signal import SignalPreprocessConfig, preprocess_trial_signals


SPEAKER_A_STATE = 0
SPEAKER_B_STATE = 1

LEFT_STATE = SPEAKER_A_STATE
RIGHT_STATE = SPEAKER_B_STATE


@dataclass
class AvgcTrial:
    subject_id: str
    trial_idx: int
    condition_id: str
    fs: int
    eeg: np.ndarray
    left_env: np.ndarray
    right_env: np.ndarray
    sample_labels: np.ndarray
    switch_time_s: float
    first_attended_side: str


def _normalize_side(raw: str) -> str:
    side = str(raw).strip().upper()
    if side.startswith("L"):
        return "L"
    if side.startswith("R"):
        return "R"
    raise ValueError(f"Unsupported side value: {raw}")


def _build_sample_labels(length: int, fs: int, initial_side: str, switch_time_s: float) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    init_state = SPEAKER_A_STATE if initial_side == "L" else SPEAKER_B_STATE
    labels[:] = init_state

    switch_sample = int(round(switch_time_s * fs))
    switch_sample = int(np.clip(switch_sample, 0, length))
    labels[switch_sample:] = SPEAKER_B_STATE if init_state == SPEAKER_A_STATE else SPEAKER_A_STATE
    return labels


def _extract_subject_id(raw_subj_id: object) -> str:
    return str(raw_subj_id)


def _build_spatial_streams(
    left_identity_env: np.ndarray,
    right_identity_env: np.ndarray,
    fs: int,
    switch_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    switch_sample = int(round(switch_time_s * fs))
    switch_sample = int(np.clip(switch_sample, 0, left_identity_env.shape[0]))

    spatial_left = left_identity_env.copy()
    spatial_right = right_identity_env.copy()
    spatial_left[switch_sample:] = right_identity_env[switch_sample:]
    spatial_right[switch_sample:] = left_identity_env[switch_sample:]
    return spatial_left, spatial_right


def load_avgc_subject_file(
    mat_path: str | Path,
    preprocess_cfg: SignalPreprocessConfig,
) -> list[AvgcTrial]:
    mat_path = Path(mat_path)
    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    subj_id = _extract_subject_id(raw["subjID"])
    fs = int(raw["fs"])

    data_trials = np.asarray(raw["data"], dtype=object)
    cond_trials = np.asarray(raw["conditionID"], dtype=object)
    init_attn_trials = np.asarray(raw["initAttention"], dtype=object)
    rand_trials = np.asarray(raw["randomization"], dtype=object)
    stimulus = raw["stimulus"]

    left_env_trials = np.asarray(stimulus.leftEnvelopes, dtype=object)
    right_env_trials = np.asarray(stimulus.rightEnvelopes, dtype=object)

    trials: list[AvgcTrial] = []
    for i in range(data_trials.shape[0]):
        trial_data = np.asarray(data_trials[i], dtype=np.float64)
        eeg = trial_data[:, :64]

        left_env = np.asarray(left_env_trials[i], dtype=np.float64).reshape(-1)
        right_env = np.asarray(right_env_trials[i], dtype=np.float64).reshape(-1)

        rand = rand_trials[i]
        switch_time_s = float(rand.switch_times)

        eeg_proc, left_proc, right_proc, out_fs = preprocess_trial_signals(
            eeg=eeg,
            left_env=left_env,
            right_env=right_env,
            source_fs=fs,
            cfg=preprocess_cfg,
        )

        left_spatial, right_spatial = _build_spatial_streams(
            left_identity_env=left_proc,
            right_identity_env=right_proc,
            fs=out_fs,
            switch_time_s=switch_time_s,
        )

        first_side = _normalize_side(str(init_attn_trials[i]))

        fas_rand = _normalize_side(str(rand.first_attended_side))
        if fas_rand != first_side:
            import warnings
            warnings.warn(
                f"[avgc] {subj_id} trial {i}: initAttention={first_side!r} "
                f"but randomization.first_attended_side={fas_rand!r}. "
                f"Using initAttention (authoritative).",
                stacklevel=2,
            )

        labels = _build_sample_labels(
            length=eeg_proc.shape[0],
            fs=out_fs,
            initial_side=first_side,
            switch_time_s=switch_time_s,
        )

        trials.append(
            AvgcTrial(
                subject_id=subj_id,
                trial_idx=i,
                condition_id=str(cond_trials[i]),
                fs=out_fs,
                eeg=eeg_proc,
                left_env=left_spatial,
                right_env=right_spatial,
                sample_labels=labels,
                switch_time_s=switch_time_s,
                first_attended_side=first_side,
            )
        )
    return trials


def discover_avgc_subject_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    return sorted(dataset_dir.glob("2024-AV-GC-AAD-sub*_preprocessed.mat"))


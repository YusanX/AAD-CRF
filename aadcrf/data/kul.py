
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from aadcrf.preprocess.signal import (
    SignalPreprocessConfig,
    preprocess_trial_signals,
    resample_to_fs,
)

LEFT_STATE = 0
RIGHT_STATE = 1

KUL_MAX_TRIALS = 8



@dataclass
class KulTrial:
    subject_id: str
    trial_idx: int
    trial_id: int
    condition_id: str
    experiment: int
    fs: int
    eeg: np.ndarray
    left_env: np.ndarray
    right_env: np.ndarray
    sample_labels: np.ndarray
    switch_time_s: float = float('inf')
    attended_ear: str = ''
    first_attended_side: str = ''



def _normalize_attended_ear(raw: object) -> str:
    s = str(raw).strip().upper()
    if s.startswith('L'):
        return 'L'
    if s.startswith('R'):
        return 'R'
    raise ValueError(f"Unrecognised attended_ear value: {raw!r}")


def _build_kul_labels(length: int, attended_ear: str) -> np.ndarray:
    label = LEFT_STATE if attended_ear == 'L' else RIGHT_STATE
    return np.full(length, label, dtype=np.int64)


def _scalar(x: object) -> int | float:
    return np.asarray(x).flat[0]



def discover_kul_subject_files(
    dataset_dir: str | Path,
) -> tuple[list[Path], bool]:
    dataset_dir = Path(dataset_dir)

    prep_dir = dataset_dir / 'preprocessed_data'
    if prep_dir.is_dir():
        files = sorted(prep_dir.glob('S*.mat'))
        if files:
            return files, True

    files = sorted(dataset_dir.glob('S*.mat'))
    if not files:
        raise FileNotFoundError(
            f"No KUL subject files found in '{dataset_dir}'.\n"
            "Expected either:\n"
            "  <kul_dir>/preprocessed_data/S*.mat  (run preprocess_data.m first)\n"
            "  <kul_dir>/S*.mat                    (raw EEG, requires stimuli/ folder)"
        )
    return files, False



def _load_kul_preprocessed(
    mat_path: Path,
    preprocess_cfg: SignalPreprocessConfig,
    max_trials: int = KUL_MAX_TRIALS,
) -> list[KulTrial]:
    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    preproc_trials = np.asarray(raw['preproc_trials'], dtype=object)

    trials: list[KulTrial] = []
    n_load = min(max_trials, preproc_trials.shape[0])

    for i in range(n_load):
        t = preproc_trials[i]

        eeg = np.asarray(t.RawData.EegData, dtype=np.float64)
        fs = int(_scalar(t.FileHeader.SampleRate))

        audio_data = np.asarray(t.Envelope.AudioData, dtype=np.float64)
        weights = np.asarray(t.Envelope.subband_weights, dtype=np.float64).ravel()

        if audio_data.ndim == 3:
            left_env_raw = audio_data[:, :, 0] @ weights
            right_env_raw = audio_data[:, :, 1] @ weights
        elif audio_data.ndim == 2:
            left_env_raw = audio_data[:, 0]
            right_env_raw = audio_data[:, 1]
        else:
            raise ValueError(
                f"Unexpected Envelope.AudioData shape {audio_data.shape} "
                f"in {mat_path.name} trial {i + 1}"
            )

        attended_ear = _normalize_attended_ear(t.attended_ear)
        trial_id     = int(_scalar(t.TrialID))
        condition    = str(t.condition).strip()
        experiment   = int(_scalar(t.experiment))
        subject_id   = str(t.subject).strip()

        eeg_proc, left_proc, right_proc, out_fs = preprocess_trial_signals(
            eeg=eeg,
            left_env=left_env_raw,
            right_env=right_env_raw,
            source_fs=float(fs),
            cfg=preprocess_cfg,
        )

        sample_labels = _build_kul_labels(eeg_proc.shape[0], attended_ear)

        trials.append(KulTrial(
            subject_id=subject_id,
            trial_idx=i,
            trial_id=trial_id,
            condition_id=condition,
            experiment=experiment,
            fs=out_fs,
            eeg=eeg_proc,
            left_env=left_proc,
            right_env=right_proc,
            sample_labels=sample_labels,
            switch_time_s=float('inf'),
            attended_ear=attended_ear,
            first_attended_side=attended_ear,
        ))

    return trials



_ENVELOPE_INTERMEDIATE_FS = 1000


def _compute_wav_envelope(
    wav_path: Path,
    power: float = 0.6,
) -> tuple[np.ndarray, float]:
    from scipy.io import wavfile as wavio
    fs, audio = wavio.read(str(wav_path))
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    mx = np.abs(audio).max()
    if mx > 0:
        audio /= mx

    if int(fs) != _ENVELOPE_INTERMEDIATE_FS:
        audio = resample_to_fs(
            audio[:, None], source_fs=float(fs), target_fs=_ENVELOPE_INTERMEDIATE_FS
        )[:, 0]

    from scipy.signal import hilbert
    env = np.abs(hilbert(audio)) ** power
    return env, float(_ENVELOPE_INTERMEDIATE_FS)


def _get_dry_wav_name(stimuli_name: str) -> str:
    stem = Path(stimuli_name).stem
    stem = re.sub(r'_(dry|HRTF|hrtf)$', '', stem, flags=re.IGNORECASE)
    return f"{stem}_dry.wav"


def _load_kul_raw(
    mat_path: Path,
    stimuli_dir: Path,
    preprocess_cfg: SignalPreprocessConfig,
    max_trials: int = KUL_MAX_TRIALS,
) -> list[KulTrial]:
    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    trials_arr = np.asarray(raw['trials'], dtype=object)

    trials: list[KulTrial] = []
    n_load = min(max_trials, trials_arr.shape[0])

    _env_cache: dict[str, tuple[np.ndarray, float]] = {}

    def _get_envelope(wav_path: Path) -> tuple[np.ndarray, float]:
        key = str(wav_path)
        if key not in _env_cache:
            _env_cache[key] = _compute_wav_envelope(wav_path)
        return _env_cache[key]

    for i in range(n_load):
        t = trials_arr[i]

        eeg = np.asarray(t.RawData.EegData, dtype=np.float64)
        fs  = int(_scalar(t.FileHeader.SampleRate))

        attended_ear = _normalize_attended_ear(t.attended_ear)
        trial_id     = int(_scalar(t.TrialID))
        condition    = str(t.condition).strip()
        experiment   = int(_scalar(t.experiment))
        subject_id   = str(t.subject).strip()

        stimuli = np.asarray(t.stimuli, dtype=object).ravel()
        left_name  = str(stimuli[0]).strip()
        right_name = str(stimuli[1]).strip()

        left_dry  = stimuli_dir / _get_dry_wav_name(left_name)
        right_dry = stimuli_dir / _get_dry_wav_name(right_name)

        if not left_dry.exists():
            left_dry = stimuli_dir / left_name
        if not right_dry.exists():
            right_dry = stimuli_dir / right_name

        if not left_dry.exists() or not right_dry.exists():
            warnings.warn(
                f"[kul] {mat_path.name} trial {i + 1}: "
                f"wav file not found — left={left_dry.name}, right={right_dry.name}. "
                "Skipping trial.",
                stacklevel=2,
            )
            continue

        left_env_raw, audio_fs_l = _get_envelope(left_dry)
        right_env_raw, audio_fs_r = _get_envelope(right_dry)

        left_at_eeg  = resample_to_fs(left_env_raw[:, None],  audio_fs_l, fs)[:, 0]
        right_at_eeg = resample_to_fs(right_env_raw[:, None], audio_fs_r, fs)[:, 0]

        eeg_proc, left_proc, right_proc, out_fs = preprocess_trial_signals(
            eeg=eeg,
            left_env=left_at_eeg,
            right_env=right_at_eeg,
            source_fs=float(fs),
            cfg=preprocess_cfg,
        )

        sample_labels = _build_kul_labels(eeg_proc.shape[0], attended_ear)

        trials.append(KulTrial(
            subject_id=subject_id,
            trial_idx=i,
            trial_id=trial_id,
            condition_id=condition,
            experiment=experiment,
            fs=out_fs,
            eeg=eeg_proc,
            left_env=left_proc,
            right_env=right_proc,
            sample_labels=sample_labels,
            switch_time_s=float('inf'),
            attended_ear=attended_ear,
            first_attended_side=attended_ear,
        ))

    return trials



def load_kul_subject_file(
    mat_path: Path,
    preprocess_cfg: SignalPreprocessConfig,
    max_trials: int = KUL_MAX_TRIALS,
    is_preprocessed: bool = True,
    stimuli_dir: Path | None = None,
) -> list[KulTrial]:
    mat_path = Path(mat_path)
    if is_preprocessed:
        return _load_kul_preprocessed(mat_path, preprocess_cfg, max_trials)
    else:
        if stimuli_dir is None:
            stimuli_dir = mat_path.parent / 'stimuli'
        return _load_kul_raw(mat_path, Path(stimuli_dir), preprocess_cfg, max_trials)

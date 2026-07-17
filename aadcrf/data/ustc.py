
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from aadcrf.preprocess.signal import SignalPreprocessConfig, preprocess_trial_signals

LEFT_STATE  = 0
RIGHT_STATE = 1

USTC_MAX_TRIALS: int | None = None

USTC_EXCLUDED: frozenset[tuple[int, int]] = frozenset({
    (13, 16),
    (14, 13),
    (16,  4),
    (17, 18),
})



@dataclass
class UstcTrial:
    subject_id:          str
    trial_idx:           int
    trial_id:            int
    fs:                  int
    eeg:                 np.ndarray
    left_env:            np.ndarray
    right_env:           np.ndarray
    sample_labels:       np.ndarray
    switch_time_s:       float = float("inf")
    attended_ear:        str   = ""
    first_attended_side: str   = ""



def _subject_num_from_id(subject_id: str) -> int:
    m = re.search(r"\d+", subject_id)
    return int(m.group()) if m else -1


def _build_labels(length: int, attended_ear: str) -> np.ndarray:
    label = LEFT_STATE if attended_ear == "L" else RIGHT_STATE
    return np.full(length, label, dtype=np.int64)


def _normalize_ear(val: object) -> str:
    s = str(val).strip().upper()
    if s.startswith("L"):
        return "L"
    if s.startswith("R"):
        return "R"
    raise ValueError(f"无法识别的注意耳朵值: {val!r}")


def _scalar(x: object) -> int | float:
    return np.asarray(x).flat[0]


def _load_cell_element(cell_arr: np.ndarray, idx: int) -> np.ndarray:
    elem = cell_arr[idx]
    if isinstance(elem, np.ndarray) and elem.dtype == object:
        elem = elem.flat[0]
    return np.asarray(elem, dtype=np.float64)



def discover_ustc_subject_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    prep_dir = dataset_dir / "preprocessed_data"
    if not prep_dir.is_dir():
        raise FileNotFoundError(
            f"预处理数据目录不存在: {prep_dir}\n"
            "请先运行 preprocess_ustc.py 生成 preprocessed_data/s*.mat 文件。"
        )
    files = sorted(prep_dir.glob("s*.mat"),
                   key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not files:
        raise FileNotFoundError(f"未在 {prep_dir} 中找到 s*.mat 文件")
    return files


def load_ustc_subject_file(
    mat_path: str | Path,
    preprocess_cfg: SignalPreprocessConfig,
    max_trials: int | None = USTC_MAX_TRIALS,
    excluded: frozenset[tuple[int, int]] = USTC_EXCLUDED,
) -> list[UstcTrial]:
    mat_path = Path(mat_path)
    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    subject_id = str(raw["subject_id"]).strip()
    fs_raw     = int(_scalar(raw["fs"]))
    sub_num    = _subject_num_from_id(subject_id)

    attended_lr = np.asarray(raw["attended_lr"]).ravel()
    trial_ids   = np.asarray(raw["trial_id"],  dtype=np.int32).ravel()

    eeg_cell   = np.asarray(raw["eeg"],       dtype=object).ravel()
    lenv_cell  = np.asarray(raw["left_env"],  dtype=object).ravel()
    renv_cell  = np.asarray(raw["right_env"], dtype=object).ravel()

    n_stored = len(trial_ids)
    if max_trials is not None:
        n_stored = min(n_stored, int(max_trials))

    trials: list[UstcTrial] = []
    for i in range(n_stored):
        t_id = int(trial_ids[i])

        if (sub_num, t_id) in excluded:
            continue

        eeg      = _load_cell_element(eeg_cell,  i)
        left_env = _load_cell_element(lenv_cell, i)
        right_env= _load_cell_element(renv_cell, i)

        left_env  = left_env.ravel()
        right_env = right_env.ravel()

        eeg_proc, lenv_proc, renv_proc, out_fs = preprocess_trial_signals(
            eeg=eeg,
            left_env=left_env,
            right_env=right_env,
            source_fs=float(fs_raw),
            cfg=preprocess_cfg,
        )

        att_lr   = int(attended_lr[i])
        att_ear  = "R" if att_lr == 1 else "L"
        labels   = _build_labels(eeg_proc.shape[0], att_ear)

        trials.append(UstcTrial(
            subject_id=subject_id,
            trial_idx=len(trials),
            trial_id=t_id,
            fs=out_fs,
            eeg=eeg_proc,
            left_env=lenv_proc,
            right_env=renv_proc,
            sample_labels=labels,
            switch_time_s=float("inf"),
            attended_ear=att_ear,
            first_attended_side=att_ear,
        ))

    return trials

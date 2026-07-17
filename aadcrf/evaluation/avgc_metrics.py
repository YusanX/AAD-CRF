
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrialMetrics:
    steady_state_accuracy: float
    switch_detection_time_s: float
    switch_detected: bool


def compute_trial_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    window_starts_s: np.ndarray,
    switch_time_s: float,
    window_s: float,
) -> TrialMetrics:
    if pred_labels.shape != true_labels.shape:
        raise ValueError("pred_labels and true_labels must have same shape.")
    if pred_labels.size == 0:
        return TrialMetrics(
            steady_state_accuracy=0.0,
            switch_detection_time_s=0.0,
            switch_detected=False,
        )

    switch_idx = int(np.searchsorted(window_starts_s, switch_time_s, side="left"))
    switch_idx = int(np.clip(switch_idx, 0, pred_labels.size - 1))

    new_state = 1 - int(true_labels[0])

    detected_idx = None
    for i in range(switch_idx, pred_labels.size):
        if int(pred_labels[i]) == new_state:
            detected_idx = i
            break

    if detected_idx is None:
        detected_idx = pred_labels.size - 1
        switch_detected = False
    else:
        switch_detected = True

    idx_pre = np.arange(0, switch_idx, dtype=np.int64)
    idx_post = np.arange(detected_idx, pred_labels.size, dtype=np.int64)
    eval_idx = np.unique(np.concatenate([idx_pre, idx_post]))
    if eval_idx.size == 0:
        steady_acc = float(np.mean(pred_labels == true_labels))
    else:
        steady_acc = float(np.mean(pred_labels[eval_idx] == true_labels[eval_idx]))

    switch_detection_time_s = abs((detected_idx - switch_idx) * window_s)
    return TrialMetrics(
        steady_state_accuracy=steady_acc,
        switch_detection_time_s=float(switch_detection_time_s),
        switch_detected=switch_detected,
    )


def summarize_metrics(metrics: list[TrialMetrics]) -> dict[str, float]:
    if not metrics:
        return {
            "steady_state_accuracy_mean": 0.0,
            "switch_detection_time_s_mean": 0.0,
            "switch_detect_rate": 0.0,
        }
    return {
        "steady_state_accuracy_mean": float(
            np.mean([m.steady_state_accuracy for m in metrics])
        ),
        "switch_detection_time_s_mean": float(
            np.mean([m.switch_detection_time_s for m in metrics])
        ),
        "switch_detect_rate": float(
            np.mean([1.0 if m.switch_detected else 0.0 for m in metrics])
        ),
    }

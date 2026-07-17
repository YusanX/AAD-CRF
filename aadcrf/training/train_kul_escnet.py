
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aadcrf.training.train_avgc_escnet import (
    ExperimentConfig,
    TrainConfig,
    WindowConfig,
    _ensure_dir,
    _log,
    _resolve_device,
    _train_fold,
    build_trial_windows,
    _run_trial_forward,
)
from aadcrf.data.kul import (
    KulTrial,
    KUL_MAX_TRIALS,
    discover_kul_subject_files,
    load_kul_subject_file,
)
from aadcrf.training.crf import hmm_forward_backward_np, hmm_forward_np
from aadcrf.models.escnet import ESCNetConfig
from aadcrf.preprocess.signal import SignalPreprocessConfig



def _expand_kul_subject_tokens(value: str) -> set[str]:
    s = str(value).strip().lower()
    if not s:
        return set()
    tokens = {s}
    m = re.search(r'(?:sub|s)(\d+)', s)
    if not m and s.isdigit():
        m_n = int(s)
    else:
        m_n = int(m.group(1)) if m else None

    if m_n is not None:
        n = m_n
        tokens.update({
            str(n), f"{n:02d}",
            f"sub{n}", f"sub{n:02d}",
            f"s{n}", f"s{n:02d}",
        })
    return tokens


def _normalize_kul_subject_selector(
    subject: Optional[list[str]],
) -> Optional[set[str]]:
    if subject is None:
        return None
    selector: set[str] = set()
    for s in subject:
        selector.update(_expand_kul_subject_tokens(s))
    return selector


def _kul_subject_matches(
    selector: Optional[set[str]],
    file_path: Path,
    subject_id: str,
) -> bool:
    if selector is None:
        return True
    candidates: set[str] = set()
    candidates.update(_expand_kul_subject_tokens(file_path.stem))
    candidates.update(_expand_kul_subject_tokens(subject_id))
    return len(candidates & selector) > 0



def load_kul_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    signal_cfg  = SignalPreprocessConfig(**raw["signal"])
    tracker_cfg = ESCNetConfig(**raw["tracker"])
    window_cfg  = WindowConfig(**raw["window"])
    train_cfg   = TrainConfig(**raw["train"])
    exp         = raw["experiment"]

    subject_raw = exp.get("subject")
    if subject_raw is None:
        subject_sel = None
    elif isinstance(subject_raw, (list, tuple, set)):
        subject_sel = [str(v) for v in subject_raw]
    else:
        subject_sel = [str(subject_raw)]

    trial_raw = exp.get("trial")
    if trial_raw is None:
        trial_sel = None
    elif isinstance(trial_raw, (list, tuple, set)):
        trial_sel = [int(v) for v in trial_raw]
    else:
        trial_sel = [int(trial_raw)]

    return ExperimentConfig(
        dataset_dir=raw["dataset"]["kul_dir"],
        output_dir=raw["output"]["dir"],
        signal=signal_cfg,
        tracker=tracker_cfg,
        window=window_cfg,
        train=train_cfg,
        p_switch_init=float(exp.get("p_switch_init", 0.001)),
        learn_p_switch=bool(exp.get("learn_p_switch", True)),
        device=str(exp.get("device", "auto")),
        max_subjects=exp.get("max_subjects"),
        max_trials_per_subject=exp.get("max_trials_per_subject", KUL_MAX_TRIALS),
        subject=subject_sel,
        trial=trial_sel,
        seed=int(exp.get("seed", 42)),
    )



@dataclass
class KulFoldResult:
    subject_id: str
    trial_idx: int
    attended_ear: str
    raw_accuracy: float
    hmm_causal_accuracy: float
    hmm_fb_accuracy: float
    learned_p_switch: float


def _evaluate_kul_trial(
    trial: KulTrial,
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    p_switch: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    windows = build_trial_windows(trial, cfg.window.window_s, cfg.window.stride_s)

    model.eval()
    with torch.no_grad():
        logits = _run_trial_forward(model, windows, device, batch_size=0)

    log_emissions = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)

    post_causal = hmm_forward_np(log_emissions, p_switch)
    post_fb     = hmm_forward_backward_np(log_emissions, p_switch)

    raw_pred    = np.argmax(log_emissions, axis=1).astype(np.int64)
    hmm_c_pred  = np.argmax(post_causal,  axis=1).astype(np.int64)
    hmm_fb_pred = np.argmax(post_fb,      axis=1).astype(np.int64)

    true_labels = windows["labels"]
    raw_acc    = float(np.mean(raw_pred    == true_labels))
    hmm_c_acc  = float(np.mean(hmm_c_pred  == true_labels))
    hmm_fb_acc = float(np.mean(hmm_fb_pred == true_labels))

    return raw_acc, hmm_c_acc, hmm_fb_acc, post_causal, post_fb



def _save_fold_json(
    fold_dir: Path,
    fold: KulFoldResult,
    cfg: ExperimentConfig,
) -> None:
    doc = {
        "config": {
            "dataset_dir": cfg.dataset_dir,
            "signal": {
                "low_hz": cfg.signal.low_hz,
                "high_hz": cfg.signal.high_hz,
                "target_fs": cfg.signal.target_fs,
                "rereference": cfg.signal.rereference,
            },
            "tracker": {
                "n_eeg_channels": cfg.tracker.n_eeg_channels,
                "n_time_steps": cfg.tracker.n_time_steps,
                "feature_dim": cfg.tracker.feature_dim,
                "eeg_temporal_kernel": cfg.tracker.eeg_temporal_kernel,
                "env_conv1_kernel": cfg.tracker.env_conv1_kernel,
                "env_conv2_kernel": cfg.tracker.env_conv2_kernel,
                "dropout": cfg.tracker.dropout,
            },
            "window": {
                "window_s": cfg.window.window_s,
                "stride_s": cfg.window.stride_s,
            },
            "train": {
                "epochs": cfg.train.epochs,
                "lr": cfg.train.lr,
                "lambda_crf": cfg.train.lambda_crf,
                "lambda_ce": cfg.train.lambda_ce,
                "warmup_crf_epoch": cfg.train.warmup_crf_epoch,
            },
            "experiment": {
                "p_switch_init": cfg.p_switch_init,
                "learn_p_switch": cfg.learn_p_switch,
                "device": cfg.device,
                "seed": cfg.seed,
            },
        },
        "fold": {
            "subject_id": fold.subject_id,
            "trial_idx": fold.trial_idx,
            "attended_ear": fold.attended_ear,
            "learned_p_switch": fold.learned_p_switch,
            "raw_accuracy": fold.raw_accuracy,
            "hmm_causal_accuracy": fold.hmm_causal_accuracy,
            "hmm_fb_accuracy": fold.hmm_fb_accuracy,
        },
    }
    (fold_dir / "result.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_kul_results(
    output_dir: Path,
    folds: list[KulFoldResult],
) -> dict[str, float]:
    rows = [
        "subject_id,trial_idx,attended_ear,"
        "raw_acc,hmm_causal_acc,hmm_fb_acc,learned_p_switch"
    ]
    for f in folds:
        rows.append(",".join([
            f.subject_id,
            str(f.trial_idx),
            f.attended_ear,
            f"{f.raw_accuracy:.6f}",
            f"{f.hmm_causal_accuracy:.6f}",
            f"{f.hmm_fb_accuracy:.6f}",
            f"{f.learned_p_switch:.6f}",
        ]))
    (output_dir / "fold_metrics.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    summary = {
        "raw_accuracy_mean":        float(np.mean([f.raw_accuracy        for f in folds])),
        "hmm_causal_accuracy_mean": float(np.mean([f.hmm_causal_accuracy for f in folds])),
        "hmm_fb_accuracy_mean":     float(np.mean([f.hmm_fb_accuracy     for f in folds])),
        "p_switch_learned_mean":    float(np.mean([f.learned_p_switch    for f in folds])),
        "n_folds":                  len(folds),
    }
    (output_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    return summary



def run_kul_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device     = _resolve_device(cfg.device)
    output_dir = _ensure_dir(cfg.output_dir)

    _log(f"[Init] device={device}  output_dir={output_dir}")
    _log(
        f"[Init] window={cfg.window.window_s}s  "
        f"p_switch_init={cfg.p_switch_init}  learn_p_switch={cfg.learn_p_switch}  "
        f"epochs={cfg.train.epochs}  warmup_crf={cfg.train.warmup_crf_epoch}  "
        f"lr={cfg.train.lr}  λ_crf={cfg.train.lambda_crf}  λ_ce={cfg.train.lambda_ce}"
    )
    _log(
        f"[Init] subject_filter={cfg.subject if cfg.subject is not None else 'ALL'}  "
        f"test_trial_filter={cfg.trial if cfg.trial is not None else 'ALL (LOTO)'}"
    )

    subject_files, is_preprocessed = discover_kul_subject_files(cfg.dataset_dir)
    _log(
        f"[Data] Found {len(subject_files)} subject files in {cfg.dataset_dir} "
        f"({'preprocessed' if is_preprocessed else 'raw'} format)"
    )

    if cfg.max_subjects is not None:
        subject_files = subject_files[: int(cfg.max_subjects)]

    max_t = int(cfg.max_trials_per_subject) if cfg.max_trials_per_subject else KUL_MAX_TRIALS

    subject_selector = _normalize_kul_subject_selector(cfg.subject)
    all_trials: dict[str, list[KulTrial]] = {}
    selected_subject_count = 0
    skipped_by_subject     = 0

    for i, sf in enumerate(subject_files, 1):
        if not _kul_subject_matches(subject_selector, sf, sf.stem):
            skipped_by_subject += 1
            _log(f"[Data] ({i}/{len(subject_files)}) {sf.name} skipped by subject filter")
            continue

        _log(f"[Data] ({i}/{len(subject_files)}) Loading {sf.name} …")

        trials = load_kul_subject_file(
            mat_path=sf,
            preprocess_cfg=cfg.signal,
            max_trials=max_t,
            is_preprocessed=is_preprocessed,
        )

        sid = trials[0].subject_id if trials else sf.stem
        selected_subject_count += 1

        if not trials:
            _log(f"[Data] Subject={sid} has no trials after loading; skipped")
            continue

        all_trials[sid] = trials
        _log(f"[Data] Subject={sid}  trials={len(trials)}")

    if subject_selector is not None and selected_subject_count == 0:
        raise ValueError(
            f"No subject matched experiment.subject={cfg.subject!r}. "
            "Check subject IDs in config (expected format: 'S1' … 'S16')."
        )
    if not all_trials:
        raise ValueError(
            "No trials available after applying subject filter/debug limits. "
            "Check experiment.subject and max_trials_per_subject."
        )
    if skipped_by_subject > 0:
        _log(f"[Data] Subject filter skipped {skipped_by_subject} subject file(s)")

    fold_results: list[KulFoldResult] = []
    subject_ids = sorted(all_trials.keys())

    for si, sid in enumerate(subject_ids, 1):
        trials = all_trials[sid]
        _log(f"\n[Subject] ({si}/{len(subject_ids)}) {sid}  ({len(trials)} trials)")

        fs = trials[0].fs

        if cfg.trial is None:
            test_indices = list(range(len(trials)))
        else:
            test_indices = sorted(
                {int(i) for i in cfg.trial if 0 <= int(i) < len(trials)}
            )
            missing = sorted(
                {int(i) for i in cfg.trial if int(i) < 0 or int(i) >= len(trials)}
            )
            if missing:
                _log(f"[Subject] {sid} ignoring out-of-range test trials: {missing}")
            if not test_indices:
                _log(f"[Subject] {sid} has no valid test trials; skipped")
                continue

        for test_idx in test_indices:
            train_trials = [t for i, t in enumerate(trials) if i != test_idx]
            test_trial   = trials[test_idx]

            _log(
                f"[Fold] {sid} test_trial={test_idx} ({test_trial.attended_ear})  "
                f"train={len(train_trials)} trials"
            )

            model, learned_p_sw = _train_fold(train_trials, cfg, device, fs)

            fold_dir = _ensure_dir(output_dir / f"{sid}_trial{test_idx:02d}")
            torch.save(model.state_dict(), fold_dir / "checkpoint.pt")
            (fold_dir / "p_switch.txt").write_text(
                f"{learned_p_sw:.8f}\n", encoding="utf-8"
            )

            raw_acc, hmm_c_acc, hmm_fb_acc, post_causal, post_fb = _evaluate_kul_trial(
                test_trial, model, cfg, device, learned_p_sw
            )
            np.save(fold_dir / "posterior_causal.npy", post_causal)
            np.save(fold_dir / "posterior_fb.npy",     post_fb)

            fold_result = KulFoldResult(
                subject_id=sid,
                trial_idx=test_idx,
                attended_ear=test_trial.attended_ear,
                raw_accuracy=raw_acc,
                hmm_causal_accuracy=hmm_c_acc,
                hmm_fb_accuracy=hmm_fb_acc,
                learned_p_switch=learned_p_sw,
            )
            fold_results.append(fold_result)
            _save_fold_json(fold_dir, fold_result, cfg)

            _log(
                f"[Fold] DONE  "
                f"raw={raw_acc:.4f}  "
                f"hmm_causal={hmm_c_acc:.4f}  "
                f"hmm_fb={hmm_fb_acc:.4f}  "
                f"p_sw={learned_p_sw:.5f}"
            )

    summary = _save_kul_results(output_dir, fold_results)

    _log("\n[Done] ═══════════════════════════════")
    _log(
        f"[Done] raw          acc={summary['raw_accuracy_mean']:.4f}"
    )
    _log(
        f"[Done] hmm_causal   acc={summary['hmm_causal_accuracy_mean']:.4f}"
    )
    _log(
        f"[Done] hmm_fb       acc={summary['hmm_fb_accuracy_mean']:.4f}"
    )
    _log(f"[Done] p_switch_learned_mean={summary['p_switch_learned_mean']:.5f}")
    return summary

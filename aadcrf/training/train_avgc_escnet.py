
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aadcrf.data.avgc import AvgcTrial, discover_avgc_subject_files, load_avgc_subject_file
from aadcrf.evaluation.avgc_metrics import TrialMetrics, compute_trial_metrics, summarize_metrics
from aadcrf.models.escnet import ESCNet, ESCNetConfig
from aadcrf.training.crf import (
    build_log_trans,
    crf_nll_loss,
    hmm_forward_backward_np,
    hmm_forward_np,
    hmm_log_likelihood,
)
from aadcrf.preprocess.signal import SignalPreprocessConfig



@dataclass
class WindowConfig:
    window_s: float = 1.0
    stride_s: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_crf: float = 1.0
    lambda_ce: float = 0.5
    grad_clip: float = 1.0
    batch_size: int = 0
    scheduler: str = "cosine"
    warmup_crf_epoch: int = 15


@dataclass
class ExperimentConfig:
    dataset_dir: str
    output_dir: str
    signal: SignalPreprocessConfig
    tracker: ESCNetConfig
    window: WindowConfig
    train: TrainConfig
    p_switch_init: float = 0.001
    learn_p_switch: bool = True
    device: str = "auto"
    max_subjects: Optional[int] = None
    max_trials_per_subject: Optional[int] = None
    subject: Optional[list[str]] = None
    trial: Optional[list[int]] = None
    seed: int = 42



def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    signal_cfg = SignalPreprocessConfig(**raw["signal"])
    tracker_cfg = ESCNetConfig(**raw["tracker"])
    window_cfg = WindowConfig(**raw["window"])
    train_cfg = TrainConfig(**raw["train"])
    exp = raw["experiment"]

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
        dataset_dir=raw["dataset"]["avgc_dir"],
        output_dir=raw["output"]["dir"],
        signal=signal_cfg,
        tracker=tracker_cfg,
        window=window_cfg,
        train=train_cfg,
        p_switch_init=float(exp.get("p_switch_init", 0.001)),
        learn_p_switch=bool(exp.get("learn_p_switch", True)),
        device=str(exp.get("device", "auto")),
        max_subjects=exp.get("max_subjects"),
        max_trials_per_subject=exp.get("max_trials_per_subject"),
        subject=subject_sel,
        trial=trial_sel,
        seed=int(exp.get("seed", 42)),
    )



def build_trial_windows(
    trial: AvgcTrial,
    window_s: float,
    stride_s: float,
) -> dict[str, np.ndarray]:
    fs = trial.fs
    win_size = int(round(window_s * fs))
    stride = int(round(stride_s * fs))

    eeg_list, left_list, right_list, labels, starts = [], [], [], [], []
    max_start = trial.eeg.shape[0] - win_size
    if max_start < 0:
        raise ValueError(
            f"Trial shorter than one window ({trial.eeg.shape[0]} < {win_size})"
        )

    for s in range(0, max_start + 1, stride):
        e = s + win_size
        lbl = int(np.round(trial.sample_labels[s:e].mean()))
        eeg_list.append(trial.eeg[s:e].T.astype(np.float32))
        left_list.append(trial.left_env[s:e].astype(np.float32))
        right_list.append(trial.right_env[s:e].astype(np.float32))
        labels.append(lbl)
        starts.append(s / fs)

    return {
        "eeg": np.stack(eeg_list),
        "left_env": np.stack(left_list),
        "right_env": np.stack(right_list),
        "labels": np.asarray(labels, np.int64),
        "window_starts_s": np.asarray(starts, np.float64),
        "switch_time_s": np.float64(trial.switch_time_s),
    }


def _to_device(
    windows: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "eeg":       torch.from_numpy(windows["eeg"]).to(device),
        "left_env":  torch.from_numpy(windows["left_env"]).to(device),
        "right_env": torch.from_numpy(windows["right_env"]).to(device),
        "labels":    torch.from_numpy(windows["labels"]).to(device),
    }


def _resolve_device(cfg_str: str) -> torch.device:
    if cfg_str == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        for idx in range(torch.cuda.device_count()):
            candidate = torch.device(f"cuda:{idx}")
            try:
                torch.zeros(1, device=candidate)
                _log(f"[Device] Auto-selected {candidate}")
                return candidate
            except RuntimeError as exc:
                _log(f"[Device] cuda:{idx} failed ({exc}), trying next …")
        _log("[Device] All CUDA devices failed — falling back to CPU.")
        return torch.device("cpu")

    device = torch.device(cfg_str)
    if device.type == "cuda":
        try:
            torch.zeros(1, device=device)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Requested device '{cfg_str}' is not usable: {exc}"
            ) from exc
    return device


def _expand_subject_tokens(value: str) -> set[str]:
    s = str(value).strip().lower()
    if not s:
        return set()

    tokens = {s}
    m = re.search(r"sub(\d+)", s)
    if m:
        n = int(m.group(1))
        tokens.update({str(n), f"{n:02d}", f"sub{n}", f"sub{n:02d}"})
    elif s.isdigit():
        n = int(s)
        tokens.update({str(n), f"{n:02d}", f"sub{n}", f"sub{n:02d}"})
    return tokens


def _normalize_subject_selector(subject: Optional[list[str]]) -> Optional[set[str]]:
    if subject is None:
        return None
    selector: set[str] = set()
    for s in subject:
        selector.update(_expand_subject_tokens(s))
    return selector


def _subject_matches(selector: Optional[set[str]], file_path: Path, subject_id: str) -> bool:
    if selector is None:
        return True
    candidates: set[str] = set()
    candidates.update(_expand_subject_tokens(file_path.stem))
    candidates.update(_expand_subject_tokens(subject_id))
    return len(candidates & selector) > 0



def build_model(cfg: ExperimentConfig, fs: int) -> nn.Module:
    tracker_cfg = ESCNetConfig(
        n_eeg_channels=cfg.tracker.n_eeg_channels,
        n_time_steps=int(round(cfg.window.window_s * fs)),
        feature_dim=cfg.tracker.feature_dim,
        eeg_hidden_dim=cfg.tracker.eeg_hidden_dim,
        env_hidden_dim=cfg.tracker.env_hidden_dim,
        temperature=cfg.tracker.temperature,
        eeg_temporal_kernel=cfg.tracker.eeg_temporal_kernel,
        env_conv1_kernel=cfg.tracker.env_conv1_kernel,
        env_conv2_kernel=cfg.tracker.env_conv2_kernel,
        dropout=cfg.tracker.dropout,
        use_multiscale=cfg.tracker.use_multiscale,
        eeg_ms_kernels=list(cfg.tracker.eeg_ms_kernels),
        env_ms_kernels=list(cfg.tracker.env_ms_kernels),
    )
    return ESCNet(tracker_cfg)



def _run_trial_forward(
    model: nn.Module,
    windows: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tw = _to_device(windows, device)
    N = tw["eeg"].shape[0]

    if batch_size <= 0 or batch_size >= N:
        return model(tw["eeg"], tw["left_env"], tw["right_env"])

    parts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        parts.append(model(
            tw["eeg"][start:end],
            tw["left_env"][start:end],
            tw["right_env"][start:end],
        ))
    return torch.cat(parts, dim=0)



def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_trans: torch.Tensor,
    lambda_crf: float,
    lambda_ce: float,
    phase: str,
) -> tuple[torch.Tensor, float, float]:
    N = logits.shape[0]
    log_emissions = F.log_softmax(logits, dim=-1)
    loss_ce = F.cross_entropy(logits, labels)

    if phase == "warmup":
        return loss_ce, 0.0, float(loss_ce.detach())

    loss_crf = crf_nll_loss(log_emissions, labels, log_trans)

    total = lambda_crf * loss_crf + lambda_ce * loss_ce
    return total, float(loss_crf.detach()), float(loss_ce.detach())



def _train_fold(
    train_trials: list[AvgcTrial],
    cfg: ExperimentConfig,
    device: torch.device,
    fs: int,
) -> tuple[nn.Module, float]:
    model = build_model(cfg, fs).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(
        f"  [Model] type=ESCNet  params={total_params:,}  trainable={trainable_params:,}"
        f"  multiscale={cfg.tracker.use_multiscale}"
    )

    p0 = float(np.clip(cfg.p_switch_init, 1e-6, 0.4999))
    p_switch_logit = nn.Parameter(
        torch.tensor(np.log(2 * p0 / (1 - 2 * p0)), dtype=torch.float32, device=device)
    )

    if cfg.learn_p_switch:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + [p_switch_logit],
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    total_steps = cfg.train.epochs * len(train_trials)
    if cfg.train.scheduler == "cosine":
        crf_steps = max(
            total_steps - cfg.train.warmup_crf_epoch * len(train_trials), 1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=crf_steps
        )
    else:
        scheduler = None

    train_windows = [
        build_trial_windows(t, cfg.window.window_s, cfg.window.stride_s)
        for t in train_trials
    ]

    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        phase = "warmup" if epoch < cfg.train.warmup_crf_epoch else "crf"

        epoch_crf = epoch_ce = epoch_total = 0.0
        n_seq = 0

        for windows in train_windows:
            if windows["labels"].size == 0:
                continue

            if cfg.learn_p_switch and phase == "crf":
                p_switch_val = 0.5 * torch.sigmoid(p_switch_logit)
                log_trans = build_log_trans(p_switch_val, device)
            else:
                with torch.no_grad():
                    p_init = torch.tensor(cfg.p_switch_init, device=device)
                log_trans = build_log_trans(p_init.item(), device)

            labels_t = torch.from_numpy(windows["labels"]).to(device)
            logits = _run_trial_forward(model, windows, device, cfg.train.batch_size)

            loss, l_crf, l_ce = _compute_loss(
                logits, labels_t, log_trans,
                cfg.train.lambda_crf, cfg.train.lambda_ce,
                phase,
            )

            optimizer.zero_grad()
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                if cfg.learn_p_switch and p_switch_logit.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p_switch_logit], cfg.train.grad_clip)
            optimizer.step()

            if scheduler is not None and epoch >= cfg.train.warmup_crf_epoch:
                scheduler.step()

            epoch_crf   += l_crf
            epoch_ce    += l_ce
            epoch_total += float(loss.detach())
            n_seq += 1
            step += 1

        if n_seq > 0:
            p_val = float(0.5 * torch.sigmoid(p_switch_logit).detach())
            _log(
                f"  epoch {epoch + 1:3d}/{cfg.train.epochs}  [{phase}] | "
                f"total={epoch_total/n_seq:.4f}  "
                f"crf={epoch_crf/n_seq:.4f}  ce={epoch_ce/n_seq:.4f}  "
                f"T={model.temperature.detach().item():.4f}  "
                f"p_sw={p_val:.5f}"
            )

    learned_p_switch = float(0.5 * torch.sigmoid(p_switch_logit).detach())
    return model, learned_p_switch



def _evaluate_trial(
    trial: AvgcTrial,
    model: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    p_switch: float,
) -> tuple[TrialMetrics, TrialMetrics, TrialMetrics, np.ndarray, np.ndarray]:
    windows = build_trial_windows(trial, cfg.window.window_s, cfg.window.stride_s)

    model.eval()
    with torch.no_grad():
        logits = _run_trial_forward(model, windows, device, batch_size=0)

    log_emissions = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)

    post_causal = hmm_forward_np(log_emissions, p_switch)

    post_fb = hmm_forward_backward_np(log_emissions, p_switch)

    raw_pred    = np.argmax(log_emissions, axis=1).astype(np.int64)
    hmm_c_pred  = np.argmax(post_causal, axis=1).astype(np.int64)
    hmm_fb_pred = np.argmax(post_fb, axis=1).astype(np.int64)

    true_labels = windows["labels"]
    starts      = windows["window_starts_s"]

    raw_m   = compute_trial_metrics(raw_pred,    true_labels, starts,
                                    windows["switch_time_s"], cfg.window.window_s)
    hmm_c_m = compute_trial_metrics(hmm_c_pred,  true_labels, starts,
                                    windows["switch_time_s"], cfg.window.window_s)
    hmm_fb_m= compute_trial_metrics(hmm_fb_pred, true_labels, starts,
                                    windows["switch_time_s"], cfg.window.window_s)

    return raw_m, hmm_c_m, hmm_fb_m, post_causal, post_fb



def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class FoldResult:
    subject_id: str
    trial_idx: int
    raw_metrics: TrialMetrics
    hmm_causal_metrics: TrialMetrics
    hmm_fb_metrics: TrialMetrics
    learned_p_switch: float


def _save_fold_json(
    fold_dir: Path,
    fold: FoldResult,
    cfg: ExperimentConfig,
) -> None:
    doc = {
        "config": {
            "dataset_dir": cfg.dataset_dir,
            "output_dir": cfg.output_dir,
            "signal": {
                "low_hz": cfg.signal.low_hz,
                "high_hz": cfg.signal.high_hz,
                "target_fs": cfg.signal.target_fs,
                "rereference": cfg.signal.rereference,
                "filter_order": cfg.signal.filter_order,
            },
            "tracker": {
                "n_eeg_channels": cfg.tracker.n_eeg_channels,
                "n_time_steps": cfg.tracker.n_time_steps,
                "feature_dim": cfg.tracker.feature_dim,
                "eeg_hidden_dim": cfg.tracker.eeg_hidden_dim,
                "env_hidden_dim": cfg.tracker.env_hidden_dim,
                "temperature": cfg.tracker.temperature,
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
                "weight_decay": cfg.train.weight_decay,
                "lambda_crf": cfg.train.lambda_crf,
                "lambda_ce": cfg.train.lambda_ce,
                "grad_clip": cfg.train.grad_clip,
                "batch_size": cfg.train.batch_size,
                "scheduler": cfg.train.scheduler,
                "warmup_crf_epoch": cfg.train.warmup_crf_epoch,
            },
            "experiment": {
                "p_switch_init": cfg.p_switch_init,
                "learn_p_switch": cfg.learn_p_switch,
                "device": cfg.device,
                "subject": cfg.subject,
                "trial": cfg.trial,
                "seed": cfg.seed,
            },
        },
        "fold": {
            "subject_id": fold.subject_id,
            "trial_idx": fold.trial_idx,
            "learned_p_switch": fold.learned_p_switch,
            "raw": {
                "accuracy": fold.raw_metrics.steady_state_accuracy,
                "switch_detection_time_s": fold.raw_metrics.switch_detection_time_s,
            },
            "hmm_causal": {
                "accuracy": fold.hmm_causal_metrics.steady_state_accuracy,
                "switch_detection_time_s": fold.hmm_causal_metrics.switch_detection_time_s,
            },
            "hmm_fb": {
                "accuracy": fold.hmm_fb_metrics.steady_state_accuracy,
                "switch_detection_time_s": fold.hmm_fb_metrics.switch_detection_time_s,
            },
        },
    }
    (fold_dir / "result.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_results(output_dir: Path, folds: list[FoldResult]) -> dict[str, float]:
    raw_summary    = summarize_metrics([f.raw_metrics         for f in folds])
    hmm_c_summary  = summarize_metrics([f.hmm_causal_metrics  for f in folds])
    hmm_fb_summary = summarize_metrics([f.hmm_fb_metrics      for f in folds])

    rows = [
        "subject_id,trial_idx,"
        "raw_acc,raw_switch_s,"
        "hmm_causal_acc,hmm_causal_switch_s,"
        "hmm_fb_acc,hmm_fb_switch_s,"
        "learned_p_switch"
    ]
    for f in folds:
        rows.append(",".join([
            f.subject_id,
            str(f.trial_idx),
            f"{f.raw_metrics.steady_state_accuracy:.6f}",
            f"{f.raw_metrics.switch_detection_time_s:.6f}",
            f"{f.hmm_causal_metrics.steady_state_accuracy:.6f}",
            f"{f.hmm_causal_metrics.switch_detection_time_s:.6f}",
            f"{f.hmm_fb_metrics.steady_state_accuracy:.6f}",
            f"{f.hmm_fb_metrics.switch_detection_time_s:.6f}",
            f"{f.learned_p_switch:.6f}",
        ]))
    (output_dir / "fold_metrics.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    summary = {
        "raw_steady_state_accuracy_mean":     raw_summary["steady_state_accuracy_mean"],
        "raw_switch_detection_time_s_mean":   raw_summary["switch_detection_time_s_mean"],
        "raw_switch_detect_rate":             raw_summary["switch_detect_rate"],
        "hmm_causal_accuracy_mean":           hmm_c_summary["steady_state_accuracy_mean"],
        "hmm_causal_switch_time_s_mean":      hmm_c_summary["switch_detection_time_s_mean"],
        "hmm_causal_switch_detect_rate":      hmm_c_summary["switch_detect_rate"],
        "hmm_fb_accuracy_mean":               hmm_fb_summary["steady_state_accuracy_mean"],
        "hmm_fb_switch_time_s_mean":          hmm_fb_summary["switch_detection_time_s_mean"],
        "hmm_fb_switch_detect_rate":          hmm_fb_summary["switch_detect_rate"],
        "p_switch_learned_mean":              float(np.mean([f.learned_p_switch for f in folds])),
    }
    (output_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    return summary



def run_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _resolve_device(cfg.device)
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

    subject_files = discover_avgc_subject_files(cfg.dataset_dir)
    if cfg.max_subjects is not None:
        subject_files = subject_files[: int(cfg.max_subjects)]
    _log(f"[Data] Found {len(subject_files)} subjects in {cfg.dataset_dir}")

    subject_selector = _normalize_subject_selector(cfg.subject)
    all_trials: dict[str, list[AvgcTrial]] = {}
    selected_subject_count = 0
    skipped_by_subject = 0
    for i, sf in enumerate(subject_files, 1):
        _log(f"[Data] ({i}/{len(subject_files)}) Loading {sf.name} ...")
        trials = load_avgc_subject_file(sf, cfg.signal)
        sid = trials[0].subject_id if trials else sf.stem

        if not _subject_matches(subject_selector, sf, sid):
            skipped_by_subject += 1
            _log(f"[Data] Subject={sid} skipped by subject filter")
            continue
        selected_subject_count += 1

        if cfg.max_trials_per_subject is not None:
            trials = trials[: int(cfg.max_trials_per_subject)]
        if not trials:
            _log(f"[Data] Subject={sid} has no trials after filtering/debug limit; skipped")
            continue

        all_trials[sid] = trials
        _log(f"[Data] Subject={sid}  trials={len(trials)}")

    if subject_selector is not None and selected_subject_count == 0:
        raise ValueError(
            f"No subject matched experiment.subject={cfg.subject}. "
            "Please check subject IDs in config."
        )
    if not all_trials:
        raise ValueError(
            "No trials available after applying subject filter/debug limits. "
            "Please check experiment.subject and max_trials_per_subject."
        )
    if skipped_by_subject > 0:
        _log(f"[Data] Subject filter skipped {skipped_by_subject} subject files")

    fold_results: list[FoldResult] = []
    subject_ids = sorted(all_trials.keys())

    for si, sid in enumerate(subject_ids, 1):
        trials = all_trials[sid]
        _log(f"\n[Subject] ({si}/{len(subject_ids)}) {sid}  ({len(trials)} trials)")

        fs = trials[0].fs
        if cfg.trial is None:
            test_indices = list(range(len(trials)))
        else:
            test_indices = sorted({int(i) for i in cfg.trial if 0 <= int(i) < len(trials)})
            missing = sorted({int(i) for i in cfg.trial if int(i) < 0 or int(i) >= len(trials)})
            if missing:
                _log(f"[Subject] {sid} ignoring out-of-range test trials: {missing}")
            if not test_indices:
                _log(f"[Subject] {sid} has no valid test trials under filter {cfg.trial}; skipped")
                continue

        for test_idx in test_indices:
            train_trials = [t for i, t in enumerate(trials) if i != test_idx]
            test_trial   = trials[test_idx]

            _log(
                f"[Fold] {sid} test_trial={test_idx}  "
                f"train={len(train_trials)} trials"
            )

            model, learned_p_sw = _train_fold(train_trials, cfg, device, fs)

            fold_dir = _ensure_dir(output_dir / f"{sid}_trial{test_idx:02d}")
            torch.save(model.state_dict(), fold_dir / "checkpoint.pt")
            (fold_dir / "p_switch.txt").write_text(
                f"{learned_p_sw:.8f}\n", encoding="utf-8"
            )

            raw_m, hmm_c_m, hmm_fb_m, post_causal, post_fb = _evaluate_trial(
                test_trial, model, cfg, device, learned_p_sw
            )
            np.save(fold_dir / "posterior_causal.npy", post_causal)
            np.save(fold_dir / "posterior_fb.npy",     post_fb)

            fold_result = FoldResult(
                subject_id=sid,
                trial_idx=test_idx,
                raw_metrics=raw_m,
                hmm_causal_metrics=hmm_c_m,
                hmm_fb_metrics=hmm_fb_m,
                learned_p_switch=learned_p_sw,
            )
            fold_results.append(fold_result)

            _save_fold_json(fold_dir, fold_result, cfg)

            _log(
                f"[Fold] DONE  "
                f"raw={raw_m.steady_state_accuracy:.4f}  "
                f"hmm_causal={hmm_c_m.steady_state_accuracy:.4f}/{hmm_c_m.switch_detection_time_s:.1f}s  "
                f"hmm_fb={hmm_fb_m.steady_state_accuracy:.4f}/{hmm_fb_m.switch_detection_time_s:.1f}s  "
                f"p_sw={learned_p_sw:.4f}"
            )

    summary = _save_results(output_dir, fold_results)

    _log("\n[Done] ═══════════════════════════════")
    _log(
        f"[Done] raw       acc={summary['raw_steady_state_accuracy_mean']:.4f}  "
        f"switch={summary['raw_switch_detection_time_s_mean']:.2f}s  "
        f"detect={summary['raw_switch_detect_rate']:.3f}"
    )
    _log(
        f"[Done] hmm_causal  acc={summary['hmm_causal_accuracy_mean']:.4f}  "
        f"switch={summary['hmm_causal_switch_time_s_mean']:.2f}s  "
        f"detect={summary['hmm_causal_switch_detect_rate']:.3f}"
    )
    _log(
        f"[Done] hmm_fb      acc={summary['hmm_fb_accuracy_mean']:.4f}  "
        f"switch={summary['hmm_fb_switch_time_s_mean']:.2f}s  "
        f"detect={summary['hmm_fb_switch_detect_rate']:.3f}"
    )
    _log(f"[Done] p_switch_learned_mean={summary['p_switch_learned_mean']:.5f}")
    return summary


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
from aadcrf.data.ustc import (
    UstcTrial,
    USTC_EXCLUDED,
    USTC_MAX_TRIALS,
    discover_ustc_subject_files,
    load_ustc_subject_file,
)
from aadcrf.training.crf import hmm_forward_backward_np, hmm_forward_np
from aadcrf.models.escnet import ESCNetConfig
from aadcrf.preprocess.signal import SignalPreprocessConfig



def _expand_ustc_tokens(value: str) -> set[str]:
    s = str(value).strip().lower()
    tokens = {s}
    m = re.search(r"(?:sub|s)?(\d+)", s)
    if m:
        n = int(m.group(1))
        tokens.update({str(n), f"{n:02d}", f"s{n}", f"s{n:02d}",
                       f"sub{n}", f"sub{n:02d}"})
    return tokens


def _normalize_ustc_selector(subject: Optional[list[str]]) -> Optional[set[str]]:
    if subject is None:
        return None
    sel: set[str] = set()
    for s in subject:
        sel.update(_expand_ustc_tokens(s))
    return sel


def _ustc_matches(selector: Optional[set[str]], file_path: Path,
                  subject_id: str) -> bool:
    if selector is None:
        return True
    cands: set[str] = set()
    cands.update(_expand_ustc_tokens(file_path.stem))
    cands.update(_expand_ustc_tokens(subject_id))
    return bool(cands & selector)



def load_ustc_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    signal_cfg  = SignalPreprocessConfig(**raw["signal"])
    tracker_cfg = ESCNetConfig(**raw["tracker"])
    window_cfg  = WindowConfig(**raw["window"])
    train_cfg   = TrainConfig(**raw["train"])
    exp         = raw["experiment"]

    def _parse_list(v):
        if v is None:
            return None
        return [v] if not isinstance(v, (list, tuple)) else list(v)

    subject_sel = _parse_list(exp.get("subject"))
    if subject_sel is not None:
        subject_sel = [str(x) for x in subject_sel]

    trial_raw = exp.get("trial")
    trial_sel = None
    if trial_raw is not None:
        trial_sel = ([int(trial_raw)] if not isinstance(trial_raw, (list, tuple))
                     else [int(x) for x in trial_raw])

    max_t = exp.get("max_trials_per_subject", USTC_MAX_TRIALS)

    return ExperimentConfig(
        dataset_dir=raw["dataset"]["ustc_dir"],
        output_dir=raw["output"]["dir"],
        signal=signal_cfg,
        tracker=tracker_cfg,
        window=window_cfg,
        train=train_cfg,
        p_switch_init=float(exp.get("p_switch_init", 0.001)),
        learn_p_switch=bool(exp.get("learn_p_switch", True)),
        device=str(exp.get("device", "auto")),
        max_subjects=exp.get("max_subjects"),
        max_trials_per_subject=int(max_t) if max_t is not None else None,
        subject=subject_sel,
        trial=trial_sel,
        seed=int(exp.get("seed", 42)),
    )



@dataclass
class UstcFoldResult:
    subject_id:          str
    trial_idx:           int
    attended_ear:        str
    raw_accuracy:        float
    hmm_causal_accuracy: float
    hmm_fb_accuracy:     float
    learned_p_switch:    float


def _evaluate_ustc_trial(
    trial: UstcTrial,
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    p_switch: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    windows = build_trial_windows(trial, cfg.window.window_s, cfg.window.stride_s)
    model.eval()
    with torch.no_grad():
        logits = _run_trial_forward(model, windows, device, batch_size=0)
    log_e = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)

    post_c  = hmm_forward_np(log_e, p_switch)
    post_fb = hmm_forward_backward_np(log_e, p_switch)

    true  = windows["labels"]
    raw_acc   = float(np.mean(np.argmax(log_e,    axis=1) == true))
    hmm_c_acc = float(np.mean(np.argmax(post_c,   axis=1) == true))
    hmm_fb_acc= float(np.mean(np.argmax(post_fb,  axis=1) == true))
    return raw_acc, hmm_c_acc, hmm_fb_acc, post_c, post_fb



def _save_fold_json(fold_dir: Path, fold: UstcFoldResult,
                    cfg: ExperimentConfig) -> None:
    doc = {
        "config": {
            "dataset_dir": cfg.dataset_dir,
            "signal": {
                "low_hz": cfg.signal.low_hz, "high_hz": cfg.signal.high_hz,
                "target_fs": cfg.signal.target_fs, "rereference": cfg.signal.rereference,
            },
            "tracker": {
                "n_eeg_channels": cfg.tracker.n_eeg_channels,
                "feature_dim": cfg.tracker.feature_dim,
                "dropout": cfg.tracker.dropout,
            },
            "window": {"window_s": cfg.window.window_s, "stride_s": cfg.window.stride_s},
            "train": {
                "epochs": cfg.train.epochs, "lr": cfg.train.lr,
                "lambda_crf": cfg.train.lambda_crf, "lambda_ce": cfg.train.lambda_ce,
                "warmup_crf_epoch": cfg.train.warmup_crf_epoch,
            },
            "experiment": {
                "p_switch_init": cfg.p_switch_init,
                "learn_p_switch": cfg.learn_p_switch,
                "device": cfg.device, "seed": cfg.seed,
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


def _save_ustc_results(output_dir: Path,
                        folds: list[UstcFoldResult]) -> dict[str, float]:
    rows = ["subject_id,trial_idx,attended_ear,"
            "raw_acc,hmm_causal_acc,hmm_fb_acc,learned_p_switch"]
    for f in folds:
        rows.append(",".join([
            f.subject_id, str(f.trial_idx), f.attended_ear,
            f"{f.raw_accuracy:.6f}", f"{f.hmm_causal_accuracy:.6f}",
            f"{f.hmm_fb_accuracy:.6f}", f"{f.learned_p_switch:.6f}",
        ]))
    (output_dir / "fold_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

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



def run_ustc_experiment(cfg: ExperimentConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device     = _resolve_device(cfg.device)
    output_dir = _ensure_dir(cfg.output_dir)

    _log(f"[Init] device={device}  output_dir={output_dir}")
    _log(f"[Init] window={cfg.window.window_s}s  "
         f"p_switch_init={cfg.p_switch_init}  learn_p_switch={cfg.learn_p_switch}  "
         f"epochs={cfg.train.epochs}  warmup_crf={cfg.train.warmup_crf_epoch}  "
         f"lr={cfg.train.lr}  λ_crf={cfg.train.lambda_crf}  λ_ce={cfg.train.lambda_ce}")
    _log(f"[Init] subject_filter={cfg.subject or 'ALL'}  "
         f"test_trial_filter={cfg.trial or 'ALL (LOTO)'}")

    subject_files = discover_ustc_subject_files(cfg.dataset_dir)
    _log(f"[Data] 找到 {len(subject_files)} 个受试者文件于 {cfg.dataset_dir}")
    if cfg.max_subjects is not None:
        subject_files = subject_files[:int(cfg.max_subjects)]

    max_t = int(cfg.max_trials_per_subject) if cfg.max_trials_per_subject else None
    selector = _normalize_ustc_selector(cfg.subject)
    all_trials: dict[str, list[UstcTrial]] = {}
    skipped = 0

    for idx, sf in enumerate(subject_files, 1):
        if not _ustc_matches(selector, sf, sf.stem):
            skipped += 1
            continue
        _log(f"[Data] ({idx}/{len(subject_files)}) 加载 {sf.name} ...")
        trials = load_ustc_subject_file(
            mat_path=sf,
            preprocess_cfg=cfg.signal,
            max_trials=max_t,
            excluded=USTC_EXCLUDED,
        )
        sid = trials[0].subject_id if trials else sf.stem
        if not trials:
            _log(f"[Data] {sid} 无有效 trial，跳过")
            continue
        all_trials[sid] = trials
        _log(f"[Data] {sid}  trials={len(trials)}")

    if selector is not None and not all_trials:
        raise ValueError(
            f"没有受试者匹配 experiment.subject={cfg.subject!r}。"
            "请检查配置中的受试者 ID（格式示例：'s1'…'s18'）。"
        )
    if not all_trials:
        raise ValueError("过滤后无可用 trial。请检查配置。")
    if skipped:
        _log(f"[Data] subject 过滤跳过了 {skipped} 个文件")

    fold_results: list[UstcFoldResult] = []
    subject_ids = sorted(all_trials.keys(),
                         key=lambda s: int(re.search(r"\d+", s).group()))

    for si, sid in enumerate(subject_ids, 1):
        trials = all_trials[sid]
        _log(f"\n[Subject] ({si}/{len(subject_ids)}) {sid}  ({len(trials)} trials)")
        fs = trials[0].fs

        test_indices = (
            list(range(len(trials))) if cfg.trial is None
            else sorted({int(i) for i in cfg.trial if 0 <= int(i) < len(trials)})
        )
        if not test_indices:
            _log(f"[Subject] {sid} 无有效测试 trial，跳过")
            continue

        for test_idx in test_indices:
            train_trials = [t for i, t in enumerate(trials) if i != test_idx]
            test_trial   = trials[test_idx]

            _log(f"[Fold] {sid} test_trial={test_idx} ({test_trial.attended_ear})  "
                 f"train={len(train_trials)} trials")

            model, learned_p_sw = _train_fold(train_trials, cfg, device, fs)

            fold_dir = _ensure_dir(output_dir / f"{sid}_trial{test_idx:02d}")
            torch.save(model.state_dict(), fold_dir / "checkpoint.pt")
            (fold_dir / "p_switch.txt").write_text(f"{learned_p_sw:.8f}\n", encoding="utf-8")

            raw_acc, hc_acc, hfb_acc, post_c, post_fb = _evaluate_ustc_trial(
                test_trial, model, cfg, device, learned_p_sw
            )
            np.save(fold_dir / "posterior_causal.npy", post_c)
            np.save(fold_dir / "posterior_fb.npy",     post_fb)

            fold_result = UstcFoldResult(
                subject_id=sid, trial_idx=test_idx,
                attended_ear=test_trial.attended_ear,
                raw_accuracy=raw_acc,
                hmm_causal_accuracy=hc_acc,
                hmm_fb_accuracy=hfb_acc,
                learned_p_switch=learned_p_sw,
            )
            fold_results.append(fold_result)
            _save_fold_json(fold_dir, fold_result, cfg)

            _log(f"[Fold] DONE  raw={raw_acc:.4f}  "
                 f"hmm_causal={hc_acc:.4f}  hmm_fb={hfb_acc:.4f}  "
                 f"p_sw={learned_p_sw:.5f}")

    summary = _save_ustc_results(output_dir, fold_results)

    _log("\n[Done] ═══════════════════════════════")
    _log(f"[Done] raw          acc={summary['raw_accuracy_mean']:.4f}")
    _log(f"[Done] hmm_causal   acc={summary['hmm_causal_accuracy_mean']:.4f}")
    _log(f"[Done] hmm_fb       acc={summary['hmm_fb_accuracy_mean']:.4f}")
    _log(f"[Done] p_switch_learned_mean={summary['p_switch_learned_mean']:.5f}")
    return summary

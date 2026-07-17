
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aadcrf.training.train_avgc_escnet import ExperimentConfig, load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ESCNet with end-to-end CRF on AVGC AAD dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "avgc_escnet.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Limit number of subjects (for debugging).",
    )
    parser.add_argument(
        "--max-trials", type=int, default=None,
        help="Limit number of trials per subject (for debugging).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (e.g. 'cuda:0', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_subjects is not None:
        cfg.max_subjects = args.max_subjects
    if args.max_trials is not None:
        cfg.max_trials_per_subject = args.max_trials
    if args.device is not None:
        cfg.device = args.device

    summary = run_experiment(cfg)

    print("\n=== Final Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nOutputs saved to: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()

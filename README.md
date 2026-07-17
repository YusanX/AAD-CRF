# AAD-CRF

Code for **End-to-End Markov State Sequence Learning for Auditory Attention Decoding**.

The project trains auditory attention decoding (AAD) backbones as neural emissions of a two-state conditional random field (CRF). It contains experiments for AVGC, KUL, and USTC with ESCNet, AADNet, LSTM, and the attn-GRU model variants.

## CRF method and HMM-post baseline

Configuration names distinguish the two experimental methods:

- Configurations without `_post` use the proposed end-to-end CRF method. The backbone is optimized with cross-entropy warm-up followed by the joint CE and CRF objective, and the transition probability can be learned with the neural emissions.
- Configurations with `_post` reproduce the conventional post-hoc HMM method. The backbone is trained with window-level cross-entropy, then a fixed-transition HMM is applied only during inference.

For example, the following command runs the paper's proposed CRF method with ESCNet on AVGC:

```bash
python scripts/train_avgc_escnet.py --config configs/avgc_escnet.yaml
```

The corresponding conventional HMM post-processing baseline is:

```bash
python scripts/train_avgc_escnet_post.py --config configs/avgc_escnet_post.yaml
```

## Installation

Python 3.10 or newer is required.

```bash
conda create -n aadcrf python=3.10
conda activate aadcrf
pip install -r requirement.txt
```

The dependency versions are reduced from the experiment environment to packages imported by this repository. For a CUDA installation, install the matching PyTorch 2.5.1 build for the local CUDA driver if the default pip build is unsuitable.

AADNet experiments use the original AADNet implementation through `aadcrf/models/aadnet.py`. Place that implementation at `AADNet/aadnet/EnvelopeAAD.py` under the repository root before running an AADNet configuration.

## Dataset paths

Replace the dataset placeholder in each selected YAML file:

- `"<PATH_TO_AVGC>"`: directory containing `2024-AV-GC-AAD-sub*_preprocessed.mat`.
- `"<PATH_TO_KUL>"`: KUL directory containing `preprocessed_data/S*.mat`, or raw `S*.mat` files and the corresponding `stimuli/` directory.
- `"<PATH_TO_USTC>"`: USTC directory containing `preprocessed_data/s*.mat`.

Outputs are written to the relative directory under `./outputs/` specified by each configuration. The dataset and output paths can be changed independently.

## Running experiments

### AVGC

Each AVGC backbone has a dedicated CRF entry point and, where provided, a `_post` HMM baseline:

```bash
python scripts/train_avgc_escnet.py --config configs/avgc_escnet.yaml
python scripts/train_avgc_aad.py --config configs/avgc_aadnet.yaml
python scripts/train_avgc_lstm.py --config configs/avgc_lstm.yaml
python scripts/train_avgc_icassp2023.py --config configs/avgc_icassp2023.yaml
```

Use the matching `_post` script and configuration for post-hoc HMM evaluation, for example:

```bash
python scripts/train_avgc_lstm_post.py --config configs/avgc_lstm_post.yaml
```

### KUL

The unified KUL entry point selects the backbone from the top-level `model` field in the configuration:

```bash
python scripts/train_kul.py --config configs/kul_escnet.yaml
python scripts/train_kul.py --config configs/kul_aadnet.yaml
python scripts/train_kul.py --config configs/kul_lstm.yaml
python scripts/train_kul.py --config configs/kul_icassp2023.yaml
```

The available ESCNet and AADNet HMM-post baselines can also be launched through their dedicated `_post` scripts.

### USTC

The unified USTC entry point likewise reads the model from the configuration:

```bash
python scripts/train_ustc.py --config configs/ustc_escnet.yaml
python scripts/train_ustc.py --config configs/ustc_aadnet.yaml
python scripts/train_ustc.py --config configs/ustc_lstm.yaml
python scripts/train_ustc.py --config configs/ustc_icassp2023.yaml
```

All entry points accept optional `--device`, `--max-subjects`, and `--max-trials` overrides where supported. Run a script with `--help` for its complete arguments.

## Repository layout

- `aadcrf/models/`: ESCNet, AADNet adapter, LSTM, and ICASSP2023 backbones.
- `aadcrf/training/`: shared CRF/HMM algorithms and dataset-specific training pipelines.
- `aadcrf/data/`: AVGC, KUL, and USTC loaders.
- `configs/`: experiment and HMM-post configurations.
- `scripts/`: command-line entry points.

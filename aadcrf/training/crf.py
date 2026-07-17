"""Differentiable HMM and CRF utilities for training and inference."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def build_log_trans(p_switch: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(p_switch, torch.Tensor):
        p = p_switch.clamp(1e-08, 0.499999)
        A = torch.stack([torch.stack([1.0 - p, p]), torch.stack([p, 1.0 - p])])
        return A.log()
    else:
        p = float(np.clip(p_switch, 1e-08, 0.499999))
        A = torch.tensor([[1.0 - p, p], [p, 1.0 - p]], dtype=torch.float32, device=device)
        return A.log()

@torch.jit.script
def _forward_log_z(log_emissions: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor) -> torch.Tensor:
    log_alpha = log_pi + log_emissions[0]
    for t in range(1, log_emissions.shape[0]):
        log_alpha = log_emissions[t] + torch.logsumexp(log_alpha.unsqueeze(1) + log_trans, dim=0)
    return torch.logsumexp(log_alpha, dim=0)

def crf_forced_score(log_emissions: torch.Tensor, labels: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor | None=None) -> torch.Tensor:
    T = log_emissions.shape[0]
    if log_pi is None:
        log_pi = torch.full((2,), -0.6931471805599453, dtype=log_emissions.dtype, device=log_emissions.device)
    t_idx = torch.arange(T, device=log_emissions.device)
    emit_score = log_emissions[t_idx, labels].sum()
    trans_score = log_trans[labels[:-1], labels[1:]].sum()
    return log_pi[labels[0]] + emit_score + trans_score

def hmm_log_likelihood(log_emissions: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor | None=None) -> torch.Tensor:
    if log_pi is None:
        log_pi = torch.full((2,), -0.6931471805599453, dtype=log_emissions.dtype, device=log_emissions.device)
    return _forward_log_z(log_emissions, log_trans, log_pi)

def crf_nll_loss(log_emissions: torch.Tensor, labels: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor | None=None) -> torch.Tensor:
    T = log_emissions.shape[0]
    log_numerator = crf_forced_score(log_emissions, labels, log_trans, log_pi)
    log_denominator = hmm_log_likelihood(log_emissions, log_trans, log_pi)
    return -(log_numerator - log_denominator) / T

def hmm_forward_np(log_emissions: np.ndarray, p_switch: float) -> np.ndarray:
    T = log_emissions.shape[0]
    p = float(np.clip(p_switch, 1e-08, 0.499999))
    log_A = np.log(np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=np.float64))
    log_pi = np.log([0.5, 0.5])
    log_alpha = np.zeros((T, 2), dtype=np.float64)
    log_alpha[0] = log_pi + log_emissions[0].astype(np.float64)
    for t in range(1, T):
        for j in range(2):
            log_alpha[t, j] = log_emissions[t, j] + np.logaddexp(log_alpha[t - 1, 0] + log_A[0, j], log_alpha[t - 1, 1] + log_A[1, j])
    posterior = np.zeros((T, 2), dtype=np.float64)
    for t in range(T):
        z = np.logaddexp(log_alpha[t, 0], log_alpha[t, 1])
        posterior[t] = np.exp(log_alpha[t] - z)
    return posterior

def hmm_forward_backward_np(log_emissions: np.ndarray, p_switch: float) -> np.ndarray:
    T = log_emissions.shape[0]
    p = float(np.clip(p_switch, 1e-08, 0.499999))
    log_A = np.log(np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=np.float64))
    log_pi = np.log([0.5, 0.5])
    log_alpha = np.zeros((T, 2), dtype=np.float64)
    log_alpha[0] = log_pi + log_emissions[0].astype(np.float64)
    for t in range(1, T):
        for j in range(2):
            log_alpha[t, j] = log_emissions[t, j] + np.logaddexp(log_alpha[t - 1, 0] + log_A[0, j], log_alpha[t - 1, 1] + log_A[1, j])
    log_beta = np.zeros((T, 2), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        for i in range(2):
            log_beta[t, i] = np.logaddexp(log_beta[t + 1, 0] + log_A[i, 0] + log_emissions[t + 1, 0], log_beta[t + 1, 1] + log_A[i, 1] + log_emissions[t + 1, 1])
    log_gamma = log_alpha + log_beta
    posterior = np.zeros((T, 2), dtype=np.float64)
    for t in range(T):
        z = np.logaddexp(log_gamma[t, 0], log_gamma[t, 1])
        posterior[t] = np.exp(log_gamma[t] - z)
    return posterior

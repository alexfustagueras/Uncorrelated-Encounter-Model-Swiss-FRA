from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.utils import denorm_seq_to_global

# ------------------------------
# Utilities
# ------------------------------


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def _ensure_tensor(
    x: torch.Tensor | np.ndarray,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Ensure x is a tensor on the specified device with dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    t = torch.from_numpy(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


# ------------------------------
# Point forecast errors
# ------------------------------


@torch.no_grad()
def ade_fde(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Args:
        y_pred: (B, T, D>=3) predicted (global) trajectories
        y_true: (B, T, D>=3) ground-truth (global) trajectories

    Returns:
        (ade, fde): each of shape (B,)
    """
    diff = torch.linalg.norm(y_pred[..., :3] - y_true[..., :3], dim=-1)  # (B,T)
    ade = diff.mean(dim=1)  # (B,)
    fde = diff[:, -1]  # (B,)
    return ade, fde


# ------------------------------
# Proper scoring rules from samples
# ------------------------------


@torch.no_grad()
def energy_score_per_horizon(
    y_samples: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Energy Score per horizon (multivariate proper score). Lower is better.

    Args:
        y_samples: (S, B, T, D>=3) samples (global frame)
        y_true:    (B, T, D>=3)   ground truth

    Returns:
        ES: (B, T)
    """
    S, B, T, D = y_samples.shape
    Y = y_samples[..., :3]  # (S,B,T,3)
    y = y_true[..., :3]  # (B,T,3)

    # term1: 1/S sum ||Y - y||
    term1 = torch.linalg.norm(Y - y.unsqueeze(0), dim=-1).mean(dim=0)  # (B,T)

    # term2: 1/(2 S^2) sum ||Y - Y'||
    # vectorized across samples
    Y_flat = Y.permute(1, 2, 0, 3).reshape(B * T, S, 3)  # (B*T, S, 3)
    diffs = Y_flat.unsqueeze(2) - Y_flat.unsqueeze(1)  # (B*T, S, S, 3)
    term2 = torch.linalg.norm(diffs, dim=-1).mean(dim=(1, 2)).view(B, T) * 0.5
    return term1 - term2


@torch.no_grad()
def energy_score_whole_path(
    y_samples: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Energy Score on the entire path vectorized as R^{3T}. Lower is better.

    Args:
        y_samples: (S, B, T, 3)
        y_true:    (B, T, 3)

    Returns:
        ES_path: (B,)
    """
    S, B, T, _ = y_samples.shape
    Y = y_samples.reshape(S, B, 3 * T)  # (S,B,3T)
    y = y_true.reshape(B, 3 * T)  # (B,3T)

    term1 = torch.linalg.norm(Y - y.unsqueeze(0), dim=-1).mean(dim=0)  # (B,)
    diffs = Y.unsqueeze(2) - Y.unsqueeze(1)  # (S,S,B,3T)
    term2 = torch.linalg.norm(diffs, dim=-1).mean(dim=(0, 1)) * 0.5  # (B,)
    return term1 - term2


@torch.no_grad()
def crps_from_samples_scalar(
    samples: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """
    Sample-based CRPS for a scalar target.

    Args:
        samples: (S, ...) samples for one scalar at matching broadcastable shape
        truth:   (...)    truth scalar(s)

    Returns:
        CRPS with shape broadcasting to truth
    """
    # CRPS = 1/S sum |x_i - y| - 1/(2 S^2) sum |x_i - x_j|
    term1 = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    diffs = samples.unsqueeze(1) - samples.unsqueeze(0)  # (S,S,...)
    term2 = diffs.abs().mean(dim=(0, 1)) * 0.5
    return term1 - term2


@torch.no_grad()
def crps_positions(y_samples: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    CRPS per coordinate (x,y,z), per horizon and batch.

    Args:
        y_samples: (S, B, T, D>=3)
        y_true:    (B, T, D>=3)

    Returns:
        crps: (B, T, 3)
    """
    S, B, T, D = y_samples.shape
    Y = y_samples[..., :3]  # (S,B,T,3)
    y = y_true[..., :3]  # (B,T,3)
    out = torch.empty(B, T, 3, device=Y.device)
    for d in range(3):
        out[..., d] = crps_from_samples_scalar(Y[..., d], y[..., d])
    return out


# ------------------------------
# Calibration diagnostics
# ------------------------------


@torch.no_grad()
def pit_values(y_samples: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Probability Integral Transform (PIT) values per axis using a robust count-based method.
    This avoids torch.searchsorted broadcasting quirks.

    Args:
        y_samples: (S, B, T, 3)  samples (positions only)
        y_true:    (B, T, 3)     ground truth

    Returns:
        pits: (B, T, 3) in (0,1)
    """
    S, B, T, D = y_samples.shape
    D = min(3, D)
    # Sort along the sample axis to get a proper empirical CDF if needed
    Y_sorted = torch.sort(y_samples[..., :D], dim=0).values  # (S,B,T,D)

    pits = torch.empty(B, T, D, device=y_samples.device, dtype=torch.float32)
    # For each axis, count how many samples are <= truth at each (B,T)
    # rank = sum_{s} 1{Y_s <= y_true}; randomized PIT: (rank + U[0,1])/(S+1)
    for d in range(D):
        # Broadcast compare: (S,B,T) <= (B,T) -> (S,B,T)
        cmp = Y_sorted[..., d] <= y_true[..., d]  # (S,B,T) boolean
        ranks = cmp.sum(dim=0)  # (B,T), integer in [0,S]
        pits[..., d] = (
            ranks.to(torch.float32) + torch.rand_like(ranks, dtype=torch.float32)
        ) / (S + 1.0)

    return pits


@torch.no_grad()
def coverage_curve_1d(
    y_samples: torch.Tensor,
    y_true: torch.Tensor,
    alphas: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Dict[float, torch.Tensor]:
    """
    Empirical coverage for central intervals per coordinate, per horizon.

    Args:
        y_samples: (S, B, T, 3)
        y_true:    (B, T, 3)
        alphas:    desired nominal coverages

    Returns:
        dict[alpha] -> coverage tensor of shape (T, 3)
    """
    S, B, T, D = y_samples.shape
    D = min(3, D)
    Y_sorted = torch.sort(y_samples[..., :D], dim=0).values  # (S,B,T,D)
    y = y_true[..., :D]

    out: Dict[float, torch.Tensor] = {}
    for a in alphas:
        lo = int(((1 - a) / 2) * S)
        hi = int((1 - (1 - a) / 2) * S) - 1
        lo = max(0, min(lo, S - 1))
        hi = max(0, min(hi, S - 1))
        inside = (y >= Y_sorted[lo]) & (y <= Y_sorted[hi])  # (B,T,D)
        out[a] = inside.to(torch.float32).mean(dim=0)  # (T,D)
    return out


@torch.no_grad()
def mvn_coverage_ellipsoids(
    y_samples: torch.Tensor,
    y_true: torch.Tensor,
    alphas: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Dict[float, torch.Tensor]:
    """
    Approximate multivariate empirical coverage by fitting a Gaussian to samples
    at each (B,T) and counting if the truth lies within the chi-square ellipsoid.

    Args:
        y_samples: (S, B, T, 3)
        y_true:    (B, T, 3)

    Returns:
        dict[alpha] -> coverage curve over horizon (T,)
    """
    # Precompute chi-square thresholds for df=3
    # If SciPy is available, use it; otherwise, use common constants.
    try:
        from scipy.stats import chi2

        chi2_thresh = {a: float(chi2.ppf(a, df=3)) for a in alphas}
    except Exception:
        # Fallback values (df=3): 50%,80%,90%,95%
        defaults = {0.5: 2.366, 0.8: 4.642, 0.9: 6.251, 0.95: 7.815}
        chi2_thresh = {a: defaults.get(a, float("nan")) for a in alphas}

    S, B, T, D = y_samples.shape
    Y = y_samples[..., :3].detach().cpu().numpy()
    y = y_true[..., :3].detach().cpu().numpy()

    covs = {a: np.zeros(T, dtype=np.float64) for a in alphas}
    for t in range(T):
        hits = {a: 0 for a in alphas}
        for b in range(B):
            X = Y[:, b, t, :]  # (S,3)
            mu = X.mean(axis=0)
            C = np.cov(X.T) + 1e-6 * np.eye(3)
            diff = y[b, t, :] - mu
            try:
                m2 = float(diff @ np.linalg.inv(C) @ diff)
            except np.linalg.LinAlgError:
                m2 = float("inf")
            for a in alphas:
                thr = chi2_thresh[a]
                if not math.isnan(thr) and m2 <= thr:
                    hits[a] += 1
        for a in alphas:
            covs[a][t] = hits[a] / max(1, B)
    return {a: torch.from_numpy(covs[a]) for a in alphas}


# ------------------------------
# Spread / sharpness
# ------------------------------


@torch.no_grad()
def positional_spread(y_samples: torch.Tensor) -> torch.Tensor:
    """
    Spread of positional samples measured as sqrt(trace(covariance)) per (B,T).

    Args:
        y_samples: (S, B, T, 3)

    Returns:
        spread: (B, T) in same units as input positions (e.g., meters)
    """
    S, B, T, _ = y_samples.shape
    Y = y_samples[..., :3]
    mu = Y.mean(dim=0)  # (B,T,3)
    dev = Y - mu.unsqueeze(0)  # (S,B,T,3)
    cov_trace = (dev**2).sum(dim=(0, -1)) / max(1, S - 1)  # (B,T)
    return torch.sqrt(cov_trace)


# ------------------------------
# Evaluation helper
# ------------------------------


# ------------------------------
# High-level evaluator
# ------------------------------

#!/usr/bin/env python3
"""
Training utilities for trajectory prediction models.

Contains:
- Learning rate schedulers (WarmupCosine)
- Exponential Moving Average (EMA)
- Model architectures:
  * ResidualGaussianBN (single Gaussian baseline)
- Training functions for all models
"""

import math
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils.utils import make_loader

class WarmupCosine:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.last_step = -1
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.last_step += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.last_step < self.warmup:
                lr = base * (self.last_step + 1) / self.warmup
            else:
                t = (self.last_step - self.warmup) / max(
                    1, self.max_steps - self.warmup
                )
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (
                    1 + math.cos(math.pi * t)
                )
            g["lr"] = lr


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    @torch.no_grad()
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def swap_into(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        def restore():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
            self.backup = {}

        return restore

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ======================= BAYESIAN NETWORK MODELS =======================


class ResidualGaussianBN(nn.Module):
    """
    Single Gaussian Bayesian Network.
    
    Predicts residuals: delta_t = state_t - state_{t-1}
    Using a single diagonal Gaussian distribution.
    """
    
    def __init__(self, state_dim: int = 7, context_dim: int = 8, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.mean_head = nn.Linear(hidden, state_dim)
        self.log_std_head = nn.Linear(hidden, state_dim)

    def forward(self, prev_state: torch.Tensor, context: torch.Tensor):
        """
        Args:
            prev_state: (B, 7) normalized previous state
            context: (B, 8) normalized context
        Returns:
            mean: (B, 7) predicted residual mean
            log_std: (B, 7) predicted log standard deviation
        """
        x = torch.cat([prev_state, context], dim=-1)
        h = self.mlp(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=-5.0, max=3.0)
        return mean, log_std
    
    def sample(self, prev_state: torch.Tensor, context: torch.Tensor, temperature: float = 1.0):
        """Sample residual from single Gaussian."""
        mean, log_std = self.forward(prev_state, context)
        std = torch.exp(log_std) * temperature
        return mean + std * torch.randn_like(mean)
    
    def log_prob(self, delta: torch.Tensor, prev_state: torch.Tensor, context: torch.Tensor):
        """Compute log probability of observed residual."""
        mean, log_std = self.forward(prev_state, context)
        var = torch.exp(2.0 * log_std)
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (delta - mean) ** 2 / var)
        return log_prob.sum(dim=-1)  # Sum over dimensions


def load_bn_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
    model_class: type = ResidualGaussianBN,
) -> tuple[torch.nn.Module, dict | None]:
    """
    Load a BN model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on (auto-detects if None)
        model_class: ResidualGaussianBN
    
    Returns:
        (model, norm_stats) tuple
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config
    model_cfg = ckpt.get("model_cfg", {})
    if not model_cfg:
        model_cfg = {"state_dim": 7, "context_dim": 8, "hidden": 512}
    
    # Create model
    model = model_class(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    
    # Extract normalization stats
    norm_stats = ckpt.get("norm_stats", None)
    
    return model, norm_stats

@torch.no_grad()
def bn_rollout(
    model: nn.Module,
    last_hist_state: torch.Tensor,
    context: torch.Tensor,
    horizon: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Autoregressive rollout for residual-based BN.
    
    Args:
        model: ResidualGaussianBN
        last_hist_state: (B, 7) last historical state (normalized)
        context: (B, 8) context features (normalized)
        horizon: Number of future steps
        temperature: Sampling temperature (1.0 = normal, <1 = conservative, >1 = exploratory)
    
    Returns:
        futures: (B, horizon, 7) predicted trajectory (normalized, absolute states)
    """
    model.eval()
    B = last_hist_state.size(0)
    device = last_hist_state.device
    
    futures = []
    prev_state = last_hist_state  # (B, 7)
    
    for _ in range(horizon):
        # Sample residual
        delta = model.sample(prev_state, context, temperature=temperature)  # (B, 7)
        
        # Update state
        next_state = prev_state + delta
        futures.append(next_state.unsqueeze(1))  # (B, 1, 7)
        prev_state = next_state
    
    return torch.cat(futures, dim=1)  # (B, horizon, 7)

@torch.no_grad()
def sample_many_bn(model, x_hist, ctx, T_out=12, n_samples=50, temperature=1.0):
    """
    Sample multiple trajectories for a batch of histories.
    
    Args:
        model: ResidualGaussianBN
        x_hist: (B, L, 7) historical states
        ctx: (B, 8) context
        T_out: Number of future steps
        n_samples: Number of samples per case
        temperature: Sampling temperature
        
    Returns:
        samples: (S, B, T, 7)
    """
    all_samples = []
    last_hist = x_hist[:, -1, :] # (B, 7)
    for _ in range(n_samples):
        samp = bn_rollout(model, last_hist, ctx, T_out, temperature=temperature)
        all_samples.append(samp)
    return torch.stack(all_samples, dim=0)

# ======================= TRAINING FUNCTIONS =======================

def train_bn(
    train_ds,
    val_ds,
    *,
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_steps: int = 5000,
    ema_decay: float = 0.9995,
    patience: int = 20,
    hidden: int = 512,
    ckpt_path: str = "best_bn.pt",
    device: torch.device | None = None,
    norm_stats: dict | None = None,
):
    """
    Train ResidualGaussianBN (single Gaussian).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    model = ResidualGaussianBN(state_dim=7, context_dim=8, hidden=hidden).to(device)

    try:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), fused=True)
    except TypeError:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    max_steps = epochs * max(1, len(train_loader))
    sched = WarmupCosine(opt, warmup_steps=warmup_steps, max_steps=max_steps, min_lr=lr * 0.05)
    ema = EMA(model, decay=ema_decay)

    def epoch_pass(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        n = 0
        
        for xb, yb, cb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            
            # Build teacher-forced sequence
            prev = torch.cat([xb[:, -1:, :], yb[:, :-1, :]], dim=1)
            
            Bsz, T, D = yb.shape
            prev_flat = prev.reshape(Bsz * T, D)
            ctx_flat = cb.unsqueeze(1).expand(Bsz, T, cb.size(-1)).reshape(Bsz * T, cb.size(-1))
            target_delta = (yb - prev).reshape(Bsz * T, D)
            
            # Compute NLL
            log_prob = model.log_prob(target_delta, prev_flat, ctx_flat)
            loss = -log_prob.mean()
            
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                sched.step()
                ema.update(model)
            
            total += float(loss)
            n += 1
        
        return total / max(1, n)

    best = float("inf")
    best_epoch = 0
    bad = 0
    
    print("Starting ResidualGaussianBN training...")
    
    for ep in range(1, epochs + 1):
        tr = epoch_pass(train_loader, True)
        
        restore = ema.swap_into(model)
        va = epoch_pass(val_loader, False)
        restore()
        
        current_lr = opt.param_groups[0]["lr"]
        
        is_best = va + 1e-6 < best
        if is_best:
            best = va
            best_epoch = ep
        
        try:
            import wandb
            wandb.log({
                "epoch": ep,
                "train_loss": tr,
                "val_loss": va,
                "learning_rate": current_lr,
                "best_val_loss": best,
                "best_epoch": best_epoch,
            }, step=ep)
        except (ImportError, AttributeError):
            pass
        
        print(f"[BN] Epoch {ep:03d} | train {tr:.6f} | val {va:.6f} | best {best:.6f} @ ep{best_epoch}")
        
        if is_best:
            bad = 0
            ema.copy_to(model)
            ckpt = {
                "model_state": model.state_dict(),
                "model_cfg": {"state_dim": 7, "context_dim": 8, "hidden": hidden},
                "epoch": ep,
                "train_loss": tr,
                "val_loss": va,
                "best_val_loss": best,
            }
            if norm_stats is not None:
                ckpt["norm_stats"] = norm_stats
            torch.save(ckpt, ckpt_path)
            ema.restore(model)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}")
                break
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"\nLoaded best model from epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.6f}")
    
    return model


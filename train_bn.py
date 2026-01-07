#!/usr/bin/env python3
"""
Train ResidualGaussianBN (autoregressive) model on GPU cluster.

This script trains the ResidualGaussianBN model using autoregressive teacher forcing.
The model predicts a diagonal Gaussian over next-step residuals conditioned on the 
previous state and context. It can be run on GPU clusters using SLURM or similar 
job schedulers.

Usage:
    python train_bn.py [--epochs 60] [--batch-size 2048] [--lr 3e-4]
"""

import sys
import os
import random
import argparse
from pathlib import Path
import psutil
import gc
import traceback

# Force immediate output flushing for debugging
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def debug_log(msg, flush=True):
    """Debug logging with immediate flush"""
    print(f"[DEBUG] {msg}", flush=flush)
    sys.stdout.flush()
    sys.stderr.flush()

def debug_memory(step_name):
    """Debug memory with immediate flush"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024
        percent = process.memory_percent()
        if torch.cuda.is_available():
            torch_mem = torch.cuda.memory_allocated() / 1024 / 1024
            torch_max = torch.cuda.max_memory_allocated() / 1024 / 1024
            debug_log(f"{step_name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, "
                     f"PyTorch={torch_mem:.1f}MB (max={torch_max:.1f}MB), {percent:.1f}%")
        else:
            debug_log(f"{step_name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, {percent:.1f}%")
    except Exception as e:
        debug_log(f"{step_name}: Error getting memory: {e}")

debug_log("SCRIPT_START")
sys.stdout.flush()

import numpy as np
debug_log("IMPORTED_NUMPY")
sys.stdout.flush()
debug_memory("AFTER_NUMPY")

import torch
debug_log("IMPORTED_TORCH")
sys.stdout.flush()
debug_memory("AFTER_TORCH")

import wandb
debug_log("IMPORTED_WANDB")
sys.stdout.flush()
debug_memory("AFTER_WANDB")

# Use the debug_memory function defined above
log_memory = debug_memory

# Add project root to Python path
REPO_ROOT = Path(__file__).parent.parent
debug_log(f"REPO_ROOT: {REPO_ROOT}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
debug_log("ADDED_REPO_TO_PATH")
debug_memory("AFTER_PATH_SETUP")

# Quiet optional logs for cleaner output
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
if "TORCH_LOGS" in os.environ:
    os.environ.pop("TORCH_LOGS", None)
debug_log("SET_ENV_VARS")

# Import project utilities
try:
    debug_log("IMPORTING_UTILS_START")
    from utils import (
        WindowParams,
        SplitConfig,
        SamplingConfig,
        StatsConfig,
        TurnSampling,
        TrajectoryDataset,
        build_or_load_dataset,
        make_loader,
        load_and_engineer
    )
    debug_log("IMPORTED_UTILS_UTILS")
    debug_memory("AFTER_UTILS_UTILS_IMPORT")
except Exception as e:
    debug_log(f"ERROR_IMPORTING_UTILS: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    debug_log("IMPORTING_TRAINING_UTILS_START")
    from utils import train_bn
    debug_log("IMPORTED_TRAINING_UTILS")
    debug_memory("AFTER_TRAINING_UTILS_IMPORT")
except Exception as e:
    debug_log(f"ERROR_IMPORTING_TRAINING_UTILS: {e}")
    traceback.print_exc()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train ResidualGaussianBN (autoregressive) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-parquet",
        type=str,
        default="dataset_cache/traffic_enroute_filtered.parquet",
        help="Path to input parquet file (relative to script directory)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.5,
        help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="Learning rate warmup steps"
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9997,
        help="EMA decay rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=512,
        help="Hidden dimension for MLP"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="models/bn_improved.pt",
        help="Path to save checkpoint (relative to script directory)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detects if None)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VT2_BN_Autoregressive",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="run_residual_bn_full",
        help="W&B run name (auto-generated if None)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)"
    )
    
    args = parser.parse_args()
    
    # Start memory profiling
    log_memory("SCRIPT_START")
    
    # Set random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device configuration
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    log_memory("AFTER_DEVICE_SETUP")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Enable optimizations if using CUDA
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_parquet = script_dir.parent / args.input_parquet
    ckpt_path = script_dir.parent / args.ckpt_path
    
    # Ensure checkpoint directory exists
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration (MATCHING 51397c8e8791f8ca.key.json)
    wparams = WindowParams(
        input_len=60,
        output_horizon=60,
        output_stride=5,
        overlap=False
    )
    
    # Train/validation/test split configuration
    scfg = SplitConfig(
        train_frac=0.7,
        val_frac=0.15,
        split_seed=42
    )

    # Sampling configuration for trajectory selection
    samp = SamplingConfig(
        n_train=2_000_000,
        n_val=500_000,
        n_test=400_000,
        train_turn=TurnSampling(
            min_turn_frac=0.15, turn_thr=0.01, consec=3,
            consider_hist=True, consider_future=True
        ),
        val_turn=TurnSampling(
            min_turn_frac=0.15, turn_thr=0.01, consec=3,
            consider_hist=True, consider_future=True
        ),
        test_turn=TurnSampling(
            min_turn_frac=0.0, turn_thr=0.01, consec=3,
            consider_hist=True, consider_future=True
        ),
    )

    # Normalization statistics configuration
    stats_cfg = StatsConfig(
        stats_seed=1234,
        stats_sample_size=2_000_000
    )

    # Try to load dataset WITHOUT loading DataFrame first (if cache exists)
    try:
        debug_log(f"CHECKING_INPUT_FILE: {input_parquet}")
        if not input_parquet.exists():
            raise FileNotFoundError(f"Input parquet file not found: {input_parquet}")
        debug_log(f"INPUT_FILE_EXISTS: {input_parquet.stat().st_size / 1024 / 1024:.1f}MB")
        debug_memory("BEFORE_BUILD_DATASET")
        
        # Try to load from cache WITHOUT loading DataFrame
        debug_log("ATTEMPTING_CACHE_LOAD_WITHOUT_DF")
        try:
            (X_train, Y_train, C_train,
             X_val, Y_val, C_val,
             X_test, Y_test, C_test,
             norm_stats, meta_train, meta_val, meta_test,
             manifest, summary) = build_or_load_dataset(
                df=None,
                wparams=wparams,
                scfg=scfg,
                samp=samp,
                stats_cfg=stats_cfg,
                parquet_path=str(input_parquet),
            )
            debug_log("SUCCESSFULLY_LOADED_FROM_CACHE_WITHOUT_DF")
            debug_memory("AFTER_BUILD_DATASET")
            df = None
        except (ValueError, Exception) as e:
            debug_log(f"CACHE_CHECK_FAILED_NEED_DF: {e}")
            debug_memory("BEFORE_LOAD_AND_ENGINEER")
            
            debug_log("CALLING_LOAD_AND_ENGINEER")
            df = load_and_engineer(str(input_parquet))
            debug_log(f"LOADED_DATASET: {len(df)} trajectory segments")
            debug_log(f"DATAFRAME_MEMORY: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
            debug_memory("AFTER_LOAD_AND_ENGINEER")
            
            (X_train, Y_train, C_train,
             X_val, Y_val, C_val,
             X_test, Y_test, C_test,
             norm_stats, meta_train, meta_val, meta_test,
             manifest, summary) = build_or_load_dataset(
                df=df,
                wparams=wparams,
                scfg=scfg,
                samp=samp,
                stats_cfg=stats_cfg,
            )
            debug_log("BUILD_OR_LOAD_DATASET_COMPLETE")
            debug_memory("AFTER_BUILD_DATASET")
            
            del df
            gc.collect()
            debug_log("DF_DELETED")
            debug_memory("AFTER_DELETE_DF")
    except Exception as e:
        debug_log(f"ERROR_IN_DATA_LOADING: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if arrays are memory-mapped
    try:
        debug_log("CHECKING_ARRAY_MEMORY_MODES")
        def check_mmap(arr, name):
            try:
                if hasattr(arr, 'filename'):
                    debug_log(f"{name}: Memory-mapped (mmap) - file: {arr.filename}")
                else:
                    size_mb = arr.nbytes / 1024 / 1024
                    debug_log(f"{name}: In RAM - size: {size_mb:.1f}MB, shape: {arr.shape}")
            except Exception as e:
                debug_log(f"ERROR_CHECKING_{name}: {e}")
        
        check_mmap(X_train, "X_train")
        check_mmap(Y_train, "Y_train")
        check_mmap(C_train, "C_train")
        check_mmap(X_val, "X_val")
        check_mmap(Y_val, "Y_val")
        check_mmap(C_val, "C_val")
        
        train_ds = TrajectoryDataset(X_train, Y_train, C_train)
        val_ds = TrajectoryDataset(X_val, Y_val, C_val)
        test_ds = TrajectoryDataset(X_test, Y_test, C_test)
        
        debug_log(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    except Exception as e:
        debug_log(f"ERROR_IN_DATASET_CREATION: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize W&B
    wandb_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "warmup_steps": args.warmup_steps,
        "ema_decay": args.ema_decay,
        "patience": args.patience,
        "hidden": args.hidden,
        "seed": args.seed,
        "input_len": wparams.input_len,
        "output_horizon": wparams.output_horizon,
        "output_stride": wparams.output_stride,
        "train_samples": samp.n_train,
        "val_samples": samp.n_val,
    }
    
    wandb_kwargs = {
        "project": args.wandb_project,
        "config": wandb_config,
    }
    if args.wandb_name:
        wandb_kwargs["name"] = args.wandb_name
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    
    wandb.init(**wandb_kwargs)
    
    # Train the model
    print("\n" + "="*60)
    print("Starting ResidualGaussianBN (single Gaussian) training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dimension: {args.hidden}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  W&B Project: {args.wandb_project}")
    print("="*60 + "\n")
    
    model = train_bn(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        patience=args.patience,
        hidden=args.hidden,
        ckpt_path=str(ckpt_path),
        device=device,
        norm_stats=norm_stats,
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Best model saved to: {ckpt_path}")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    try:
        debug_log("ENTERING_MAIN")
        main()
        debug_log("MAIN_COMPLETE")
    except Exception as e:
        debug_log(f"FATAL_ERROR_IN_MAIN: {e}")
        traceback.print_exc()
        sys.exit(1)
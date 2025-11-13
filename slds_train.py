#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import json
import socket
import time
import signal
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

# --- Prefer local VT_2 sources to avoid version mismatches ---
LOCAL_LIBS = [
    '/home/fusg/VT_2/pybasicbayes-master',
    '/home/fusg/VT_2/pyhsmm-master',
    '/home/fusg/VT_2/pylds-master',
    '/home/fusg/VT_2/pyslds-master',
]
for p in reversed(LOCAL_LIBS):
    if p not in sys.path:
        sys.path.insert(0, p)
for m in ['pyslds', 'pybasicbayes', 'pyhsmm', 'pylds']:
    if m in sys.modules:
        del sys.modules[m]

import pyslds  # noqa: E402
import pyhsmm  # noqa: E402
from pyslds.models import DefaultSLDS  # noqa: E402

# --- Compatibility patches for pybasicbayes Regression under NumPy >= 2 ---
from pybasicbayes.distributions.regression import Regression as _Regression  # noqa: E402
from pybasicbayes.util.general import objarray as _objarray  # noqa: E402
from pybasicbayes.util.general import inv_psd as _inv_psd  # noqa: E402

def _reg_empty_statistics(self):
    D_in, D_out = self.D_in, self.D_out
    return np.array([
        np.zeros((D_out, D_out)),
        np.zeros((D_out, D_in)),
        np.zeros((D_in, D_in)),
        0,
    ], dtype=object)

def _reg_standard_to_natural(nu, S, M, K):
    Kinv = _inv_psd(K)
    A = S + M.dot(Kinv).dot(M.T)
    B = M.dot(Kinv)
    C = Kinv
    d = nu
    return _objarray([A, B, C, d])

def _reg_get_statistics(self, data):
    assert isinstance(data, (list, tuple, np.ndarray))
    if isinstance(data, list):
        return sum((self._get_statistics(d) for d in data), self._empty_statistics())
    elif isinstance(data, tuple):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)
        n = x.shape[0] if x.ndim == 2 else y.shape[0]
        y2 = y.T if y.ndim == 2 else y[:, None].T
        x2 = x.T if x.ndim == 2 else x[:, None].T
        yyT = y2 @ y2.T
        yxT = y2 @ x2.T
        xxT = x2 @ x2.T
        return np.array([yyT, yxT, xxT, n], dtype=object)
    else:
        data = data[~np.isnan(data).any(1)]
        y = data[:, :self.D_out].T
        x = data[:, self.D_out:].T
        n = x.shape[1]
        yyT = y @ y.T
        yxT = y @ x.T
        xxT = x @ x.T
        return np.array([yyT, yxT, xxT, n], dtype=object)

# Apply patches
_Regression._empty_statistics = _reg_empty_statistics
_Regression._standard_to_natural = staticmethod(_reg_standard_to_natural)
_Regression._get_statistics = _reg_get_statistics

# ===================== Checkpointing & Resume (4D-only) =====================
# Globals for signal-safe checkpointing
MODEL_REF = None
CURRENT_ITR = 0
_LAST_LOGLIKES = []
ARGS = None

def save_checkpoint(model, args, log_likelihoods, itr):
    """Save a best-effort checkpoint with model and minimal training state."""
    base = os.path.splitext(args.output)[0]
    ckpt_path = f"{base}.ckpt_itr{int(itr)}.pkl"
    ckpt = {
        'model': model,
        'training_info': {
            'n_iterations_done': int(itr),
            'log_likelihoods': list(log_likelihoods),
            'time': datetime.now().isoformat(),
        },
        'metadata': globals().get('metadata', None),
        'checkpoint': True,
    }
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt, f)
    # Write a tiny JSON sidecar for quick discovery
    sidecar = os.path.splitext(ckpt_path)[0] + '.json'
    with open(sidecar, 'w') as f:
        json.dump({'ckpt': ckpt_path, 'itr': int(itr), 'time': ckpt['training_info']['time']}, f)
    print(f"[checkpoint] saved: {ckpt_path}", flush=True)

def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _sig_handler(signum, frame):
    """On SIGTERM/SIGINT, dump a checkpoint and exit cleanly."""
    print(f"[signal] {signum} -> saving checkpoint (itr={CURRENT_ITR})", flush=True)
    try:
        if MODEL_REF is not None and ARGS is not None:
            save_checkpoint(MODEL_REF, ARGS, _LAST_LOGLIKES, CURRENT_ITR)
    finally:
        # Ensure process exits after best-effort save
        sys.exit(0)

# Register signal handlers early
signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)
# ===========================================================================

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        slds_data = pickle.load(f)
    observations = slds_data['observations']
    mode_sequences = slds_data.get('mode_sequences')
    metadata = slds_data['metadata']
    return observations, mode_sequences, metadata

def build_model(metadata):
    """Build a 4D-only SLDS model with identity emissions.

    Expects features: ['vx', 'vy', 'vertical_rate', 'altitude']
    """
    D_obs = int(metadata.get('D_obs', 4))
    if D_obs != 4:
        raise ValueError(f"This trainer is 4D-only. Prepared data reports D_obs={D_obs}. Regenerate data with 4D [vx, vy, vertical_rate, altitude].")
    D_latent = int(metadata.get('D_latent', D_obs))
    if D_latent != 4:
        raise ValueError(f"This trainer is 4D-only. Prepared data reports D_latent={D_latent}.")

    # Enforce feature names to expected 4D order
    feature_names = ['vx', 'vy', 'vertical_rate', 'altitude']
    K_modes = int(metadata.get('K_modes', max(1, int(metadata.get('N_typecode_clusters', 1)) * int(metadata.get('N_phases', 1)))))

    # Identity observation
    identity_C = np.eye(D_obs)

    # Initial dynamics and process noise
    As_init = [np.eye(D_latent) for _ in range(K_modes)]
    Q_init = [0.3 * np.eye(D_latent) for _ in range(K_modes)]

    model = DefaultSLDS(
        K=K_modes,
        D_obs=D_obs,
        D_latent=D_latent,
        As=As_init,
        sigma_statess=Q_init,
        Cs=identity_C,
        sigma_obss=0.05 * np.eye(D_obs),
        alpha=3.0,
    )

    # Priors: light damping towards identity
    M_prior = 0.95 * np.eye(D_latent)
    S0_prior = np.eye(D_latent)
    K_prior = 1.0 * np.eye(D_latent)
    nu0 = D_latent + 5

    for k in range(K_modes):
        dyn = model.dynamics_distns[k]
        dyn.nu_0 = nu0
        dyn.S_0 = S0_prior.copy()
        dyn.M = M_prior.copy()
        dyn.K = K_prior.copy()
        dyn.natural_hypparam = dyn._standard_to_natural(dyn.nu_0, dyn.S_0, dyn.M, dyn.K)

    return model, K_modes, feature_names

def check_and_optionally_stabilize(model, K_modes, clip=True, thresh=1.01, clip_to=0.99):
    unstable = []
    for k in range(K_modes):
        A_k = model.dynamics_distns[k].A
        ev = np.linalg.eigvals(A_k)
        mx = float(np.max(np.abs(ev)))
        if mx > thresh:
            unstable.append((k, mx))

    if not unstable:
        return {'unstable_count': 0, 'stabilized': 0}

    n_stabilized = 0
    if clip:
        for k, _ in unstable:
            A = model.dynamics_distns[k].A
            e, V = np.linalg.eig(A)
            e_clipped = np.where(np.abs(e) > clip_to, clip_to * e / np.abs(e), e)
            A_stable = V @ np.diag(e_clipped) @ np.linalg.inv(V)
            model.dynamics_distns[k].A = np.real(A_stable)
            n_stabilized += 1

    return {'unstable_count': len(unstable), 'stabilized': n_stabilized}

def main():
    parser = argparse.ArgumentParser(description='Train SLDS on prepared data (Slurm-friendly).')
    parser.add_argument('--data', default='/store/fusg/VT2/slds/slds_data_prepared.pkl', help='Path to prepared data pickle')
    parser.add_argument('--output', default='/store/fusg/VT2/slds/trained_slds_model.pkl', help='Output path for trained model pickle')
    parser.add_argument('--n-train', type=int, default=5000, help='Number of flights to use for training')
    parser.add_argument('--n-iter', type=int, default=200, help='Gibbs iterations')
    parser.add_argument('--log-every', type=int, default=10, help='Log-likelihood cadence')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save checkpoint every N iterations (0=use --log-every)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pkl to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-stabilize', action='store_true', help='Do not post-hoc stabilize A matrices')
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', default='VT2_SLDS', help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, help='W&B entity (team/org), e.g., fustaale-zhaw')
    parser.add_argument('--wandb-name', default=None, help='W&B run name (default auto)')
    parser.add_argument('--wandb-tags', default='', help='Comma-separated list of W&B tags')
    parser.add_argument('--wandb-mode', default=None, choices=[None, 'online', 'offline'], help='W&B mode override')

    args = parser.parse_args()
    global ARGS
    ARGS = args

    # Threading hints for BLAS
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    if cpus > 0:
        os.environ['OMP_NUM_THREADS'] = str(cpus)
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)

    np.random.seed(args.seed)

    # Optional Weights & Biases setup
    wb = None
    if args.wandb:
        try:
            import wandb  # noqa: WPS433
            if args.wandb_mode:
                os.environ['WANDB_MODE'] = args.wandb_mode
            run_name = args.wandb_name or f"SLDS_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tags = [t for t in (args.wandb_tags.split(',') if args.wandb_tags else []) if t]
            wb = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                tags=tags,
                config={
                    'script': 'slds_train.py',
                    'data': args.data,
                    'output': args.output,
                    'n_train': int(args.n_train),
                    'n_iter': int(args.n_iter),
                    'log_every': int(args.log_every),
                    'seed': int(args.seed),
                    'stabilize_posthoc': not args.no_stabilize,
                    'host': socket.gethostname(),
                },
                settings=wandb.Settings(start_method='thread')
            )
        except Exception as e:  # noqa: BLE001
            print('[wandb] init failed, proceeding without W&B:', repr(e))
            wb = None

    print('=== SLDS TRAIN START ===')
    print('host:', socket.gethostname())
    print('python:', sys.executable)
    print('start time:', datetime.now().isoformat())
    print('data:', args.data)
    print('output:', args.output)
    print('n_train:', args.n_train, 'n_iter:', args.n_iter)

    observations, mode_sequences, metadata = load_data(args.data)
    model, K_modes, feature_names = build_model(metadata)

    # Optional resume from checkpoint (replaces model and continues from saved iteration)
    start_itr = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        print(f"[resume] loading checkpoint: {args.resume}")
        ck = load_checkpoint(args.resume)
        model = ck.get('model', model)
        ti = ck.get('training_info', {})
        start_itr = int(ti.get('n_iterations_done', 0))
        saved_lls = list(ti.get('log_likelihoods', []))
        _LAST_LOGLIKES.extend(saved_lls)
        # Prefer metadata from checkpoint if present
        if 'metadata' in ck and ck['metadata'] is not None:
            globals()['metadata'] = ck['metadata']
        print(f"[resume] restored; starting at iter {start_itr}")

    if wb is not None:
        try:
            wb.config.update({
                'K_modes': K_modes,
                'D_obs': 4,
                'D_latent': 4,
                'feature_names': feature_names,
                'prior_damping_rho': 0.95,
                'sigma_obss_diag': 0.05,
            }, allow_val_change=True)
        except Exception:
            pass

    # Add data
    n_train = min(args.n_train, len(observations))
    print(f'Adding {n_train} flights...')
    for i, y in enumerate(tqdm(observations[:n_train], desc='add_data', miniters=100)):
        # Enforce 4D observations
        if y.shape[1] != 4:
            raise ValueError(f"Observed sequence at index {i} has D={y.shape[1]}, expected 4. Regenerate prepared data.")
        z = None
        if mode_sequences is not None and i < len(mode_sequences):
            z = mode_sequences[i]
        if z is not None and len(z) == len(y):
            model.add_data(y, stateseq=np.asarray(z, dtype=np.int32), initialize_from_prior=False)
        else:
            model.add_data(y)

    # Train
    N_ITER = int(args.n_iter)
    LOG_EVERY = int(args.log_every)
    CKPT_EVERY = int(args.checkpoint_every) if int(args.checkpoint_every) > 0 else LOG_EVERY
    log_likelihoods = []
    t0 = time.time()
    print(f'Gibbs: {N_ITER} iterations (log every {LOG_EVERY}), checkpoint every {CKPT_EVERY}, start at {start_itr}')

    # Expose model for signal handler
    global MODEL_REF
    MODEL_REF = model

    for itr in tqdm(range(start_itr, N_ITER), desc='gibbs'):
        global CURRENT_ITR
        CURRENT_ITR = int(itr)
        model.resample_model()
        if itr % LOG_EVERY == 0 or itr == N_ITER - 1:
            ll = float(model.log_likelihood())
            log_likelihoods.append(ll)
            _LAST_LOGLIKES.append(ll)
            print(f'iter {itr:4d} | LL={ll:.2f}')
            if wb is not None:
                try:
                    wb.log({'iter': int(itr), 'log_likelihood': ll})
                except Exception:
                    pass
        # Periodic checkpointing
        if CKPT_EVERY and (itr % CKPT_EVERY == 0 or itr == N_ITER - 1):
            try:
                save_checkpoint(model, args, log_likelihoods, itr)
            except Exception as e:
                print('[checkpoint] periodic save failed:', repr(e))
    t1 = time.time()
    print(f'Training wall time: {(t1 - t0)/60:.2f} min')
    if wb is not None:
        try:
            wb.log({'wall_time_min': (t1 - t0)/60.0})
        except Exception:
            pass

    # Stability check and optional clipping
    stats = check_and_optionally_stabilize(model, K_modes, clip=(not args.no_stabilize))
    if stats['unstable_count'] > 0:
        print(f"Unstable modes detected: {stats['unstable_count']} | stabilized: {stats['stabilized']}")
    else:
        print('All modes stable.')
    if wb is not None:
        try:
            wb.log({'unstable_detected': stats['unstable_count'], 'stabilized': stats['stabilized']})
        except Exception:
            pass

    # Package and save
    trained_model = {
        'model': model,
        'training_info': {
            'n_iterations': N_ITER,
            'n_flights_trained': n_train,
            'log_likelihoods': log_likelihoods,
            'final_ll': log_likelihoods[-1] if log_likelihoods else None,
            'feature_names': feature_names,
        },
        'metadata': metadata,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(trained_model, f)
    print('Saved model to:', args.output)

    # Also write a small JSON summary next to it
    summary_path = os.path.splitext(args.output)[0] + '_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'host': socket.gethostname(),
            'python': sys.executable,
            'start_time': datetime.now().isoformat(),
            'data': args.data,
            'output': args.output,
            'n_train': n_train,
            'n_iter': N_ITER,
            'log_every': LOG_EVERY,
            'unstable_detected': stats['unstable_count'],
            'stabilized': stats['stabilized'],
            'final_ll': log_likelihoods[-1] if log_likelihoods else None,
        }, f, indent=2)
    print('Summary written to:', summary_path)

    print('=== SLDS TRAIN DONE ===')
    # Log artifacts to W&B at the end
    if wb is not None:
        try:
            import wandb  # noqa: WPS433
            art = wandb.Artifact('slds_model', type='model')
            art.add_file(args.output)
            if os.path.exists(summary_path):
                art.add_file(summary_path)
            wb.log_artifact(art)
            wb.finish()
        except Exception as e:  # noqa: BLE001
            print('[wandb] artifact logging failed:', repr(e))

if __name__ == '__main__':
    main()

"""
obs_compat.py - keep cars trained before an observation change working.

Observation versions (per stacked frame):
    v1   N_RAYS x 4 ray features + FRAME_EXTRA      (closeness, static, vehicle, pedestrian)
    v2   N_RAYS x 5 ray features + FRAME_EXTRA      (+ the hit vehicle's turn signal)

Two ways to use an older (v1) model with the current (v2) environment:
    * downgrade_index(): pick the v1 values out of a v2 observation, so the old
      model sees exactly what it was trained on (watching, testing);
    * upgrade_model():   widen the model's first layers with ZERO weights for
      the new inputs, so it drives exactly as before but can now LEARN to use
      them (continue training, --init-from the previous level's car).
"""
from __future__ import annotations

import numpy as np

from .env import FRAME_DIM, FRAME_DIM_V1, FRAME_EXTRA
from .sensors import N_RAYS, RAY_FEATURES


def obs_version(dim, frame_stack=4):
    if dim == FRAME_DIM * frame_stack:
        return 2
    if dim == FRAME_DIM_V1 * frame_stack:
        return 1
    raise ValueError(f"unknown observation size {dim} (expected {FRAME_DIM * frame_stack} "
                     f"or the older {FRAME_DIM_V1 * frame_stack})")


def downgrade_index(frame_stack=4):
    """Indices into a v2 observation that give the v1 observation."""
    per = []
    for i in range(N_RAYS):
        per += [i * RAY_FEATURES + f for f in range(4)]
    per += [N_RAYS * RAY_FEATURES + e for e in range(FRAME_EXTRA)]
    per = np.array(per)
    return np.concatenate([k * FRAME_DIM + per for k in range(frame_stack)])


def upgrade_columns(frame_stack=4):
    """For each v2 input: the v1 input it comes from, or -1 if it is new."""
    src = np.full(FRAME_DIM * frame_stack, -1, dtype=np.int64)
    src[downgrade_index(frame_stack)] = np.arange(FRAME_DIM_V1 * frame_stack)
    return src


def upgrade_obs_rms(rms, frame_stack=4):
    """Widen VecNormalize running stats: new inputs start at mean 0, var 1."""
    src = upgrade_columns(frame_stack)
    mean = np.zeros(len(src))
    var = np.ones(len(src))
    has = src >= 0
    mean[has] = rms.mean[src[has]]
    var[has] = rms.var[src[has]]
    rms.mean, rms.var = mean, var
    return rms


def upgrade_model(old, venv, frame_stack=4):
    """A new PPO model on `venv` (v2 observations) that behaves exactly like the
    v1 model `old`: first-layer weights copied, zero weights for the new inputs."""
    import torch
    from stable_baselines3 import PPO

    src = torch.as_tensor(upgrade_columns(frame_stack))
    has = src >= 0
    new = PPO("MlpPolicy", venv, device="cpu", seed=0,
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
              learning_rate=old.learning_rate, target_kl=old.target_kl, gamma=old.gamma,
              gae_lambda=old.gae_lambda, clip_range=old.clip_range, ent_coef=old.ent_coef,
              vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
              policy_kwargs=old.policy_kwargs, verbose=0)
    sd_old = old.policy.state_dict()
    sd_new = new.policy.state_dict()
    in_old = FRAME_DIM_V1 * frame_stack
    for k, w in sd_old.items():
        if w.ndim == 2 and w.shape[1] == in_old and sd_new[k].shape[1] == len(src):
            wn = torch.zeros_like(sd_new[k])
            wn[:, has] = w[:, src[has]]
            sd_new[k] = wn
        else:
            sd_new[k] = w
    new.policy.load_state_dict(sd_new)
    new.num_timesteps = old.num_timesteps
    return new

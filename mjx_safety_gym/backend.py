"""Which MJX implementation builds models and data: MuJoCo Warp or MJX-JAX.

TRAINING DEFAULTS TO 'warp' SINCE 2026-10-04 (train_ppo --impl; see
notes/warp_migration.md): 5.3x training throughput on the laptop RTX 4050 and
3.1-3.3x on a 3090, with the constraint solver run to convergence, and the same
physics as MJX-JAX to ~1e-7. Everything else -- main.py, replay, render and
eval scripts -- defaults to 'jax' here: it runs on CPU (Warp needs a CUDA GPU)
and single-env viewing does not need Warp's throughput. MJX_IMPL overrides the
default for scripts without a flag.

Warp needs its buffers sized up front, which MJX-JAX does not:
  naconmax  contacts for ALL worlds combined (one shared buffer, not per env)
  njmax     constraint rows per world
`configure()` sizes them from the number of worlds the caller will batch;
measured peak use on the ant (1024 worlds, random actions, full solver):
1,310 contacts in total and 20 rows per world, against 64 x worlds and 256.

Every env builds its MJX model and data through `put_model` / `make_data`
here, so the backend is chosen in ONE place and must be chosen BEFORE any env
is constructed (the model is put on the device in the env's __init__).
"""
from __future__ import annotations

import os

from mujoco import mjx

IMPLS = ("warp", "jax")

_config = {
    # Library default jax (runs anywhere); the trainer configures warp.
    "impl": os.environ.get("MJX_IMPL") or "jax",
    "nworld": 1,
    "contacts_per_world": 64,
    "njmax": 256,
}


def configure(
    impl: str | None = None,
    nworld: int | None = None,
    contacts_per_world: int | None = None,
    njmax: int | None = None,
) -> str:
    """Set the backend and Warp's buffer sizes; None keeps the current value.

    Returns the active implementation. Call before constructing any env.
    """
    if impl is not None:
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
        _config["impl"] = impl
    for key, value in (("nworld", nworld), ("contacts_per_world", contacts_per_world),
                       ("njmax", njmax)):
        if value is not None:
            if int(value) < 1:
                raise ValueError(f"{key} must be >= 1, got {value}")
            _config[key] = int(value)
    return active()


def active() -> str:
    """The implementation in use."""
    return _config["impl"]


def put_model(m):
    return mjx.put_model(m, impl=active())


def make_data(model, mj_model):
    """`mj_model` (the host MjModel) is required by Warp: its make_data only
    accepts a mujoco.MjModel and uses it for sizes alone, so any compiled
    variant of the same structure works (per-env bodies included)."""
    if active() == "jax":
        return mjx.make_data(model, impl="jax")
    return mjx.make_data(
        mj_model,
        impl="warp",
        naconmax=_config["contacts_per_world"] * _config["nworld"],
        njmax=_config["njmax"],
    )

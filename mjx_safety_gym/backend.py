"""warp-probe: choose the MJX implementation from the environment.

MJX_IMPL unset or 'jax' = MJX-JAX (every run so far). 'warp' = MuJoCo Warp
through MJX. Warp needs contact/constraint buffer sizes up front:
MJX_NACONMAX = contacts for ALL worlds combined (default 64 x MJX_NWORLD),
MJX_NJMAX = constraint rows per world (default 256).
"""
import os

from mujoco import mjx

IMPL = os.environ.get("MJX_IMPL") or None


def put_model(m):
    return mjx.put_model(m, impl=IMPL)


def make_data(model, mj_model=None):
    """`mj_model` (host MjModel) is required for warp: its make_data only
    takes a mujoco.MjModel, and only uses it for sizes."""
    if IMPL != "warp":
        return mjx.make_data(model, impl=IMPL)
    nworld = int(os.environ.get("MJX_NWORLD", "1024"))
    return mjx.make_data(
        mj_model,
        impl=IMPL,
        naconmax=int(os.environ.get("MJX_NACONMAX", 64 * nworld)),
        njmax=int(os.environ.get("MJX_NJMAX", 256)),
    )

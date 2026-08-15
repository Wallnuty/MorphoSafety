"""Shared fixtures.

TESTS RUN ON THE CPU BACKEND BY DEFAULT, deliberately. This project's own
stability rules (see the plan file's "Session stability" section) allow at most
one GPU/JAX process at a time, because two concurrent CUDA contexts in a 7.6
GiB WSL guest is the documented cause of a full VSCode/WSL hang. Tests that
grabbed the GPU could not be run while a training job was in flight, which is
exactly when you most want to run them. On CPU the whole suite is compile-bound
(~20 s for the ant's step, then ~10 ms/step), which is fine for a handful of
steps per test.

Override with `JAX_PLATFORMS=cuda pytest ...` if you specifically want to test
GPU behaviour -- but note that GPU rollouts here are NOT bitwise reproducible
across runs (contact chaos amplifies float32 association differences), so any
test asserting exact equality should stay on CPU.
"""

from __future__ import annotations

import os
import warnings

# Must be set before jax is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest  # noqa: E402

# mujoco-mjx 3.3.2 casts float64 max to float32 in collision_convex._box_box
# (`jp.where(jp.isinf(dist), jp.finfo(float).max, dist)`), which warns on every
# single reset. Upstream, harmless, and loud enough to bury real warnings.
warnings.filterwarnings(
    "ignore", message="overflow encountered in cast", category=RuntimeWarning
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds an ant env and compiles physics (tens of seconds)"
    )


@pytest.fixture(scope="session")
def run_forward_point():
    """RunForward with the point robot -- cheapest env that exercises the task.

    Session-scoped because construction plus the first jitted step dominates
    runtime; the tests themselves only take a few steps each.
    """
    from mjx_safety_gym.envs.run_forward import RunForward

    return RunForward(robot="point")


@pytest.fixture(scope="session")
def go_to_goal_point():
    from mjx_safety_gym.envs.go_to_goal import GoToGoal

    return GoToGoal(robot="point")


@pytest.fixture(scope="session")
def minefield_point():
    """Minefield with the point robot -- RunForward's arena minus the vases."""
    from mjx_safety_gym.envs.minefield import Minefield

    return Minefield(robot="point")


@pytest.fixture(scope="session")
def run_forward_ant():
    """ant_gym on RunForward -- the configuration actually being trained."""
    from mjx_safety_gym.envs.run_forward import RunForward

    return RunForward(robot="ant_gym", healthy_reward=0.002, terminate_on_flip=True)

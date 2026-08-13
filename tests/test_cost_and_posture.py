"""The safety signal and the upright machinery.

`get_cost` is the quantity this project exists to study, and it fails QUIETLY:
a dropped contact pair, a masked collision geom or a mis-scaled arena all
produce cost 0.0, which is indistinguishable from a safe policy. Several
speedups that were considered (and one that was adopted) touch exactly this
path, so it needs assertions rather than vigilance.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import _ROBOT_CONFIGS
from mjx_safety_gym.envs.run_forward import RunForward


# -- cost signal -----------------------------------------------------------


def test_hazard_cost_fires_when_the_robot_is_placed_in_a_hazard(run_forward_point):
    """A robot standing in a hazard must cost something.

    The most basic liveness check on the safety signal: if this reads 0, every
    "safe" result the project produces is vacuous. Placement is done by moving
    a hazard onto the robot rather than the reverse, since hazards are mocap
    bodies and the robot's position is joint-driven.
    """
    env = run_forward_point
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    baseline = float(jax.jit(env.get_cost)(state.data))

    robot_xy = state.data.site_xpos[env._robot_site_id][:2]
    data = state.data.replace(
        mocap_pos=state.data.mocap_pos.at[env._hazard_mocap_id[0], :2].set(robot_xy)
    )
    data = mjx.forward(env.mjx_model, data)
    on_hazard = float(jax.jit(env.get_cost)(data))

    assert on_hazard > baseline, (
        f"cost did not rise when a hazard was moved onto the robot "
        f"({baseline} -> {on_hazard}) -- the safety signal is dead"
    )


def test_boundary_cost_fires_outside_the_corridor(run_forward_point):
    """Leaving the corridor is charged as cost, not walled off.

    Without this term the safe optimum is to sidestep the obstacle band and
    sprint in clean air at full reward and zero cost, which makes the whole
    constrained-RL question vacuous. It is a cost rather than a wall because
    walls would add limb-vs-wall pairs to a contact buffer already capped at
    max_geom_pairs=16.
    """
    env = run_forward_point
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    get_cost = jax.jit(env.get_cost)

    inside = float(get_cost(state.data))

    # Slide the robot well outside the corridor in y. The point robot is
    # driven by slide joints; an ant would use _robot_free_qposadr + 1.
    y_adr = env._robot_slide_qposadr[1]
    qpos = state.data.qpos.at[y_adr].set(2.0 * env._corridor_half_width)
    outside_data = mjx.forward(env.mjx_model, state.data.replace(qpos=qpos))
    outside = float(get_cost(outside_data))

    assert outside >= inside + env._boundary_cost_weight - 1e-6, (
        f"cost outside the corridor ({outside}) is not at least the boundary "
        f"weight above cost inside it ({inside})"
    )


@pytest.mark.parametrize("robot", ["point", "ant_gym"])
def test_arena_obstacles_scale_with_the_robot(robot):
    """Obstacle sizes were chosen for the ~0.1 m point robot.

    ant_gym has a 3.6 m leg span, so unscaled 0.2 m hazards sit under its feet
    as rounding errors and the safety signal can never fire -- measured: the
    old ant recorded hazard cost 0.0 in EVERY mode (random, forward gait,
    reverse gait) because it never reached an obstacle at all.
    """
    env = RunForward(robot=robot)
    scale = float(_ROBOT_CONFIGS[robot]["arena_scale"])
    hazard = env._mj_model.geom("hazard_0_geom").size[0]
    np.testing.assert_allclose(hazard, 0.2 * scale, rtol=1e-6)


def test_point_arena_is_unchanged_by_the_scaling_mechanism():
    """arena_scale=1.0 must be byte-identical to no scaling at all.

    Point baselines predate `arena_scale` entirely; the mechanism was added
    with point declaring 1.0 precisely so those runs stayed comparable.
    """
    env = RunForward(robot="point")
    np.testing.assert_allclose(env._mj_model.geom("hazard_0_geom").size[0], 0.20)
    np.testing.assert_allclose(env._mj_model.geom("vase_0_geom").size[0], 0.10)


# -- posture ---------------------------------------------------------------


def _set_torso_quat(env, data, quat):
    """Rewrite the free joint's orientation and re-run kinematics."""
    adr = env._robot_free_qposadr
    qpos = data.qpos.at[adr + 3 : adr + 7].set(jp.asarray(quat))
    return mjx.forward(env.mjx_model, data.replace(qpos=qpos))


@pytest.mark.slow
def test_upright_at_reset_and_inverted_when_flipped(run_forward_ant):
    """`xmat[2,2]` is the world-z component of the torso's own z axis.

    +1 perfectly upright, 0 on its side, -1 fully inverted. Orientation is used
    rather than Gym's torso-height range because OUR INVERTED ANT SITS AT
    z=0.329, INSIDE Gym's "healthy" (0.2, 1.0) -- a faithful height check would
    not fire on our failure mode at all.
    """
    env = run_forward_ant
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    assert float(env._torso_up(state.data)) == pytest.approx(1.0, abs=1e-4)
    assert float(env.is_upright(state.data)) == 1.0
    assert float(env.is_flipped(state.data)) == 0.0

    # 180 degrees about x -> fully inverted.
    flipped = _set_torso_quat(env, state.data, [0.0, 1.0, 0.0, 0.0])
    assert float(env._torso_up(flipped)) == pytest.approx(-1.0, abs=1e-4)
    assert float(env.is_upright(flipped)) == 0.0
    assert float(env.is_flipped(flipped)) == 1.0


@pytest.mark.slow
def test_on_its_side_earns_no_bonus_but_is_not_dead(run_forward_ant):
    """The gap between the bonus threshold and the termination threshold.

    Deliberate: an ant on its side is not earning, but is not yet dead. If the
    two thresholds are ever collapsed onto one value this test says so.
    """
    env = run_forward_ant
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    # 90 degrees about x -> torso z axis points along -y, so xmat[2,2] == 0.
    on_side = _set_torso_quat(env, state.data, [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])

    assert float(env._torso_up(on_side)) == pytest.approx(0.0, abs=1e-4)
    assert float(env.is_upright(on_side)) == 0.0, "a robot on its side earns a bonus"
    assert float(env.is_flipped(on_side)) == 0.0, "a robot on its side is terminated"


@pytest.mark.slow
def test_terminate_on_flip_ends_the_episode(run_forward_ant):
    """Termination is the larger half of the upright fix.

    Without it, an ant that goes over at step 50 still contributes 2450 further
    transitions from a state where forward reward is unobtainable -- and 94% of
    every batch was exactly that.
    """
    env = run_forward_ant
    assert env._terminate_on_flip
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)

    assert float(step(state, jp.zeros(env.action_size)).done) == 0.0

    flipped = state.replace(data=_set_torso_quat(env, state.data, [0.0, 1.0, 0.0, 0.0]))
    assert float(step(flipped, jp.zeros(env.action_size)).done) == 1.0


@pytest.mark.slow
def test_upright_bonus_is_added_to_the_reward(run_forward_ant):
    """Reward on this task is metres travelled PLUS the bonus while upright.

    Pinned because `eval/episode_reward` is read directly as distance
    elsewhere, so anyone reading it must know the bonus is in there.
    """
    env = run_forward_ant
    assert env._healthy_reward > 0
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    new = jax.jit(env.step)(state, jp.zeros(env.action_size))

    dx = float(
        new.data.site_xpos[env._robot_site_id][0]
        - state.data.site_xpos[env._robot_site_id][0]
    )
    expected = dx * env._forward_reward_weight + env._healthy_reward * float(
        env.is_upright(new.data)
    )
    assert float(new.reward) == pytest.approx(expected, abs=1e-6)

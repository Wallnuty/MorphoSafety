"""Minefield: the fast variant of RunForward, with no dynamic obstacles.

Two things have to hold for this task to be worth having, and each fails
silently in its own way:

  * The SAVING must be real. Minefield exists only because vases carry a free
    joint each and hazards carry none. If a vase ever crept back into the
    arena the task would still train, still report sensible numbers, and just
    be slow -- so the DOF counts are asserted, not the wall clock.
  * The COST SIGNAL must survive. Dropping vases removes the contact-based
    half of get_cost. If the distance-based half broke too, cost would read
    0.0 everywhere, which is indistinguishable from a perfectly safe policy.

Measured on the laptop GPU at the training default of 256 envs, through the
real wrapper stack: run 870 env-steps/s, minefield 2726 -- 3.13x. The gap
widens with batch size because run saturates the GPU (781 -> 875 from 128 to
256 envs) while minefield keeps scaling (1240 -> 2726).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import pytest
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import GoToGoal
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.envs.run_forward import RunForward


# -- structure: where the speedup actually comes from ----------------------


def test_minefield_has_no_dynamic_obstacles(minefield_point, run_forward_point):
    """No vase bodies, and nothing left for the contact-based cost to hit.

    The DOF counts are the real assertion. A vase is a free body (7 qpos,
    6 qvel each), so re-adding one would be invisible in behaviour and
    expensive in throughput -- exactly the kind of regression that gets
    noticed months later as "training got slow".
    """
    mine, run = minefield_point, run_forward_point

    vase_bodies = [
        mine.mj_model.body(i).name
        for i in range(mine.mj_model.nbody)
        if mine.mj_model.body(i).name.startswith("vase_")
    ]
    assert vase_bodies == [], f"Minefield grew vases back: {vase_bodies}"
    assert mine.spec["vases"].num_objects == 0
    assert mine._collision_obstacle_geoms_ids == [], (
        "collidable obstacle geoms remain, so get_cost still pays for the "
        "contact path it was supposed to shed"
    )

    assert mine.mj_model.nq < run.mj_model.nq
    assert mine.mj_model.nv < run.mj_model.nv
    # 10 vases x (7 qpos, 6 qvel) is the whole difference; anything else means
    # a body appeared or vanished somewhere unexpected.
    assert run.mj_model.nq - mine.mj_model.nq == 70
    assert run.mj_model.nv - mine.mj_model.nv == 60


def test_minefield_keeps_the_obstacle_count_of_run(minefield_point, run_forward_point):
    """Cheaper, not emptier.

    20 hazards replaces 10 hazards + 10 vases. If the default silently dropped
    to 10 the corridor would be half as cluttered and every cost number would
    quietly become incomparable with the run task's.
    """
    mine, run = minefield_point, run_forward_point
    run_obstacles = run.spec["hazards"].num_objects + run.spec["vases"].num_objects
    mine_obstacles = mine.spec["hazards"].num_objects + mine.spec["vases"].num_objects
    assert mine.spec["hazards"].num_objects == Minefield.DEFAULT_NUM_HAZARDS == 20
    assert mine_obstacles == run_obstacles == 20


def test_minefield_rejects_a_hazard_free_corridor():
    """0 hazards + 0 vases is a corridor with no obstacle cost at all.

    _post_init also reads the cost threshold off `hazard_0_geom`, so this
    would fail later and less clearly. Better to refuse it.
    """
    with pytest.raises(ValueError, match="at least one hazard"):
        Minefield(robot="point", num_hazards=0)


def test_obstacle_counts_are_parameters_not_literals():
    """The knob Minefield is built on must actually reach the arena."""
    env = Minefield(robot="point", num_hazards=5)
    assert env.spec["hazards"].num_objects == 5
    haz_bodies = sum(
        1
        for i in range(env.mj_model.nbody)
        if env.mj_model.body(i).name.startswith("hazard_")
    )
    assert haz_bodies == 5


# -- the invariant that makes checkpoints portable -------------------------


def test_all_three_tasks_share_an_observation_width(
    minefield_point, run_forward_point, go_to_goal_point
):
    """Lidar is binned by RING, never by object count.

    This is what lets a policy trained on minefield warm-start run without
    network surgery -- the whole point of iterating on the fast task. It would
    break the moment anything task-specific were appended to the observation.
    """
    widths = {
        "minefield": minefield_point.observation_size,
        "run": run_forward_point.observation_size,
        "goal": go_to_goal_point.observation_size,
    }
    assert len(set(widths.values())) == 1, widths

    state = jax.jit(minefield_point.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape == (minefield_point.observation_size,)


def test_changing_the_hazard_count_does_not_change_the_observation(minefield_point):
    """Obstacle count must not leak into obs width, or --num_hazards would
    silently invalidate every existing checkpoint."""
    assert Minefield(robot="point", num_hazards=5).observation_size == (
        minefield_point.observation_size
    )


# -- cost: the half of the signal that has to survive ----------------------


def test_hazard_cost_still_fires_without_vases(minefield_point):
    """Dropping vases removes the CONTACT half of get_cost.

    The distance half is independent -- hazards are charged by a swept-sphere
    test against the robot's limbs, never by contacts -- but "independent" is
    an assumption about code that was just edited, so it gets asserted. A dead
    signal here reads as 0.0 cost, i.e. as a perfectly safe policy.
    """
    env = minefield_point
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    baseline = float(jax.jit(env.get_cost)(state.data))

    robot_xy = state.data.site_xpos[env._robot_site_id][:2]
    data = state.data.replace(
        mocap_pos=state.data.mocap_pos.at[env._hazard_mocap_id[0], :2].set(robot_xy)
    )
    data = mjx.forward(env.mjx_model, data)
    on_hazard = float(jax.jit(env.get_cost)(data))

    assert on_hazard > baseline, (
        f"hazard cost is dead with vases removed ({baseline} -> {on_hazard})"
    )


def test_minefield_inherits_the_run_reward(minefield_point):
    """Episode return must still equal net +x displacement, exactly.

    Minefield changes the arena only. If it ever changed the reward, the
    telescoping property that makes this family of tasks learnable -- and
    makes return directly readable as metres -- would go with it.
    """
    env = minefield_point
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    site = env._robot_site_id
    x0 = float(state.data.site_xpos[site][0])
    rng = jax.random.PRNGKey(1)
    total = 0.0
    for _ in range(30):
        rng, k = jax.random.split(rng)
        action = jax.random.uniform(k, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
        total += float(state.reward)
    x1 = float(state.data.site_xpos[site][0])

    assert total == pytest.approx(x1 - x0, abs=1e-5), (
        "episode return is no longer net +x displacement"
    )
    # Guard against a vacuous pass: if the robot never moved, the assertion
    # above holds trivially at 0 == 0.
    assert abs(x1 - x0) > 1e-6, "robot did not move; the check proves nothing"


# -- regression guard on the tasks that already worked ---------------------


def test_run_and_goal_still_have_ten_of_each(run_forward_point, go_to_goal_point):
    """num_hazards/num_vases became parameters. Their defaults must be the
    literals they replaced, or every pre-existing result silently changes
    arena."""
    for env in (run_forward_point, go_to_goal_point):
        assert env.spec["hazards"].num_objects == 10
        assert env.spec["vases"].num_objects == 10
        assert len(env._collision_obstacle_geoms_ids) == 10

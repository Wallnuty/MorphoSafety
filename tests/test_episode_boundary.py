"""The episode-boundary reward bug, in both environments.

THE BUG THIS FILE EXISTS FOR. `BraxAutoResetWrapper` (mujoco_playground)
restores `data` and `obs` from the reset state when an episode ends, and leaves
every OTHER `info` key untouched. So any env that carries a position-like
scalar in `state.info` and differences against it measures the first step of
each new episode against the PREVIOUS episode's final value.

Measured before the fix: RunForward produced -0.02472 where the correct value
was +0.0007 -- wrong by 25x and of the opposite sign -- and at a real episode
length the spurious reward is ~1 m against a typical per-step ~0.001 m, i.e. a
1000x outlier once per episode, landing exactly where PPO's advantage
normalisation lets one outlier distort a whole batch. GoToGoal had the
identical bug via `last_goal_dist` for every run this project had ever done,
including the point baselines.

It produced no error, no NaN, and no crash. It was found by hand. Hence tests.

THE INVARIANT, and why it is stated this way: `state.data` is the one thing the
wrapper DOES restore, so a reward derived from it is correct across boundaries
by construction. Therefore for the step immediately following a `done`, reward
must equal the displacement between `prev_state.data` and `new_state.data` --
the same relation that holds mid-episode. A carried scalar breaks exactly this
step and nothing else.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from mjx_safety_gym.algorithms.train_ppo import wrap_for_brax_training

# Short enough that boundaries happen within a few steps, and action_repeat=1
# so one wrapper step is one physics decision and rewards are not summed.
EPISODE_LENGTH = 4
ACTION_REPEAT = 1


def _wrapped(env):
    return wrap_for_brax_training(
        env, episode_length=EPISODE_LENGTH, action_repeat=ACTION_REPEAT
    )


def _rollout(env, n_steps, seed=0):
    """Step the wrapped env, recording (prev_state, new_state) pairs.

    ACTIONS MUST ACTUALLY MOVE THE ROBOT. With zero actions the point never
    changes x, so a stale carried `last_x` coincidentally equals the next
    episode's reset x and the boundary bug is invisible -- an earlier draft of
    this test passed against a deliberately reintroduced bug for exactly that
    reason. `test_the_boundary_check_is_not_vacuous` pins the requirement.
    """
    # No jax.vmap here: wrap_for_brax_training already includes VmapWrapper,
    # so `wrapped.reset` takes a BATCH of keys and batching again gives each
    # lane a scalar key ("expected key_data.ndim >= 1").
    wrapped = _wrapped(env)
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)

    state = reset(jax.random.split(jax.random.PRNGKey(seed), 1))
    key = jax.random.PRNGKey(seed + 1)
    pairs = []
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(
            sub, (1, env.action_size), minval=-1.0, maxval=1.0
        )
        new_state = step(state, action)
        pairs.append((state, new_state))
        state = new_state
    return pairs


def test_run_forward_reward_is_correct_across_the_boundary(run_forward_point):
    """reward == x(new.data) - x(prev.data), at every step INCLUDING post-reset.

    Runs long enough to cross several episode boundaries. If a carried `last_x`
    is ever reintroduced, the step after each `done` is the one that breaks.
    """
    env = run_forward_point
    site = env._robot_site_id
    pairs = _rollout(env, n_steps=3 * EPISODE_LENGTH + 2)

    boundaries_seen = 0
    for i, (prev, new) in enumerate(pairs):
        x_prev = float(prev.data.site_xpos[0, site, 0])
        x_new = float(new.data.site_xpos[0, site, 0])
        reward = float(new.reward[0])
        just_reset = bool(prev.done[0] > 0)

        if bool(new.done[0] > 0):
            # On the done step itself the wrapper has already swapped `data`
            # for the reset state's, so new.data no longer describes the state
            # the reward was computed from. Nothing to assert here.
            continue

        expected = (x_new - x_prev) * env._forward_reward_weight
        if env._healthy_reward:
            expected += env._healthy_reward * float(env.is_upright(
                jax.tree.map(lambda a: a[0], new.data)
            ))
        assert reward == pytest.approx(expected, abs=1e-6), (
            f"step {i}: reward {reward} != displacement {expected}"
            + (" -- THIS IS THE STEP AFTER AN EPISODE BOUNDARY, the exact "
               "failure mode of a carried last_x in state.info"
               if just_reset else "")
        )
        boundaries_seen += just_reset

    assert boundaries_seen >= 2, (
        f"only {boundaries_seen} post-boundary steps checked -- the test is not "
        "actually exercising the bug it exists for"
    )


def test_the_boundary_check_is_not_vacuous(run_forward_point):
    """The robot must move in x within an episode, or the boundary test lies.

    A carried `last_x` is only WRONG if the episode's final x differs from the
    next episode's reset x. The point robot always resets to exactly
    `self._start_x`, so under zero actions a stale last_x equals the fresh
    reset value and a reintroduced bug produces no error at all. This test
    exists because that is not hypothetical: it is what an earlier draft of
    the test above did, and it passed against an injected bug.
    """
    env = run_forward_point
    site = env._robot_site_id
    pairs = _rollout(env, n_steps=3 * EPISODE_LENGTH + 2)

    # Largest within-episode x excursion away from the spawn line.
    drift = max(
        abs(float(new.data.site_xpos[0, site, 0]) - env._start_x) for _, new in pairs
    )
    assert drift > 1e-3, (
        f"robot never left the start line (max drift {drift:.2e} m), so the "
        "boundary tests cannot distinguish a carried scalar from a correct "
        "one. Use actions that actually produce motion."
    )


def test_go_to_goal_reward_is_correct_across_the_boundary(go_to_goal_point):
    """Same invariant for GoToGoal, whose carried scalar was `last_goal_dist`.

    Distances are read from `mocap_pos`, NOT `xpos`, and that distinction is
    load-bearing: `_reset_goal` writes mocap_pos without re-running forward
    kinematics, so xpos holds the OLD goal for one step after a mid-episode
    respawn. The naive xpos version of the fix was measured paying a
    STATIONARY robot -1.63 -- strictly worse than the bug it replaced.
    """
    env = go_to_goal_point
    site, mocap = env._robot_site_id, env._goal_mocap_id
    pairs = _rollout(env, n_steps=3 * EPISODE_LENGTH + 2)

    def goal_dist(state):
        return float(
            jp.linalg.norm(
                state.data.mocap_pos[0, mocap][:2] - state.data.site_xpos[0, site][0:2]
            )
        )

    boundaries_seen = 0
    for i, (prev, new) in enumerate(pairs):
        if bool(new.done[0] > 0):
            continue
        # The +1 goal bonus fires on reaching a goal and is not a distance
        # delta; skip those steps rather than reimplement the bonus here.
        if float(new.info["goal_reached"][0]) > 0:
            continue
        just_reset = bool(prev.done[0] > 0)
        expected = goal_dist(prev) - goal_dist(new)
        assert float(new.reward[0]) == pytest.approx(expected, abs=1e-5), (
            f"step {i}: reward != goal-distance delta"
            + (" -- STEP AFTER A BOUNDARY (the last_goal_dist bug)"
               if just_reset else "")
        )
        boundaries_seen += just_reset

    assert boundaries_seen >= 2


def test_goal_respawn_uses_mocap_pos_not_xpos(go_to_goal_point):
    """The naive fix for the boundary bug was WORSE than the bug.

    `_reset_goal` writes `mocap_pos` without re-running forward kinematics, so
    `data.xpos[goal_body]` still holds the OLD goal for one step after a
    mid-episode respawn. Differencing against xpos therefore pays a STATIONARY
    robot the full distance to the new goal as negative reward -- measured at
    -1.63 where the correct value is 0.00, several times per episode.

    This needs its own test because the boundary tests above cannot see it: a
    random policy essentially never reaches a goal (200 seeds x 400 steps is
    only 1.6 s of sim), so the respawn path is never taken. Verified: injecting
    the xpos version passes every other test in this file. The respawn is
    forced here by moving the goal onto the robot.
    """
    env = go_to_goal_point
    site, mocap = env._robot_site_id, env._goal_mocap_id
    step = jax.jit(env.step)
    zero = jp.zeros(env.action_size)

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    # Drop the goal onto the robot so the next step trips `goal_dist < 0.3`.
    robot_xy = state.data.site_xpos[site][:2]
    state = state.replace(
        data=state.data.replace(
            mocap_pos=state.data.mocap_pos.at[mocap, :2].set(robot_xy)
        )
    )

    state = step(state, zero)  # reaches the goal -> _reset_goal fires
    assert float(state.info["goal_reached"]) > 0, "respawn was not triggered"

    x_before = np.asarray(state.data.site_xpos[site][:2])
    state = step(state, zero)  # the step the bug corrupts
    moved = float(np.linalg.norm(np.asarray(state.data.site_xpos[site][:2]) - x_before))

    # A robot that barely moved cannot legitimately earn a large reward.
    assert abs(float(state.reward)) < moved + 1e-3, (
        f"reward {float(state.reward):.4f} on a step where the robot moved only "
        f"{moved:.2e} m -- the goal distance is being differenced across the "
        "respawn, i.e. xpos (stale) is being read instead of mocap_pos"
    )


def test_reset_and_step_agree_on_info_keys(run_forward_point, go_to_goal_point):
    """A key present in only one of reset/step is a pytree structure mismatch.

    `BraxAutoResetWrapper` selects between the live state and the reset state
    with `jax.tree.map`, so the two must have identical structure. Getting this
    wrong fails at trace time with a tree-mismatch error rather than silently,
    but it fails deep inside the wrapper stack where the message names no key.
    """
    for env in (run_forward_point, go_to_goal_point):
        reset_state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        step_state = jax.jit(env.step)(reset_state, jp.zeros(env.action_size))
        assert set(reset_state.info) == set(step_state.info), (
            f"{type(env).__name__}: reset/step info keys differ: "
            f"{set(reset_state.info) ^ set(step_state.info)}"
        )


def test_run_forward_carries_no_position_state(run_forward_point):
    """RunForward must keep NOTHING position-like in info.

    The general rule the boundary bug taught: nothing position-like may live in
    `state.info`, because the auto-reset wrapper does not restore it. This is a
    structural guard -- it catches a reintroduction at the point of writing
    rather than via a subtly wrong reward curve weeks later.
    """
    state = jax.jit(run_forward_point.reset)(jax.random.PRNGKey(0))
    banned = {"last_x", "last_goal_dist", "prev_x", "start_x", "last_pos"}
    assert not (banned & set(state.info)), (
        f"position-like keys in RunForward info: {banned & set(state.info)}. "
        "These survive the auto-reset boundary -- derive from state.data."
    )


def test_episode_return_equals_net_forward_displacement(run_forward_point):
    """RunForward's whole design claim: episode return IS metres travelled.

    The reward telescopes to x_final - x_initial, which is what makes
    `eval/episode_reward` directly readable as distance and what makes the task
    escape the freeze attractor that killed GoToGoal for the ant. Verified
    exactly (0.00e+00) when the task was written; pinned here because a stray
    additive term would break the interpretation of every logged number without
    breaking training.

    Uses healthy_reward=0 (the point's default), since an upright bonus is by
    design NOT a displacement.
    """
    env = run_forward_point
    assert env._healthy_reward == 0.0, "this invariant only holds without a bonus"

    site = env._robot_site_id
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(3))
    x0 = float(state.data.site_xpos[site, 0])
    key = jax.random.PRNGKey(4)
    total = 0.0
    for _ in range(25):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
        total += float(state.reward)
    x1 = float(state.data.site_xpos[site, 0])

    np.testing.assert_allclose(total, x1 - x0, atol=1e-5)

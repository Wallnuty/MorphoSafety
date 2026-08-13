"""Configuration invariants that failed silently or failed late.

None of these are about physics; they are about the plumbing between the CLI,
the wrappers and the checkpoint writer -- the layer where this project has
repeatedly lost whole runs to something that raised no error until hours in.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import pytest

from mjx_safety_gym.algorithms import train_ppo
from mjx_safety_gym.algorithms.train_ppo import (
    _ROBOT_DEFAULTS,
    apply_robot_defaults,
    wrap_for_brax_training,
)
from mjx_safety_gym.envs.go_to_goal import _ROBOT_XMLS


def _parse(argv):
    """Parse a command line exactly as `main()` does, defaults included."""
    return train_ppo.build_argparser().parse_args(argv)


# -- per-robot defaults ----------------------------------------------------


def test_every_robot_has_defaults():
    """A robot with no `_ROBOT_DEFAULTS` entry raises KeyError at startup.

    Cheap guard on adding a robot XML and forgetting the config half.
    """
    assert set(_ROBOT_DEFAULTS) == set(_ROBOT_XMLS), (
        f"robots without training defaults: {set(_ROBOT_XMLS) - set(_ROBOT_DEFAULTS)}"
    )


@pytest.mark.parametrize("robot", sorted(_ROBOT_DEFAULTS))
def test_defaults_fill_in_unset_flags(robot):
    args = _parse(["--robot", robot])
    for name in _ROBOT_DEFAULTS[robot]:
        assert getattr(args, name) is None, (
            f"--{name} must parse with default=None so an explicit CLI value is "
            "distinguishable from an unset one"
        )
    apply_robot_defaults(args)
    for name, value in _ROBOT_DEFAULTS[robot].items():
        assert getattr(args, name) == value


def test_explicit_flags_are_never_overwritten_by_defaults():
    """The whole point of `default=None`: the user's value wins."""
    args = _parse(["--robot", "ant_gym", "--episode_length", "77",
                   "--action_repeat", "3", "--discounting", "0.5"])
    apply_robot_defaults(args)
    assert (args.episode_length, args.action_repeat, args.discounting) == (77, 3, 0.5)


def test_point_defaults_are_frozen():
    """Point is the only robot with converged baselines; its config must not move.

    These three values were bit-identical before and after `_ROBOT_DEFAULTS`
    was introduced, deliberately, so that point baselines stayed valid across
    that change. Anything that edits them invalidates comparisons silently.
    """
    assert _ROBOT_DEFAULTS["point"] == {
        "action_repeat": 4, "episode_length": 1000, "discounting": 0.9,
        "healthy_reward": 0.0, "terminate_on_flip": False,
    }


def test_ants_get_an_upright_bonus_and_flip_termination():
    """Without these the ant trains upside down -- measured 94% of all steps.

    1.5M steps of training moved uprightness by 0.3 points because nothing in
    the reward mentioned being upright and falling had no consequence. Every
    working ant in safety-gymnasium, CRAX and Gym pairs the morphology with a
    healthy bonus AND termination; we had adopted the morphology alone.
    """
    for robot in ("ant", "ant_gym"):
        d = _ROBOT_DEFAULTS[robot]
        assert d["healthy_reward"] > 0.0, f"{robot} has no upright bonus"
        assert d["terminate_on_flip"] is True, f"{robot} does not terminate on flip"


def test_healthy_bonus_cannot_be_farmed_by_freezing():
    """Sizing constraint, not a style preference.

    A zero-action ant_gym was measured to stay upright for 100% of an episode,
    so the bonus is fully collectable by doing NOTHING -- it re-creates the
    freeze attractor that RunForward exists to escape if it is large enough to
    compete with walking. An earlier 0.005/step paid a frozen ant 12.5 over an
    episode, inside the range an early walking policy earns. Keep the
    full-episode bonus well under the metres an early gait can travel.
    """
    for robot in ("ant", "ant_gym"):
        d = _ROBOT_DEFAULTS[robot]
        per_episode = d["healthy_reward"] * d["episode_length"] / d["action_repeat"]
        assert per_episode <= 5.0, (
            f"{robot}: a frozen policy farms {per_episode:.1f} reward per episode "
            "just by standing still. Reward on this task IS metres travelled, so "
            "this competes directly with learning to walk."
        )


# -- checkpoint paths ------------------------------------------------------


def test_checkpoint_logdir_is_made_absolute(tmp_path, monkeypatch):
    """orbax rejects relative paths -- but only at SAVE time, i.e. after eval.

    A relative path therefore does not fail at startup; it fails once the
    training is already done. A 1.5M-step run was lost to exactly this: it
    trained 55 minutes, printed its final eval, then threw on every checkpoint
    write and left nothing on disk. `cluster/ant_gym_run_baseline.sbatch`
    passes a relative path, so that 8.5h job would have saved nothing.
    """
    monkeypatch.chdir(tmp_path)
    args = _parse(["--robot", "point", "--checkpoint_logdir", "checkpoints/rel"])
    resolved = train_ppo.resolve_checkpoint_logdir(args)
    assert resolved is not None
    assert resolved.is_absolute(), f"{resolved} is relative -- orbax will reject it"


def test_no_checkpoint_disables_the_logdir():
    args = _parse(["--robot", "point", "--no_checkpoint"])
    assert train_ppo.resolve_checkpoint_logdir(args) is None


# -- action_repeat semantics -----------------------------------------------


def test_action_repeat_sums_reward_and_advances_steps_by_the_repeat(run_forward_point):
    """`num_timesteps` counts PHYSICS steps, not decisions, and this is why.

    `CostEpisodeWrapper` holds each action for `action_repeat` inner steps,
    SUMS reward and cost over them, and grows `info["steps"]` by
    `action_repeat` -- so an episode of `episode_length` inner steps takes
    `episode_length // action_repeat` decisions.

    scripts/eval_checkpoint.py ignored all of this and ran 625 raw steps with a
    fresh action each: a QUARTER of an episode at FOUR TIMES the control rate.
    It does not cancel as a scale factor -- this project's own sweep measured
    control period as worth 2.7x in achievable gait travel -- and it produced a
    13x disagreement in episode_cost against the training log.
    """
    env = run_forward_point
    repeat, length = 4, 16
    wrapped = wrap_for_brax_training(env, episode_length=length, action_repeat=repeat)
    step = jax.jit(wrapped.step)

    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 1))
    assert int(state.info["steps"][0]) == 0

    n_decisions = 0
    while not bool(state.done[0]):
        state = step(state, jp.zeros((1, env.action_size)))
        n_decisions += 1
        assert int(state.info["steps"][0]) == n_decisions * repeat, (
            "info['steps'] must advance by action_repeat per decision"
        )
        assert n_decisions <= length, "episode never terminated"

    assert n_decisions == length // repeat, (
        f"episode took {n_decisions} decisions, expected {length // repeat} "
        f"(episode_length {length} / action_repeat {repeat})"
    )


def test_single_decision_reward_equals_the_sum_over_the_repeat(run_forward_point):
    """The summing half of the same contract, checked against the raw env."""
    env = run_forward_point
    repeat = 4
    wrapped = wrap_for_brax_training(env, episode_length=1000, action_repeat=repeat)

    key = jax.random.PRNGKey(11)
    action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)

    wrapped_state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(5), 1))
    wrapped_state = jax.jit(wrapped.step)(wrapped_state, action[None])

    raw = jax.jit(env.reset)(jax.random.PRNGKey(5))
    raw_step = jax.jit(env.step)
    total = 0.0
    for _ in range(repeat):
        raw = raw_step(raw, action)
        total += float(raw.reward)

    assert float(wrapped_state.reward[0]) == pytest.approx(total, abs=1e-5)


# -- observation width -----------------------------------------------------


@pytest.mark.parametrize("robot", ["point", "ant_gym"])
def test_observation_size_matches_the_actual_observation(robot):
    """`observation_size` builds the PPO network; a mismatch is a shape error
    deep in the training loop rather than at env construction."""
    from mjx_safety_gym.envs.run_forward import RunForward

    env = RunForward(robot=robot)
    obs = jax.jit(env.reset)(jax.random.PRNGKey(0)).obs
    assert obs.shape == (env.observation_size,)


def test_run_forward_keeps_go_to_goal_observation_width():
    """RunForward parks the goal instead of deleting it, specifically so that
    obs width matches GoToGoal and checkpoints/networks stay interchangeable.
    """
    from mjx_safety_gym.envs.go_to_goal import GoToGoal
    from mjx_safety_gym.envs.run_forward import RunForward

    for robot in ("point", "ant_gym"):
        assert RunForward(robot=robot).observation_size == GoToGoal(
            robot=robot
        ).observation_size

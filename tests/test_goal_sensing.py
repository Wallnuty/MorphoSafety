"""Goal bearing/range in the observation, and the distance-delta reward.

Added after measuring that the trained ant walked in an essentially arbitrary
direction -- net +x median +1.6 m over a range of -7.8 to +11.9, i.e. half of
all episodes ended BEHIND the start line, with 99.1% of cost coming from the
corridor boundary. It could not have done better: the sensor block is
bit-identical at y=0, y=0.99 and y=3.0, and the goal lidar ring read exactly
zero in all 10,000 observations sampled.

The failure mode these guard against is a WIDTH mismatch. `goal_observation`
changes observation width, so any script that builds the env with bare
defaults constructs a network of the wrong shape for the checkpoint it is
about to load -- and the corridor tasks are built in four separate places
(train_ppo, main.py, eval_checkpoint.py, and the tests).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

from mjx_safety_gym.algorithms import train_ppo
from mjx_safety_gym.envs.run_forward import RunForward


@pytest.fixture(scope="session")
def goal_env():
    return RunForward(robot="point", goal_observation=True, goal_reward_weight=1.0)


# -- width ------------------------------------------------------------------


def test_goal_observation_adds_exactly_three_dims(goal_env, run_forward_point):
    assert goal_env.observation_size == run_forward_point.observation_size + 3
    state = jax.jit(goal_env.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape == (goal_env.observation_size,)


def test_reported_width_matches_actual_width(goal_env):
    """observation_size is what builds the network; get_obs is what feeds it.

    They are computed in different places (task_observation_size vs
    task_observations), so they can drift apart -- and the failure is a shape
    error at load time, far from the cause.
    """
    state = jax.jit(goal_env.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape[0] == goal_env.observation_size


def test_replay_and_eval_build_the_same_env_as_training():
    """main.py and eval_checkpoint.py must not construct a narrower env.

    Both used to call RunForward(robot=...) bare. With goal_observation
    defaulted ON for the ants that silently yields a 76-wide env for a 79-wide
    checkpoint, so they now go through robot_env_kwargs -- this asserts that
    table actually turns the flag on.
    """
    for robot in ("ant", "ant_gym"):
        kwargs = train_ppo.robot_env_kwargs(robot)
        assert kwargs["goal_observation"] is True, robot
        assert kwargs["goal_reward_weight"] > 0, robot
    assert train_ppo.robot_env_kwargs("point")["goal_observation"] is False
    # Every key must be a real constructor argument, or replay crashes.
    import inspect

    params = inspect.signature(RunForward.__init__).parameters
    for robot in ("point", "ant", "ant_gym"):
        for key in train_ppo.robot_env_kwargs(robot):
            assert key in params, f"{key} is not a RunForward argument"


def test_morphology_genes_stay_at_the_tail():
    """Goal features are appended BEFORE the genes.

    MorphologyDomainRandomizationWrapper writes genes per-lane and reads them
    back from the obs tail; putting anything after them silently misaligns the
    entire morphology-conditioning path.
    """
    from mjx_safety_gym.morphology import NUM_GENES

    env = RunForward(robot="point", goal_observation=True,
                     morphology_conditioning=True)
    genes = jp.arange(NUM_GENES, dtype=jp.float32) / NUM_GENES
    env._morphology_genes = genes
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(state.obs[-NUM_GENES:]),
                               np.asarray(genes), rtol=1e-6)


# -- loading older checkpoints ----------------------------------------------


def test_checkpoint_obs_width_reads_the_first_layer():
    """Width is recovered from the weights, since nothing records it.

    Training writes an empty ConfigDict, so a checkpoint carries no statement
    of the observation width it was built for -- only the shape of `hidden_0`'s
    kernel, which is (obs_width, hidden_width).
    """
    params = {"params": {"hidden_0": {"kernel": np.zeros((76, 32))},
                         "hidden_1": {"kernel": np.zeros((32, 32))}}}
    assert train_ppo.checkpoint_obs_width(params) == 76
    # Unrecognised layouts must say "don't know", never guess.
    assert train_ppo.checkpoint_obs_width({"params": {}}) is None
    assert train_ppo.checkpoint_obs_width(None) is None


def test_old_narrow_checkpoints_still_load():
    """A pre-goal-sensing (76-wide) checkpoint must still be replayable.

    This is the crash it fixes, verbatim:
        ScopeParamShapeError: Initializer expected to generate shape (76, 32)
        but got shape (79, 32) ... for parameter "kernel" in "/hidden_0"
    Both main.py and scripts/eval_checkpoint.py hit it on every ant checkpoint
    trained before 2026-08-15.
    """
    build = lambda **kw: RunForward(robot="point", **kw)
    wide = RunForward(robot="point", goal_observation=True).observation_size
    narrow = RunForward(robot="point", goal_observation=False).observation_size

    # Asking for the narrow width must flip goal_observation off...
    env, kwargs, default_width = train_ppo.build_env_for_checkpoint(
        build, "ant", narrow
    )
    assert env.observation_size == narrow
    assert kwargs["goal_observation"] is False
    # ...and take the goal REWARD with it: a checkpoint from before one is from
    # before the other, so a replay must not report a reward it never trained on.
    assert kwargs["goal_reward_weight"] == 0.0
    # The reported default must be the width the DEFAULTS gave, not the width
    # that was settled on -- an inverted version of this printed "73" for a
    # 79-wide default, i.e. it was off by twice the flag's contribution.
    assert default_width == wide

    # Asking for the width the defaults already give must NOT rebuild.
    env, kwargs, default_width = train_ppo.build_env_for_checkpoint(build, "ant", wide)
    assert default_width is None and env.observation_size == wide
    assert kwargs["goal_observation"] is True

    # Unknown width (unreadable checkpoint) leaves the defaults alone.
    _, kwargs, default_width = train_ppo.build_env_for_checkpoint(build, "ant", None)
    assert default_width is None and kwargs["goal_observation"] is True


def test_irreconcilable_width_fails_with_a_useful_message():
    """A width no flag can produce means the wrong ROBOT, and must say so
    rather than dying inside flax."""
    build = lambda **kw: RunForward(robot="point", **kw)
    with pytest.raises(SystemExit, match="different robot"):
        train_ppo.build_env_for_checkpoint(build, "ant", 999)


# -- the signal itself ------------------------------------------------------


def test_goal_features_encode_bearing_and_range(goal_env):
    """cos/sin of the goal's relative bearing, and normalised distance."""
    env = goal_env
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    adr = env._robot_free_qposadr
    slide = env._robot_slide_qposadr

    def features_at(x, y):
        qpos = state.data.qpos
        if slide is not None:
            qpos = qpos.at[jp.array(slide)].set(jp.array([x, y]))
        else:
            qpos = qpos.at[adr : adr + 2].set(jp.array([x, y]))
        d = mjx.forward(env.mjx_model, state.data.replace(qpos=qpos))
        return np.asarray(env.task_observations(d)), float(env.goal_distance(d))

    # Directly on the centreline, goal straight ahead: cos=1, sin=0.
    f, dist = features_at(env._start_x, 0.0)
    assert f[0] == pytest.approx(1.0, abs=1e-3)
    assert f[1] == pytest.approx(0.0, abs=1e-3)
    assert f[2] == pytest.approx(dist / env._corridor_length, rel=1e-4)

    # Off to one side, the bearing must tilt the other way -- and the SIGN must
    # differ between the two sides, which is the whole point.
    f_left, _ = features_at(env._start_x, +1.0)
    f_right, _ = features_at(env._start_x, -1.0)
    assert np.sign(f_left[1]) == -np.sign(f_right[1])
    assert abs(f_left[1]) > 1e-3, "lateral offset produced no bearing change"

    # Range must grow as the robot falls back.
    _, near = features_at(0.0, 0.0)
    _, far = features_at(env._start_x, 0.0)
    assert far > near


def test_goal_reward_telescopes_to_distance_closed(goal_env):
    """Summed reward must equal (d_initial - d_final).

    This is the property that makes the reward readable as metres, and it is
    also the episode-boundary guard: the previous distance is recomputed from
    prev_data rather than carried in state.info, because BraxAutoResetWrapper
    restores data but not info.
    """
    env = goal_env
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    d0 = float(env.goal_distance(state.data))

    rng = jax.random.PRNGKey(3)
    total = 0.0
    for _ in range(40):
        rng, k = jax.random.split(rng)
        action = jax.random.uniform(k, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
        total += float(state.reward)
    d1 = float(env.goal_distance(state.data))

    # forward_reward_weight is still 1.0 by default, so subtract the +x term.
    x_term = float(state.data.site_xpos[env._robot_site_id][0]) - env._start_x
    assert total - x_term == pytest.approx(d0 - d1, abs=1e-4)
    assert abs(d0 - d1) > 1e-6, "robot did not move; the check proves nothing"


def test_lateral_drift_is_penalised_unlike_pure_forward_reward(goal_env):
    """The reason for the whole change.

    Pure +x reward is flat in y, so sidestepping costs nothing. The
    distance-delta term must charge for it, or there is still no preference for
    going straight.
    """
    env = goal_env
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    slide, adr = env._robot_slide_qposadr, env._robot_free_qposadr

    def data_at(x, y):
        qpos = state.data.qpos
        if slide is not None:
            qpos = qpos.at[jp.array(slide)].set(jp.array([x, y]))
        else:
            qpos = qpos.at[adr : adr + 2].set(jp.array([x, y]))
        return mjx.forward(env.mjx_model, state.data.replace(qpos=qpos))

    here = data_at(env._start_x, 0.0)
    sideways = data_at(env._start_x, 0.5)      # pure lateral move
    forward = data_at(env._start_x + 0.5, 0.0)  # same distance, toward the goal

    r_side = float(env.get_reward(sideways, here))
    r_fwd = float(env.get_reward(forward, here))
    assert r_fwd > r_side, "moving toward the goal must beat moving sideways"
    assert r_side < 0, "a purely lateral move must lose ground, not be free"

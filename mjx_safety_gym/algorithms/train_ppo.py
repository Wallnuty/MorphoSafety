"""Train PPO + CRPO/Lagrangian on GoToGoal.

Minimal, Hydra-free training entry point, styled like the top-level
main.py demo script. Ported/assembled from safe-learning (ss2r)'s
ss2r/algorithms/ppo and ss2r/benchmark_suites/mujoco_playground.

Defaults follow ss2r's reference config for this task
(ss2r/configs/experiment/go_to_goal_simple_ppo.yaml + configs/agent/ppo.yaml),
except for `--num_envs`/`--num_evals`, which are sized for a 6 GiB laptop GPU
rather than the reference's 2048 envs -- see `--num_envs` help text.

Usage:
    python -m mjx_safety_gym.algorithms.train_ppo --robot point \
        --penalizer crpo --num_timesteps 200_000
"""

import argparse
import functools
from pathlib import Path
from typing import Optional

import jax
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper

from mjx_safety_gym import jax_cache
from mjx_safety_gym import morphology as morphology_lib
from mjx_safety_gym.algorithms.penalizers import get_penalizer
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.algorithms.ppo import train as ppo_train
from mjx_safety_gym.algorithms.wrappers import (
    CostEpisodeWrapper,
    MorphologyDomainRandomizationWrapper,
    Saute,
)
from mjx_safety_gym.envs.go_to_goal import _ROBOT_XMLS, GoToGoal
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.envs.run_forward import RunForward


# Checkpoints live outside the package so they are not swept up by a package
# install or by `setuptools.packages.find`. Both this module and main.py resolve
# the same convention, so a trained policy is picked up for replay without
# anyone having to pass a path around.
CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"


# Per-robot defaults for the settings whose correct value depends on how fast
# the robot physically moves. Everything else is shared.
#
# NOTE: --safety_discounting is left at 0.9 for both robots on purpose. The
# same slow-timescale argument applies to the cost critic, but changing it
# also shifts the constraint's semantics while the safety-budget calibration
# is still an open question -- a research call, not a bug fix.
# !! action_repeat is NOT the frameskip. GoToGoal.step calls mjx_env.step with
# n_substeps=2, so the real control period is action_repeat * 2 * timestep, and
# simulated seconds per episode is
#     (episode_length / action_repeat) * action_repeat * 2 * timestep
#   = episode_length * 2 * timestep.
# An earlier version of this table set the ant to action_repeat=10 to match
# safety-gymnasium's frameskip_binom_n=10, having missed that n_substeps=2 was
# already in the chain -- that made the control period 0.2 s, DOUBLE theirs.
# Measured cost of getting this wrong (scripted-gait sweep, 20 s of sim time,
# best gait period per row):
#
#     action_repeat  control period  best gait travel  random travel
#                 2          0.04 s           2.718 m        0.276 m
#                 4          0.08 s           4.090 m        0.620 m   <- best
#                 6          0.12 s           1.979 m        0.635 m
#                 8          0.16 s           2.133 m        0.637 m
#                10          0.20 s           1.499 m        0.634 m
#
# Random-action travel saturates by action_repeat=4, so coarser control buys no
# extra exploration, while the achievable-locomotion CEILING falls 2.7x from 4
# to 10. action_repeat stays at 4 for both robots; episode duration is bought
# with episode_length instead, which is free of that tradeoff.
#
# The ant's episode_length of 2500 is 625 decisions = 50 s of simulated time
# (against the point's 1000 -> 250 decisions = 20 s). The ant needs it: at its
# measured best of ~0.2 m/s it covers ~10 m in 50 s, which is what makes a
# 12 m corridor (envs/run_forward.py) a task it can actually make progress on.
# safety-gymnasium gives its own ant 100 s per episode and its point 20 s.
#
# discounting 0.97 at a 0.08 s control period is a ~2.7 s horizon, about seven
# gait cycles at the measured 0.4 s best gait period. The point keeps ss2r's
# 0.9 (a 0.4 s horizon), which is fine for a robot whose "gait" is one actuator.
# healthy_reward / terminate_on_flip are point-exempt on purpose: the point
# robot is a slider-driven puck with no meaningful "upright", so inverting is
# not a failure mode it can have, and terminating it on torso orientation
# would be nonsense. The ants get both -- see RunForward.is_upright for the
# measurement that motivated them.
_ROBOT_DEFAULTS = {
    "point": {
        "action_repeat": 4, "episode_length": 1000, "discounting": 0.9,
        "healthy_reward": 0.0, "terminate_on_flip": False,
        "goal_reward_weight": 0.0, "goal_observation": False,
        "terminate_on_goal": False,
    },
    "ant": {
        "action_repeat": 4, "episode_length": 2500, "discounting": 0.97,
        # 0.0002, NOT ant_gym's 0.002, and that 10x is measured rather than
        # taste. The bonus is paid per INNER step, so a full episode pays
        # healthy_reward * episode_length -- 5.000 at 0.002, confirmed exactly
        # by rolling out a zero-action policy. Against ant_gym's best scripted
        # gait (40.93 m) that is 12% of what locomotion can earn. Against THIS
        # ant's best scripted gait (3.03 m, it has 46x less torque per kg) the
        # same 0.002 pays 165% of everything walking could ever earn, and a
        # random policy measured 4.789 against a frozen 5.000 -- i.e. the
        # reward was almost entirely a constant collected for existing. That is
        # exactly the freeze attractor RunForward exists to escape.
        #
        # 0.0002 pays 0.5/episode = 16% of achievable, matching ant_gym's ratio.
        #
        # terminate_on_flip stays ON but is nearly inert here and that is fine:
        # this ant flipped in 0.0% of zero-action and 12.5% of random-action
        # seeds, against ant_gym's 94% inversion. The upright problem is
        # ant_gym's, caused by its torque; this is cheap insurance in case a
        # trained policy moves fast enough to tip itself.
        "healthy_reward": 0.0002, "terminate_on_flip": True,
        #
        # GOAL SENSING ON BY DEFAULT (2026-08-15). Measured on the 1.68M-step
        # checkpoint over 16 episodes: net +x median +1.6 m with a range of
        # -7.8 to +11.9, i.e. HALF THE EPISODES END BEHIND THE START LINE. It
        # walks fine; it has no idea which way to walk. 99.1% of its cost was
        # the corridor boundary, first crossed at decision 34 of 625.
        #
        # Nothing in the observation could have told it: the sensor block is
        # bit-identical at y=0, y=0.99 and y=3.0, and the goal lidar ring read
        # exactly zero in all 10,000 observations sampled (goal 11 m away,
        # LIDAR_MAX_DIST 2.0). See envs/run_forward.py's goal-sensing section.
        #
        # NOTE THIS CHANGES OBSERVATION WIDTH, 76 -> 79 for the ants. Older ant
        # checkpoints will not load against it; pass --goal_observation false to
        # reproduce a pre-2026-08-15 run.
        "goal_reward_weight": 1.0, "goal_observation": True,
        # terminate_on_goal OFF (2026-08-17). Measured worth: the ants reach
        # the goal at decision 248 of 625, so 60% of every episode is
        # post-arrival dead time and turning this on is ~2.5x more useful
        # experience per env-step. Left off by default only so results recorded
        # before that date stay reproducible; turn it on for new work.
        "terminate_on_goal": False,
    },
    # ant_gym is 4x the ant's length scale but its measured best gait period is
    # similar (0.5 s vs 0.4 s), so the same control period applies. It travels
    # far more per second, which episode_length does not need to change for --
    # 50 s is ~20-40 m for it, matching the 48 m default corridor.
    #
    # DISCOUNTING 0.995, NOT THE 0.97 SHARED WITH ant. THIS IS THE FIX FOR
    # SPRINT-AND-FLIP AND IT IS NOT A REWARD CHANGE.
    #
    # Measured: at its best, ant_gym scored episode_reward 8.77 with
    # avg_episode_length 101.5 out of 2500 -- it terminates at 4% of the
    # episode. That is 2.0 s of sim, i.e. 4.32 m/s. It is not walking, it is
    # launching itself and falling over. The bonus is only 0.20 of that 8.77,
    # so the reward COEFFICIENTS are not what produces this.
    #
    # The discount factor is. At 0.97 with a 0.08 s control period the horizon
    # is 1/(1-g) = 33 decisions = 2.7 s, and the ant survives 25 decisions =
    # 2.0 s -- its planning horizon barely outlasts its own lifespan, and
    # reaching the end of the episode is discounted by 0.97^625 = 5.4e-09. The
    # ~41 m available from staying upright is INVISIBLE to the value function.
    # Comparing discounted values of the two strategies (sprint 0.338
    # m/decision for 25 decisions then terminate, vs a sustained 0.066
    # m/decision for all 625):
    #
    #     gamma   horizon    sprint V   walk V   better
    #     0.970     2.7 s       6.06     2.19   sprint   <- was here
    #     0.990     8.0 s       7.60     6.55   sprint
    #     0.995    16.0 s       8.06    12.55   WALK     <- now here
    #     0.998    40.0 s       8.36    23.41   WALK
    #
    # Undiscounted the ordering is never in doubt (8.6 m vs 41.0 m). PPO was
    # optimising the objective we gave it correctly; the objective was wrong.
    # 0.995 is the first value that flips the ordering, with margin, without
    # going to 0.998 where value-function variance gets unpleasant.
    #
    # The old 0.97 was chosen so the horizon covered "about seven gait cycles
    # at the measured 0.4 s best gait period" -- sound for LEARNING A GAIT, and
    # silent about SURVIVING AN EPISODE. That tradeoff only became visible once
    # terminate_on_flip existed, which came later.
    #
    # ant stays at 0.97 deliberately: it runs 2100-2500 steps, so it is not
    # dying inside its horizon and this argument does not apply to it. Change
    # one robot at a time.
    #
    # NOT YET VALIDATED. Raising gamma raises return variance and value bias,
    # so a run may look worse before better. The matched A/B is against the
    # 8.77 m checkpoint in checkpoints/ant_gym_upright_chain/gen1.
    "ant_gym": {
        "action_repeat": 4, "episode_length": 2500, "discounting": 0.995,
        # 0.002/step is ~5 over a full 2500-step episode. Sized against what a
        # FROZEN policy can farm, not against the best gait: measured, a
        # zero-action ant_gym stays upright 100% of the episode, so the bonus
        # is fully collectable by doing nothing. It only has to beat flailing
        # (the 1.5M policy managed +0.7 m of net +x while inverted) while
        # staying well under a plausible LEARNED walk. An earlier 0.005 paid a
        # frozen ant 12.5, which is inside the range an early walking policy
        # would earn -- that re-creates the freeze attractor this task exists
        # to escape, and the ~40 m figure it was sized against is a best-case
        # SCRIPTED gait, not something a policy reaches early.
        "healthy_reward": 0.002, "terminate_on_flip": True,
        # Same reasoning as ant's, above. ant_gym's corridor is 48 m and its
        # goal sits 44 m from the start line, so its goal lidar ring is even
        # further out of range than ant's -- it has never had a heading signal
        # either. Observation width 76 -> 79.
        "goal_reward_weight": 1.0, "goal_observation": True,
        # terminate_on_goal OFF (2026-08-17). Measured worth: the ants reach
        # the goal at decision 248 of 625, so 60% of every episode is
        # post-arrival dead time and turning this on is ~2.5x more useful
        # experience per env-step. Left off by default only so results recorded
        # before that date stay reproducible; turn it on for new work.
        "terminate_on_goal": False,
    },
}


def apply_robot_defaults(args: argparse.Namespace) -> None:
    """Fill in per-robot defaults for flags left unset on the command line.

    These three flags parse with `default=None` precisely so an explicit CLI
    value is distinguishable from an unset one; anything the user passes is
    left untouched.
    """
    resolved = []
    for name, value in _ROBOT_DEFAULTS[args.robot].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
            resolved.append(f"{name}={value}")
    if resolved:
        print(f"[{args.robot}] robot defaults applied: {', '.join(resolved)}")


# Which _ROBOT_DEFAULTS keys are constructor arguments of the corridor envs.
# The rest (action_repeat, episode_length, discounting) are training knobs and
# are not accepted by the env.
_ENV_DEFAULT_KEYS = (
    "healthy_reward", "terminate_on_flip", "goal_reward_weight",
    "goal_observation", "terminate_on_goal",
)


def robot_env_kwargs(robot: str) -> dict:
    """Corridor-env constructor kwargs matching what training uses for `robot`.

    Replay and evaluation must build their env through this rather than with
    bare defaults. `goal_observation` changes the observation WIDTH (76 -> 79
    for the ants), so a script that constructs `RunForward(robot=...)` plainly
    gets a 76-wide env and then loads a 79-wide checkpoint into it -- which
    fails loudly if you are lucky and silently mis-shapes the policy if you are
    not. The non-width flags matter too: replaying without `terminate_on_flip`
    shows episodes that training would have ended.
    """
    defaults = _ROBOT_DEFAULTS[robot]
    return {k: defaults[k] for k in _ENV_DEFAULT_KEYS if k in defaults}


def checkpoint_obs_width(policy_params) -> Optional[int]:
    """Observation width the saved policy's first layer was built for.

    Read off the WEIGHTS, not from a saved config: training writes an empty
    ConfigDict, so a checkpoint records nothing at all about the env it came
    from. `hidden_0` is brax's name for the first Dense layer of an MLP and its
    kernel is (obs_width, hidden_width), so the width survives in the shape
    even though nobody wrote it down.

    None if the layout is not what we expect, which callers treat as "cannot
    tell" and proceed.
    """
    try:
        return int(policy_params["params"]["hidden_0"]["kernel"].shape[0])
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def checkpoint_policy_layers(policy_params) -> Optional[tuple[int, ...]]:
    """Hidden layer WIDTHS the saved policy was built with.

    Same trick and same reason as checkpoint_obs_width above: the checkpoint
    records nothing about its network, so the parameter shapes are the only
    surviving evidence. Needed because --policy_hidden_layer_sizes is not the
    default for morphology-conditioned runs -- those use (256,)*4, since the
    (32,)*4 default is 8x narrower than the value nets and has to produce a
    different gait per body. Rebuilding such a checkpoint with the default
    raises a flax shape error rather than loading.

    The trailing Dense is the output head (2 * action_size), not a hidden
    layer, so it is dropped. None if the layout is unrecognised.
    """
    try:
        params = policy_params["params"]
        sizes, i = [], 0
        while f"hidden_{i}" in params:
            sizes.append(int(params[f"hidden_{i}"]["kernel"].shape[1]))
            i += 1
        return tuple(sizes[:-1]) if len(sizes) > 1 else None
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


# The full lidar stack the corridor tasks used before 2026-08-15. Kept only so
# older checkpoints can still be replayed -- see build_env_for_checkpoint.
_LEGACY_LIDAR_GROUPS = ("obstacle", "goal", "object")


def _env_kwarg_candidates(robot: str) -> list[dict]:
    """Env configurations to try against a checkpoint, current first.

    TWO separate changes have moved the corridor observation width, and a
    checkpoint records only the total, so reconciliation is a small search
    rather than a single flag flip:

        goal_observation   +3   (2026-08-15, ON by default for the ants)
        lidar_groups      +32   (2026-08-15, narrowed to the obstacle ring)

    For the ant that makes 44 (current), 47 (current + goal sensing), 76
    (legacy) and 79 (legacy + goal sensing) all reachable, and every ant
    checkpoint in the repo predates both changes at 76.
    """
    current = robot_env_kwargs(robot)
    legacy_reward = {**current, "goal_observation": False, "goal_reward_weight": 0.0}
    return [
        current,
        {**current, "goal_observation": not current.get("goal_observation", False),
         "goal_reward_weight": (
             0.0 if current.get("goal_observation") else current["goal_reward_weight"]
         )},
        {**legacy_reward, "lidar_groups": _LEGACY_LIDAR_GROUPS},
        {**current, "lidar_groups": _LEGACY_LIDAR_GROUPS},
    ]


def build_env_for_checkpoint(build, robot: str, want_width: Optional[int]):
    """Build a corridor env whose observation width matches a checkpoint's.

    `build` is called with env kwargs and returns the env.

    Exists because loading a checkpoint into a differently-shaped env raises
    deep inside flax:

        ScopeParamShapeError: Initializer expected to generate shape (76, 32)
        but got shape (79, 32) ... for parameter "kernel" in "/hidden_0"

    which names neither the flag responsible nor the checkpoint. The goal
    OBSERVATION and the goal REWARD always move together here, because a
    checkpoint from before one is from before the other, and replaying with a
    reward the policy never trained against would misreport its return.

    Returns (env, kwargs, default_width): `default_width` is None when the
    defaults already fitted, else the width they would have produced. It is
    returned rather than left for the caller to derive -- deriving it by
    re-applying a delta against the flag's new value is easy to get backwards
    (it was, and printed "73" for a 79-wide default).
    """
    default_width = None
    for i, kwargs in enumerate(_env_kwarg_candidates(robot)):
        env = build(**kwargs)
        if i == 0:
            default_width = env.observation_size
            if want_width is None or default_width == want_width:
                return env, kwargs, None
        if env.observation_size == want_width:
            return env, kwargs, default_width

    raise SystemExit(
        f"Cannot load this checkpoint against --robot {robot}.\n"
        f"  the checkpoint's policy expects an observation of width {want_width}\n"
        f"  this robot/task produces {default_width}\n"
        f"No combination of goal_observation and lidar_groups reaches "
        f"{want_width}, so the checkpoint was most likely trained on a "
        f"different robot (observation width also depends on the robot's "
        f"sensor count)."
    )


def default_checkpoint_dir(robot: str) -> Path:
    """Where runs for `robot` write checkpoints unless told otherwise."""
    return CHECKPOINT_ROOT / robot


def checkpoint_owner(dir_name: str) -> Optional[str]:
    """Which robot a checkpoint directory belongs to, or None.

    A directory belongs to `robot` if it is named exactly `robot` or starts
    with `robot_`. THE LONGEST MATCH WINS, and that is the whole point of this
    function: `ant` is a prefix of `ant_gym`, so `ant_gym_upright_chain` would
    otherwise be claimed by BOTH robots. Since the two ants share an
    observation width (76) and action size (8), a mismatched checkpoint loads
    without any error and silently replays the wrong robot's policy -- the
    same silent-wrong-thing failure class as the stale-checkpoint problem this
    function exists to fix.

    Requiring the separator (not a bare `startswith`) keeps a hypothetical
    `antelope` from being claimed by `ant`.
    """
    owners = [
        r for r in _ROBOT_XMLS
        if dir_name == r or dir_name.startswith(f"{r}_")
    ]
    return max(owners, key=len) if owners else None


def latest_checkpoint(robot: str) -> Optional[Path]:
    """Newest checkpoint for `robot` anywhere under CHECKPOINT_ROOT, or None.

    Searches EVERY run directory belonging to the robot, not just
    `checkpoints/<robot>/`. Runs land in per-experiment directories
    (`ant_run_chain/gen3/`, `ant_gym_upright_5M/`, ...), so the old
    `checkpoints/<robot>/`-only lookup silently replayed whatever last happened
    to be written there -- in practice a 573k-step GoToGoal policy from five
    days before the run the user actually wanted.

    NEWEST IS BY MTIME, NOT BY STEP NUMBER. Every resumed run restarts its own
    step counter at 0, so a chain's `gen5/000001679360` is *newer* training
    than `gen4/000005038080` while sorting lower either lexicographically or
    numerically. `scripts/train_chain.sh` picks its restore point the same way,
    for the same reason.

    Leaf step directories sit one level down for a plain run
    (`ant/000000573440`) and two for a chain (`ant_run_chain/gen5/000001679360`),
    so both depths are searched.
    """
    if not CHECKPOINT_ROOT.is_dir():
        return None

    candidates: list[Path] = []
    for run_dir in CHECKPOINT_ROOT.iterdir():
        if not run_dir.is_dir() or checkpoint_owner(run_dir.name) != robot:
            continue
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.isdigit():
                candidates.append(child)
            else:  # a generation directory: look one level deeper
                candidates.extend(
                    g for g in child.iterdir() if g.is_dir() and g.name.isdigit()
                )

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def wrap_for_brax_training(
    env,
    episode_length: int,
    action_repeat: int = 1,
    already_batched: bool = False,
):
    """Vmap + cost-aware episode wrapper + mujoco_playground's auto-reset.

    Mirrors ss2r's wrap_for_brax_training, but built from `CostEpisodeWrapper`
    (which threads info["cost"] through episode aggregation) instead of
    mujoco_playground's own wrap_for_brax_training (which uses brax's vanilla
    EpisodeWrapper and drops the "cost" key added later by the eval wrapper).

    `already_batched`, if set, skips `VmapWrapper`: `env` already introduces
    the num_envs batch dimension itself (morphology randomization's
    `MorphologyDomainRandomizationWrapper`, applied by the caller before this
    function -- see `train()`). Any obs-shape-changing wrapper (that one,
    `Saute`) MUST be applied to the raw env before this function is called,
    never wrapped around this function's return value -- see
    `MorphologyDomainRandomizationWrapper`'s docstring for why (a real crash,
    not a style preference): `CostEpisodeWrapper` carries `state` through its
    own internal `action_repeat` scan, so obs width has to be final and
    stable before it ever sees the env.
    """
    if not already_batched:
        env = brax_training.VmapWrapper(env)
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    env = playground_wrapper.BraxAutoResetWrapper(env)
    return env


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot", choices=sorted(_ROBOT_XMLS), default="point"
    )
    parser.add_argument(
        "--penalizer",
        choices=["crpo", "ppo_lagrangian", "saute", "none"],
        default="crpo",
    )
    parser.add_argument(
        "--safety_budget",
        type=float,
        default=150.0,
        help="Raised from ss2r's own reference value of 25: that number was "
        "calibrated for Saute+terminate=true (go_to_goal_simple_ppo.yaml), "
        "where hitting the budget ends the episode immediately, implicitly "
        "bounding cost. Under CRPO/Lagrangian (no early termination) cost "
        "runs for the full episode -- porting 25 across that mechanism "
        "change, on top of this repo's stricter surface-based (not centre-"
        "based) hazard cost, measurably produced a degenerate policy for the "
        "point (0.19 goals/ep vs a 3.94 unconstrained ceiling, even with a "
        "well-tuned multiplier). 150 sits above the untrained ant's step-0 "
        "cost of 108.94; still an extrapolation, not a measurement -- pair "
        "the first real run with a short unconstrained baseline to check it.",
    )
    parser.add_argument("--crpo_eta", type=float, default=0.0)
    parser.add_argument("--crpo_burnin", type=int, default=0)
    parser.add_argument(
        "--lagrangian_multiplier_lr",
        type=float,
        default=7e-7,
        help="ss2r's own default (agent/penalizer/ppo_lagrangian.yaml). The "
        "previous default of 1e-2 traces to ss2r's go1_sim_to_real "
        "experiment, an unrelated robot/task -- not validated for "
        "go_to_goal (whose own ss2r reference config uses Saute, not "
        "Lagrangian, and never overrides this). At 7e-7 the multiplier will "
        "move far more slowly than the 90-273 range measured at 1e-2 -- "
        "watch training logs for it staying near its initial value "
        "(under-enforcing) rather than assuming this is well-calibrated.",
    )
    parser.add_argument(
        "--initial_lagrange_multiplier",
        type=float,
        default=0.01,
        help="ss2r's own default (same file as --lagrangian_multiplier_lr). "
        "Previously hardcoded to 0.0 and not reachable from the CLI at all.",
    )
    parser.add_argument(
        "--saute_penalty",
        type=float,
        default=0.0,
        help="Reward substituted once the saute budget is exhausted. ss2r's "
        "own shipped default is 0.0 -- i.e. relying on episode termination "
        "(if enabled) rather than a reward penalty to teach the constraint.",
    )
    parser.add_argument(
        "--saute_terminate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="End the episode once the saute budget is exhausted. Off by "
        "default, matching ss2r's shipped saute.yaml -- with it off, "
        "training only ever sees the reward-penalty shaping, never a hard "
        "stop. Only applies to the TRAINING env; the eval-side Saute wrapper "
        "always uses terminate=False/penalty=0.0 so evaluation reports raw "
        "behaviour rather than budget-exhaustion artifacts, matching ss2r.",
    )
    parser.add_argument(
        "--saute_termination_probability",
        type=float,
        default=1.0,
        help="Only relevant with --saute_terminate: probability that budget "
        "exhaustion actually ends the episode (soft-terminates otherwise). "
        "ss2r's default is 1.0 (always terminate once triggered).",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=None,
        help="Physics steps per episode (NOT decisions -- CostEpisodeWrapper "
        "advances `steps` by action_repeat, so decisions per episode is "
        "episode_length // action_repeat). Defaults per robot: point 1000 "
        "(10 s of simulated time), ant 2500 (25 s) -- see _ROBOT_DEFAULTS.",
    )
    parser.add_argument(
        "--num_morphologies",
        type=int,
        default=0,
        help="Train a single policy across this many distinct, randomly "
        "sampled ant bodies instead of the nominal ant (0 disables morphology "
        "randomization). --num_envs and --num_eval_envs must each be a "
        "multiple of this value -- each sampled body is replicated to fill "
        "the rest of the batch. Only wired up for --robot ant (point has no "
        "morphology parameters). See mjx_safety_gym/morphology.py.",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=None,
        help="Physics steps per policy decision. Fewer decisions per episode "
        "at unchanged physics fidelity -- a throughput win, and it lengthens "
        "the effective horizon seen under --discounting. Defaults per robot: "
        "point 4 (ss2r's reference config), ant 10 (matching "
        "safety-gymnasium's own frameskip_binom_n for the ant; measured to "
        "raise random-policy displacement ~40%% at fixed simulated time). "
        "Note --num_timesteps counts physics steps INCLUDING action_repeat, "
        "so raising this shrinks the gradient-step count at a fixed budget.",
    )
    parser.add_argument(
        "--task",
        choices=["goal", "run", "minefield"],
        default="goal",
        help="'goal' is the original navigate-to-a-respawning-goal task. 'run' "
        "is RunForward: start at one end of a corridor and get as far in +x as "
        "possible without hitting anything. 'run' exists because 'goal' is "
        "provably unlearnable for the ant -- its distance-delta reward is convex, "
        "so undirected motion has negative expected return while freezing scores "
        "0, and the trained ant converged to standing still (0.011 m/episode vs "
        "0.313 m for random actions). 'run' rewards x-displacement, which is "
        "LINEAR, so exploration is free. See envs/run_forward.py. 'minefield' is "
        "'run' with NO VASES and twice the hazards: same reward, same corridor, "
        "same observation width, but every dynamic body is gone (vases are ~82%% "
        "of nq for the ant), which is a large throughput win. Use 'minefield' to "
        "iterate and 'run' once things work -- see envs/minefield.py for what "
        "the weaker cost signal gives up.",
    )
    parser.add_argument(
        "--num_hazards",
        type=int,
        default=None,
        help="[--task minefield] Hazards scattered along the corridor. Defaults "
        "to 20, which keeps the obstacle COUNT equal to 'run' (10 hazards + 10 "
        "vases) so the corridor is not made emptier by dropping the vases, only "
        "cheaper. Hazards are mocap bodies with no DOFs and no contacts, so "
        "raising this is close to free in physics -- it costs one more row in "
        "get_cost's distance matrix. Placement is rejection-sampled with a "
        "bounded retry, so very high counts silently start overlapping.",
    )
    parser.add_argument(
        "--corridor_length",
        type=float,
        default=None,
        help="[--task run] Corridor length in metres. Defaults to 12 x the "
        "robot's arena_scale (12 m for ant, 48 m for ant_gym), because a "
        "corridor only means anything relative to the robot in it. The 10 hazards and 10 "
        "vases are spread across it, so obstacle DENSITY falls as this rises -- "
        "size it to the robot, or it never reaches the first hazard and cost is "
        "identically zero. Measured over a 50 s episode at the 0.08 s control "
        "period: safety-gymnasium's ant covers ~3 m under a scripted gait "
        "(~1.1 m random), the Gym/Brax ant ~21 m (~6 m random).",
    )
    parser.add_argument(
        "--corridor_half_width",
        type=float,
        default=None,
        help="[--task run] Half-width of the corridor. Defaults to 1 x the "
        "robot's arena_scale. Leaving it costs "
        "--boundary_cost_weight per step.",
    )
    parser.add_argument(
        "--forward_reward_weight",
        type=float,
        default=1.0,
        help="[--task run] Scale on per-step +x progress. Episode return is "
        "this times total metres travelled.",
    )
    parser.add_argument(
        "--ctrl_cost_weight",
        type=float,
        default=0.0,
        help="[--task run] Quadratic action penalty. DEFAULTS TO 0 ON PURPOSE: "
        "a control cost is paid by any moving policy and not by a frozen one, so "
        "a positive weight re-creates the freeze attractor this task exists to "
        "escape. Raise only once the robot reliably walks.",
    )
    parser.add_argument(
        "--boundary_cost_weight",
        type=float,
        default=1.0,
        help="[--task run] Cost per step for leaving the corridor. Without it "
        "the safe optimum is to step out of the obstacle band and run in clean "
        "air -- full reward, zero cost -- making the constraint vacuous.",
    )
    parser.add_argument(
        "--healthy_reward",
        type=float,
        default=None,
        help="[--task run] Per-step bonus while the torso is upright (tilted "
        "under ~60 deg). Resolves per robot. MEASURED 2026-08-12: without it "
        "ant_gym is inverted for 94.4%% of steps untrained and 94.1%% after "
        "1.5M steps of training -- training moved it 0.3 points, because "
        "nothing in the reward mentioned staying upright. Note this is the "
        "same class of hazard as --ctrl_cost_weight in reverse: it pays a "
        "FROZEN upright policy, so it must stay well under what walking earns "
        "(a measured good gait travels ~40 m per episode; the ant_gym default "
        "of 0.005/step is worth ~12 over 2500 steps).",
    )
    parser.add_argument(
        "--terminate_on_flip",
        type=lambda v: v.lower() not in ("0", "false", "no"),
        default=None,
        help="[--task run] End the episode when the torso inverts. Resolves "
        "per robot. This is the larger half of the fix: without it an ant that "
        "goes over at step 50 still contributes 2450 further transitions from "
        "a state where forward reward is unobtainable.",
    )
    parser.add_argument(
        "--goal_reward_weight",
        type=float,
        default=None,
        help="[--task run/minefield] Weight on how much CLOSER to the goal the "
        "robot got this step. Telescopes over an episode to (d_initial - "
        "d_final). Unlike --forward_reward_weight this charges for lateral "
        "motion, which is the point: +x progress is flat in y, so nothing ever "
        "preferred going straight over drifting -- measured, half of all "
        "episodes ended BEHIND the start line. Resolves per robot (1.0 for the "
        "ants, 0 for the point). Scale is close to cosmetic since "
        "normalize_advantage is on; what matters is the ratio to "
        "--healthy_reward.",
    )
    parser.add_argument(
        "--goal_observation",
        type=lambda v: v.lower() not in ("0", "false", "no"),
        default=None,
        help="[--task run/minefield] Put the goal's bearing and range straight "
        "into the observation as 3 numbers (cos, sin of relative bearing, and "
        "normalised distance). DELIBERATELY PRIVILEGED STATE -- not a sensor "
        "any real robot has. The goal lidar ring that would have carried this "
        "reads exactly zero for entire episodes (the goal is 11 m away for "
        "ant, 44 m for ant_gym, against LIDAR_MAX_DIST = 2.0). "
        "CHANGES OBSERVATION WIDTH 76 -> 79 for the ants, so checkpoints do "
        "not transfer across this flag.",
    )
    parser.add_argument("--num_timesteps", type=int, default=5_000_000)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=512,
        help="512 is where a 6 GiB laptop GPU saturates. Measured 2026-08-16, "
        "ant on minefield, compile-corrected training/sps: 1155 at 128 envs, "
        "3593 at 256 (3.11x -- small batches are kernel-launch-bound, not "
        "physics-bound), 4747 at 512 (1.32x), 4900 at 1024 (1.03x), 4735 at "
        "2048 (a net LOSS). "
        "512 is also the largest value that needs no other change: validate() "
        "requires num_envs to divide batch_size * num_minibatches (default "
        "32*16 = 512), so 128/256/512 share an identical learning config while "
        "1024+ forces the gradient batch up, halving the updates per env-step "
        "for 3%% more throughput. "
        "Raising this is otherwise free rather than a tradeoff: brax computes "
        "env_step_per_training_step as batch_size * unroll_length * "
        "num_minibatches * action_repeat, with NO num_envs term, so 256 -> 512 "
        "changes neither the data per gradient update nor the number of "
        "updates -- only whether it is gathered as 512 envs x 10 steps once or "
        "256 envs x 10 steps twice. "
        "VRAM is not the constraint at any of these sizes (flat ~3913 MiB from "
        "128 to 2048; that is JAX's preallocated arena, not demand) -- which "
        "only became true once --task minefield dropped the vases. The older "
        "note that 2048 gets OOM-killed host-side is STALE: it predates the "
        "WSL RAM bump to 10 GB and the 79 -> 47 observation narrowing, and "
        "2048 now runs clean. Re-measure before raising it on a cluster.",
    )
    parser.add_argument(
        "--num_eval_envs",
        type=int,
        default=32,
        help="Evaluation vmaps this over --num_eval_episodes, so the parallel "
        "env count during eval is num_eval_envs * num_eval_episodes. Keep "
        "that product within what --num_envs shows is comfortable.",
    )
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_minibatches", type=int, default=16)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument(
        "--integrator",
        choices=["rk4", "implicitfast", "euler"],
        default=None,
        help="Override the integrator the robot's XML declares. Default None "
        "keeps the XML's own choice, so existing runs and checkpoints are "
        "unaffected. This is the single largest throughput lever available: "
        "ant.xml declares RK4 (inherited from safety-gymnasium), which does 4 "
        "force evaluations per step against 1 for implicitfast/euler, and "
        "physics is ~90%% of per-decision cost -- measured 3.5x cheaper "
        "stepping on CPU. NOT free: at the shipped timestep of 0.01 a scripted "
        "gait travels 1.010 m under RK4 vs 0.770 m under implicitfast (~25%% "
        "less), though passive settling is near-identical and neither is "
        "unstable. point.xml already uses euler by default, so overriding the "
        "ant makes the two robots consistent. Measure both speed AND learning "
        "before adopting -- see cluster/ant_throughput_sweep.sbatch.",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=10,
        help="Below the reference's 15: evaluation dominated wall-clock in "
        "short runs (223s of a 297s run). Purely a logging-granularity knob.",
    )
    parser.add_argument(
        "--policy_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[32, 32, 32, 32],
        help="Defaults to ppo/networks.py's own default (32,)*4 -- fine for a "
        "single fixed body, likely too small once the policy is conditioned "
        "on --num_morphologies morphologies at once (the value/cost-value "
        "networks are already (256,)*5, so the policy is the bottleneck). "
        "If per-morphology eval returns come out near-identical under "
        "morphology randomization, widen this first.",
    )
    parser.add_argument(
        "--terminate_on_goal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="End the episode when the robot reaches the goal. OFF by default "
        "so every result before 2026-08-17 stays reproducible. Measured on the "
        "50M morphology run: the ants arrive at decision 248 of 625 on average, "
        "so 60%% of every episode is spent next to a goal that pays nothing "
        "more (the reward telescopes -- once the distance is closed there is "
        "nothing left to earn). Turning this on is ~2.5x more useful "
        "experience per env-step. "
        "CAVEAT: healthy_reward is paid per step, so terminating early means a "
        "FAST arrival collects less of it than a slow one -- a backwards "
        "incentive worth ~0.30 of a ~22.5 return (1.3%%). The discount "
        "(0.97/decision) dominates it by far, but set --healthy_reward 0 if you "
        "want the objective clean.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-4)
    parser.add_argument(
        "--discounting",
        type=float,
        default=None,
        help="Reward discount factor. Defaults per robot: point 0.9 (ss2r's "
        "reference config), ant 0.97 -- a ~3.3 s effective horizon at the "
        "ant's 0.1 s control period, against a measured best gait period of "
        "0.3 s. See _ROBOT_DEFAULTS.",
    )
    parser.add_argument("--safety_discounting", type=float, default=0.9)
    parser.add_argument("--clipping_epsilon", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the mean action instead of sampling. Defaults to False "
        "so that the reported cost and the constrained cost describe the SAME "
        "policy: the cost critic is fit to stochastic on-policy rollouts, so "
        "`safety_budget` only binds on the stochastic policy. Evaluating "
        "deterministically measures something the constraint never targeted -- "
        "measured 2026-08-02, constraint-implied cost matched stochastic eval "
        # %% not %: argparse runs every help string through `help % params`, so
        # a literal percent sign raises TypeError and takes ALL of --help down.
        "to 0.8%% at the feasibility crossing but was off by 36%% against "
        "deterministic eval at the same point, and by ~2.5x early in "
        "training (an untrained policy has mean action ~0, so the "
        "deterministic policy barely moves and looks spuriously safe). Also "
        "matches ppo.train's own default, which the CLI previously overrode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_logdir",
        type=str,
        default=None,
        help="Directory for checkpoints, one subdirectory per eval. Defaults "
        "to <repo>/checkpoints/<robot>, which is where main.py looks. Pass an "
        "explicit path when sweeping hyperparameters, otherwise the runs "
        "overwrite each other's steps.",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable checkpointing entirely (throwaway/debug runs).",
    )
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume from a specific checkpoint step directory.",
    )
    return parser


def validate(args: argparse.Namespace) -> None:
    """Fail fast on config errors that would otherwise surface as bare asserts."""
    if args.batch_size * args.num_minibatches % args.num_envs != 0:
        raise SystemExit(
            f"batch_size * num_minibatches ({args.batch_size} * "
            f"{args.num_minibatches} = {args.batch_size * args.num_minibatches}) "
            f"must be a multiple of num_envs ({args.num_envs}); brax's PPO "
            f"reshapes the rollout into minibatches along the env axis."
        )
    if args.episode_length % args.action_repeat != 0:
        raise SystemExit(
            f"episode_length ({args.episode_length}) must be divisible by "
            f"action_repeat ({args.action_repeat}); evaluation unrolls for "
            f"episode_length // action_repeat steps and would otherwise cut "
            f"episodes short."
        )
    if args.num_morphologies:
        if args.robot not in morphology_lib._MORPH_ROBOTS:
            raise SystemExit(
                f"--num_morphologies is not wired up for --robot {args.robot} "
                f"(point has no morphology parameters). Supported: "
                f"{sorted(morphology_lib._MORPH_ROBOTS)}."
            )
        if args.num_envs % args.num_morphologies != 0:
            raise SystemExit(
                f"--num_envs ({args.num_envs}) must be a multiple of "
                f"--num_morphologies ({args.num_morphologies}); each sampled "
                f"body is replicated to fill the training batch."
            )
        if args.num_eval_envs % args.num_morphologies != 0:
            raise SystemExit(
                f"--num_eval_envs ({args.num_eval_envs}) must be a multiple of "
                f"--num_morphologies ({args.num_morphologies}) -- evaluation's "
                f"own internal vmap needs exactly num_eval_envs bodies "
                f"(num_eval_episodes is a separate, outer vmap)."
            )


def resolve_checkpoint_logdir(args: argparse.Namespace) -> Optional[Path]:
    """Absolute checkpoint directory for this run, or None if disabled.

    .resolve() is NOT cosmetic. orbax raises "Checkpoint path should be
    absolute" from inside the save call, which happens at the FIRST EVAL -- i.e.
    after all the compilation and, with a small --num_evals, potentially after
    hours of training. A 1.5M-step run was lost to exactly this: it trained for
    55 minutes, reported its final eval, and then threw on every checkpoint
    write, leaving nothing on disk. The default from default_checkpoint_dir()
    is already absolute; only a user-supplied relative --checkpoint_logdir could
    trip it, which is the natural thing to type and is what every cluster script
    here does.

    Split out of `train()` so it is reachable without starting a training run --
    a failure mode that only appears after the first eval is exactly the kind
    that needs to be testable in a second. See tests/test_training_config.py.
    """
    if args.no_checkpoint:
        return None
    return (
        Path(args.checkpoint_logdir).resolve()
        if args.checkpoint_logdir
        else default_checkpoint_dir(args.robot)
    )


def train(args: argparse.Namespace):
    resolved_logdir = resolve_checkpoint_logdir(args)
    checkpoint_logdir = None if resolved_logdir is None else str(resolved_logdir)
    if checkpoint_logdir is not None:
        print(f"Checkpoints: {checkpoint_logdir}")

    def build_env():
        common = dict(
            robot=args.robot,
            morphology_conditioning=bool(args.num_morphologies),
            integrator=args.integrator,
        )
        if args.task in ("run", "minefield"):
            corridor = dict(
                corridor_length=args.corridor_length,
                corridor_half_width=args.corridor_half_width,
                forward_reward_weight=args.forward_reward_weight,
                ctrl_cost_weight=args.ctrl_cost_weight,
                boundary_cost_weight=args.boundary_cost_weight,
                healthy_reward=args.healthy_reward,
                terminate_on_flip=args.terminate_on_flip,
                terminate_on_goal=args.terminate_on_goal,
                goal_reward_weight=args.goal_reward_weight,
                goal_observation=args.goal_observation,
                **common,
            )
            if args.task == "minefield":
                # None lets Minefield apply its own default rather than this
                # module deciding the count in two places.
                return Minefield(num_hazards=args.num_hazards, **corridor)
            return RunForward(**corridor)
        return GoToGoal(**common)

    env = build_env()
    eval_env = build_env()

    # Composition order matters and is NOT arbitrary: any obs-shape-changing
    # wrapper (Saute, morphology randomization) must be applied to the raw
    # env, before wrap_for_brax_training -- never around its output. See
    # MorphologyDomainRandomizationWrapper's docstring for the crash this
    # avoids (CostEpisodeWrapper carries `state` through its own internal
    # action_repeat scan; obs width has to already be final before it ever
    # sees the env, or that scan's carry/output types mismatch on step one).
    if args.penalizer == "saute":
        # Saute is not a Penalizer (no cost critic, no CRPO/Lagrangian switch)
        # -- it's an env wrapper that shapes reward directly, so it trains
        # through the exact same safe=False/penalizer=None path as
        # --penalizer none. Eval side always uses penalty=0.0/terminate=False
        # regardless of the CLI flags, matching ss2r's saute_eval: evaluation
        # should report raw behaviour, not budget-exhaustion artifacts.
        env = Saute(
            env,
            args.safety_discounting,
            args.safety_budget,
            args.saute_penalty,
            args.saute_terminate,
            args.saute_termination_probability,
        )
        eval_env = Saute(
            eval_env, args.safety_discounting, args.safety_budget, 0.0, False
        )

    if args.num_morphologies:
        # Eval reuses the SAME sampled population as training (so evaluation
        # measures the bodies actually trained on), just replicated to a
        # different width -- see the num_eval_envs check in validate().
        rng = jax.random.PRNGKey(args.seed)
        train_rng, eval_rng = jax.random.split(rng)
        # model_builder is what makes this task-correct. Without it
        # randomization_fn falls back to morphology.build_mj_model, which
        # hardcodes GoToGoal's arena -- silently wrong physics on --task
        # run/minefield. See GoToGoal.build_morphology_model.
        train_batched, train_in_axes, train_genes = morphology_lib.randomization_fn(
            env.mjx_model,
            train_rng,
            args.num_morphologies,
            args.num_envs,
            integrator=args.integrator,
            model_builder=env.build_morphology_model,
        )
        env = MorphologyDomainRandomizationWrapper(
            env, train_batched, train_in_axes, train_genes
        )
        eval_batched, eval_in_axes, eval_genes = morphology_lib.randomization_fn(
            eval_env.mjx_model,
            eval_rng,
            args.num_morphologies,
            args.num_eval_envs,
            integrator=args.integrator,
            model_builder=eval_env.build_morphology_model,
        )
        eval_env = MorphologyDomainRandomizationWrapper(
            eval_env, eval_batched, eval_in_axes, eval_genes
        )

    train_env = wrap_for_brax_training(
        env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        already_batched=bool(args.num_morphologies),
    )
    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        already_batched=bool(args.num_morphologies),
    )

    penalizer_name = None if args.penalizer in ("none", "saute") else args.penalizer
    penalizer, penalizer_params = get_penalizer(
        penalizer_name,
        eta=args.crpo_eta,
        burnin=args.crpo_burnin,
        multiplier_lr=args.lagrangian_multiplier_lr,
        initial_lagrange_multiplier=args.initial_lagrange_multiplier,
    )

    def progress_fn(step, metrics):
        print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(args.policy_hidden_layer_sizes),
    )

    make_policy, params, metrics = ppo_train.train(
        environment=train_env,
        eval_env=eval_env,
        network_factory=network_factory,
        num_timesteps=args.num_timesteps,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        num_eval_episodes=args.num_eval_episodes,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        unroll_length=args.unroll_length,
        num_evals=args.num_evals,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        safety_discounting=args.safety_discounting,
        clipping_epsilon=args.clipping_epsilon,
        max_grad_norm=args.max_grad_norm,
        deterministic_eval=args.deterministic_eval,
        seed=args.seed,
        safety_budget=args.safety_budget,
        penalizer=penalizer,
        penalizer_params=penalizer_params,
        safe=penalizer is not None,
        progress_fn=progress_fn,
        checkpoint_logdir=checkpoint_logdir,
        restore_checkpoint_path=args.restore_checkpoint_path,
    )
    return make_policy, params, metrics


if __name__ == "__main__":
    args = build_argparser().parse_args()
    apply_robot_defaults(args)
    validate(args)
    print(f"JAX compilation cache: {jax_cache.configure()}")
    train(args)

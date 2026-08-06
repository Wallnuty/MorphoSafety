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
from mjx_safety_gym.envs.go_to_goal import GoToGoal


# Checkpoints live outside the package so they are not swept up by a package
# install or by `setuptools.packages.find`. Both this module and main.py resolve
# the same convention, so a trained policy is picked up for replay without
# anyone having to pass a path around.
CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"


def default_checkpoint_dir(robot: str) -> Path:
    """Where runs for `robot` write checkpoints unless told otherwise."""
    return CHECKPOINT_ROOT / robot


def latest_checkpoint(robot: str) -> Optional[Path]:
    """Newest checkpoint for `robot`, or None if nothing has been trained.

    `brax.training.checkpoint.save` writes one zero-padded step directory per
    eval, so the lexicographic max is also the newest.
    """
    root = default_checkpoint_dir(robot)
    if not root.is_dir():
        return None
    steps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else None


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
    parser.add_argument("--robot", choices=["point", "ant"], default="point")
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
    parser.add_argument("--episode_length", type=int, default=1000)
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
        default=4,
        help="Physics steps per policy decision. The reference config uses 4, "
        "which quarters the number of decisions per episode at unchanged "
        "physics fidelity -- a large throughput win, and it lengthens the "
        "effective horizon seen under discounting=0.9.",
    )
    parser.add_argument("--num_timesteps", type=int, default=5_000_000)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=256,
        help="Deviates from the reference's 2048: on a 6 GiB laptop GPU, "
        "throughput saturates near 256 (2.1x over 64 envs, vs 2.8x at 1024 "
        "for 5.5x the compile time), and 2048 gets OOM-killed host-side "
        "during XLA compilation. Raise it on a cluster.",
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-4)
    parser.add_argument("--discounting", type=float, default=0.9)
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
        "to 0.8% at the feasibility crossing but was off by 36% against "
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
        if args.robot != "ant":
            raise SystemExit(
                "--num_morphologies is only wired up for --robot ant "
                "(point has no morphology parameters)."
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


def train(args: argparse.Namespace):
    if args.no_checkpoint:
        checkpoint_logdir = None
    else:
        checkpoint_logdir = str(
            Path(args.checkpoint_logdir)
            if args.checkpoint_logdir
            else default_checkpoint_dir(args.robot)
        )
        print(f"Checkpoints: {checkpoint_logdir}")

    env = GoToGoal(
        robot=args.robot, morphology_conditioning=bool(args.num_morphologies)
    )
    eval_env = GoToGoal(
        robot=args.robot, morphology_conditioning=bool(args.num_morphologies)
    )

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
        train_batched, train_in_axes, train_genes = morphology_lib.randomization_fn(
            env.mjx_model, train_rng, args.num_morphologies, args.num_envs
        )
        env = MorphologyDomainRandomizationWrapper(
            env, train_batched, train_in_axes, train_genes
        )
        eval_batched, eval_in_axes, eval_genes = morphology_lib.randomization_fn(
            eval_env.mjx_model, eval_rng, args.num_morphologies, args.num_eval_envs
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
    validate(args)
    print(f"JAX compilation cache: {jax_cache.configure()}")
    train(args)

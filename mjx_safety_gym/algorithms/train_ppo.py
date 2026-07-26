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

import jax
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper

from mjx_safety_gym import jax_cache
from mjx_safety_gym.algorithms.penalizers import get_penalizer
from mjx_safety_gym.algorithms.ppo import train as ppo_train
from mjx_safety_gym.algorithms.wrappers import CostEpisodeWrapper
from mjx_safety_gym.envs.go_to_goal import GoToGoal


def wrap_for_brax_training(env, episode_length: int, action_repeat: int = 1):
    """Vmap + cost-aware episode wrapper + mujoco_playground's auto-reset.

    Mirrors ss2r's wrap_for_brax_training, but built from `CostEpisodeWrapper`
    (which threads info["cost"] through episode aggregation) instead of
    mujoco_playground's own wrap_for_brax_training (which uses brax's vanilla
    EpisodeWrapper and drops the "cost" key added later by the eval wrapper).
    """
    env = brax_training.VmapWrapper(env)
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    env = playground_wrapper.BraxAutoResetWrapper(env)
    return env


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["point", "ant"], default="point")
    parser.add_argument(
        "--penalizer", choices=["crpo", "ppo_lagrangian", "none"], default="crpo"
    )
    parser.add_argument("--safety_budget", type=float, default=25.0)
    parser.add_argument("--crpo_eta", type=float, default=0.0)
    parser.add_argument("--crpo_burnin", type=int, default=0)
    parser.add_argument("--lagrangian_multiplier_lr", type=float, default=1e-2)
    parser.add_argument("--episode_length", type=int, default=1000)
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-4)
    parser.add_argument("--discounting", type=float, default=0.9)
    parser.add_argument("--safety_discounting", type=float, default=0.9)
    parser.add_argument("--clipping_epsilon", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
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


def train(args: argparse.Namespace):
    env = GoToGoal(robot=args.robot)
    eval_env = GoToGoal(robot=args.robot)
    train_env = wrap_for_brax_training(
        env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )
    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )

    penalizer_name = None if args.penalizer == "none" else args.penalizer
    penalizer, penalizer_params = get_penalizer(
        penalizer_name,
        eta=args.crpo_eta,
        burnin=args.crpo_burnin,
        multiplier_lr=args.lagrangian_multiplier_lr,
    )

    def progress_fn(step, metrics):
        print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    make_policy, params, metrics = ppo_train.train(
        environment=train_env,
        eval_env=eval_env,
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
    )
    return make_policy, params, metrics


if __name__ == "__main__":
    args = build_argparser().parse_args()
    validate(args)
    print(f"JAX compilation cache: {jax_cache.configure()}")
    train(args)

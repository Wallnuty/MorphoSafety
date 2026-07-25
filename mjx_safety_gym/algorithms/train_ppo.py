"""Train PPO + CRPO/Lagrangian on GoToGoal.

Minimal, Hydra-free training entry point, styled like the top-level
main.py demo script. Ported/assembled from safe-learning (ss2r)'s
ss2r/algorithms/ppo and ss2r/benchmark_suites/mujoco_playground.

Usage:
    python -m mjx_safety_gym.algorithms.train_ppo --robot point \
        --penalizer crpo --num_timesteps 200_000
"""

import argparse

import jax
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper

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
    parser.add_argument("--num_timesteps", type=int, default=200_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_eval_envs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument("--num_evals", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def train(args: argparse.Namespace):
    env = GoToGoal(robot=args.robot)
    eval_env = GoToGoal(robot=args.robot)
    train_env = wrap_for_brax_training(env, episode_length=args.episode_length)
    eval_env = wrap_for_brax_training(eval_env, episode_length=args.episode_length)

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
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        unroll_length=args.unroll_length,
        num_evals=args.num_evals,
        learning_rate=args.learning_rate,
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
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    train(args)

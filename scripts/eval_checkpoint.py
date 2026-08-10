"""Evaluate a trained checkpoint over many episodes, against an untrained control.

WHY THIS EXISTS. The in-training eval on this laptop is capped at
`--num_eval_envs 8 --num_eval_episodes 2` = 16 episodes, because the wider
default (32x10 = 320) reliably segfaults here. On RunForward the per-episode
spread is large (reward std ~3.7 against means under 1), so 16 episodes gives a
standard error of ~+-0.93 -- wide enough that a real improvement of a metre or
two is indistinguishable from noise. Three consecutive in-training evals
measured +0.56, -0.32 and +0.89, which says nothing.

Separating measurement from training fixes that. Train with `--num_evals 2`
(the configuration that has actually completed on this machine; every crash so
far has struck at the third eval) and checkpoint, then run this afterwards with
as many episodes as you like -- there is no gradient step to interleave, so the
eval width that crashes *training* is not the constraint here.

The untrained control matters as much as the trained number. "Distance 1.2 m"
means nothing on its own; what matters is whether it beats a freshly
initialised policy evaluated under exactly the same arena seeds, and by more
than the standard error of the difference. Both are run on the SAME seeds
(common random numbers), which removes arena-layout variance from the
comparison rather than averaging over it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics

from mjx_safety_gym import jax_cache
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.envs.run_forward import RunForward
from mjx_safety_gym.envs.go_to_goal import GoToGoal


def build_policy(ckpt_dir: Path | None, env, deterministic: bool, seed: int):
    """Return an action fn. `ckpt_dir=None` gives a freshly initialised policy.

    Mirrors main.py's loader: the network is rebuilt from the env's own shapes
    because training writes an empty ConfigDict, and orbax is driven directly
    rather than through brax's checkpoint.load (which chokes on the optimizer
    subtree this repo also saves).
    """
    obs_shape = (env.observation_size,)
    network = ppo_networks.make_ppo_networks(
        obs_shape, env.action_size, preprocess_observations_fn=lambda x, _: x
    )
    if ckpt_dir is None:
        key = jax.random.PRNGKey(seed)
        params = network.policy_network.init(key)
        value = network.value_network.init(jax.random.PRNGKey(seed + 1))
        normalizer = running_statistics.init_state(jp.zeros(obs_shape))
        return ppo_networks.make_inference_fn(network)(
            (normalizer, params, value), deterministic=deterministic
        )
    steps = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
    leaf = steps[-1] if steps else ckpt_dir
    loaded = ocp.PyTreeCheckpointer().restore(str(leaf.resolve()))
    normalizer = running_statistics.RunningStatisticsState(**loaded[0])
    policy_params, value_params = loaded[1]["policy"], loaded[1]["value"]
    print(f"  loaded {leaf}")
    return ppo_networks.make_inference_fn(network)(
        (normalizer, policy_params, value_params), deterministic=deterministic
    )


def rollout_many(env, policy_fn, seeds, n_steps):
    """Roll out one episode per seed, vmapped. Returns per-episode arrays."""

    def one(seed):
        state = env.reset(jax.random.PRNGKey(seed))

        def body(carry, _):
            st, key = carry
            key, ak = jax.random.split(key)
            action, _ = policy_fn(st.obs, ak)
            nst = env.step(st, action)
            return (nst, key), (nst.reward, nst.info["cost"], nst.info["out_of_bounds"])

        (st, _), (r, c, oob) = jax.lax.scan(
            body, (state, jax.random.PRNGKey(seed + 10_000)), (), n_steps
        )
        # Summed reward IS total +x displacement in metres, exactly -- verified
        # to 0.00e+00. RunForward deliberately carries no "distance" metric,
        # because anything in state.info survives the episode boundary that
        # BraxAutoResetWrapper does not clear.
        return r.sum(), c.sum(), oob.sum(), r.sum()

    return jax.jit(jax.vmap(one))(jp.asarray(seeds))


def summarise(name, out):
    r, c, oob, dist = (np.asarray(x) for x in out)
    n = len(r)
    return {
        "name": name,
        "distance": dist.mean(),
        "distance_se": dist.std(ddof=1) / np.sqrt(n),
        "cost": c.mean(),
        "cost_se": c.std(ddof=1) / np.sqrt(n),
        "boundary": oob.mean(),
        "hazard": c.mean() - oob.mean(),
        "raw": dist,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="ant_gym")
    ap.add_argument("--task", choices=["run", "goal"], default="run")
    ap.add_argument("--checkpoint", default="checkpoints/ant_gym_run")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument(
        "--steps",
        type=int,
        default=625,
        help="env.step calls per episode. Training uses episode_length // "
        "action_repeat = 2500 // 4 = 625 for the ant robots.",
    )
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    jax_cache.configure()
    env = (
        RunForward(robot=args.robot)
        if args.task == "run"
        else GoToGoal(robot=args.robot)
    )
    seeds = np.arange(args.episodes) + 1000  # common random numbers across arms

    print(f"{args.robot} / {args.task}, {args.episodes} episodes x {args.steps} steps")
    print("untrained control:")
    untrained = summarise(
        "untrained",
        rollout_many(env, build_policy(None, env, args.deterministic, 0), seeds, args.steps),
    )
    ckpt = Path(args.checkpoint)
    trained = None
    if ckpt.exists():
        print("trained:")
        trained = summarise(
            "trained",
            rollout_many(
                env, build_policy(ckpt, env, args.deterministic, 0), seeds, args.steps
            ),
        )
    else:
        print(f"  (no checkpoint at {ckpt}, control only)")

    print()
    print(f"{'policy':<11}{'distance m':>13}{'cost':>12}{'boundary':>11}{'hazard':>9}")
    print("-" * 58)
    for row in filter(None, (untrained, trained)):
        print(
            f"{row['name']:<11}{row['distance']:>8.3f} +-{row['distance_se']:<4.2f}"
            f"{row['cost']:>7.1f} +-{row['cost_se']:<4.1f}"
            f"{row['boundary']:>11.1f}{row['hazard']:>9.1f}"
        )

    if trained is not None:
        # paired difference: same arena seed in both arms, so the per-episode
        # difference cancels layout variance instead of averaging over it
        d = trained["raw"] - untrained["raw"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print()
        print(f"paired improvement: {d.mean():+.3f} +- {se:.3f} m  "
              f"({d.mean()/se if se else float('nan'):+.1f} SE)")
        if d.mean() > 2 * se:
            print("  => LEARNED: trained beats untrained by more than 2 SE")
        elif d.mean() < -2 * se:
            print("  => WORSE than untrained by more than 2 SE")
        else:
            print("  => INCONCLUSIVE: within 2 SE of no change. Either it has not"
                  " learned yet, or more episodes are needed to see it.")


if __name__ == "__main__":
    main()

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

WHAT THE SEEDS DO AND DO NOT PIN. They pin the arena layout and the action
noise. They do NOT pin the trajectory. Measured 2026-08-11: running this script
twice on identical seeds and an identical checkpoint, changing nothing but
adding two extra outputs to the scan, moved the untrained control's mean
episode cost from 8.8 to 26.6 and its boundary cost from 5.2 to 18.9. The extra
outputs re-fuse the XLA graph, which changes float association, which chaotic
contact dynamics amplify to full decorrelation over 625 steps. So:

  * The +-SE printed below is the spread ACROSS EPISODES within one run. It is
    not a run-to-run reproducibility bound, and it understates how much a
    number will move if anything about the compiled graph changes.
  * At 128 episodes, `cost`/`boundary` do not resolve. In that same pair of
    runs the trained-vs-untrained cost comparison came out "7x worse" once and
    "identical" the other time. Do not quote cost from this harness without
    re-running it and seeing the effect twice.
  * `net +x` and `path` DID replicate in sign and rough magnitude. Prefer them.
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

    site = env._robot_site_id

    def one(seed):
        state = env.reset(jax.random.PRNGKey(seed))

        def body(carry, _):
            st, key = carry
            key, ak = jax.random.split(key)
            action, _ = policy_fn(st.obs, ak)
            nst = env.step(st, action)
            # per-step displacement, so a policy that thrashes in place is
            # distinguishable from one that stands still -- net displacement
            # alone cannot tell them apart, and that ambiguity is exactly what
            # made the first run of this script unreadable.
            d = nst.data.site_xpos[site][:2] - st.data.site_xpos[site][:2]
            return (nst, key), (
                nst.reward,
                nst.info["cost"],
                nst.info["out_of_bounds"],
                jp.linalg.norm(d),
                d[1],
            )

        (st, _), (r, c, oob, seglen, dy) = jax.lax.scan(
            body, (state, jax.random.PRNGKey(seed + 10_000)), (), n_steps
        )
        # Summed reward IS total +x displacement in metres, exactly -- verified
        # to 0.00e+00. RunForward deliberately carries no "distance" metric,
        # because anything in state.info survives the episode boundary that
        # BraxAutoResetWrapper does not clear.
        return r.sum(), c.sum(), oob.sum(), r.sum(), seglen.sum(), dy.sum()

    return jax.jit(jax.vmap(one))(jp.asarray(seeds))


def summarise(name, out):
    r, c, oob, dist, path, dy = (np.asarray(x) for x in out)
    n = len(r)
    return {
        "name": name,
        "distance": dist.mean(),
        "distance_se": dist.std(ddof=1) / np.sqrt(n),
        "cost": c.mean(),
        "cost_se": c.std(ddof=1) / np.sqrt(n),
        "boundary": oob.mean(),
        "hazard": c.mean() - oob.mean(),
        "path": path.mean(),
        "lateral": np.abs(dy).mean(),
        "raw": dist,
        "raw_cost": c,
        "raw_path": path,
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
    print(f"{'policy':<11}{'net +x m':>13}{'path m':>9}{'|dy| m':>8}"
          f"{'cost':>12}{'boundary':>10}{'hazard':>8}")
    print("-" * 71)
    for row in filter(None, (untrained, trained)):
        print(
            f"{row['name']:<11}{row['distance']:>8.3f} +-{row['distance_se']:<4.2f}"
            f"{row['path']:>9.2f}{row['lateral']:>8.2f}"
            f"{row['cost']:>7.1f} +-{row['cost_se']:<4.1f}"
            f"{row['boundary']:>10.1f}{row['hazard']:>8.1f}"
        )
    print()
    print("net +x is the episode return. `path` is total distance travelled by "
          "the torso\nsite (thrashing in place shows up here and nowhere else); "
          "`|dy|` is net lateral\ntravel, which is what running out of the "
          "corridor looks like.")

    if trained is not None:
        # paired differences: same arena seed in both arms, so the per-episode
        # difference cancels layout variance instead of averaging over it
        print()
        for label, key, unit in (
            ("net +x  ", "raw", "m"),
            ("path    ", "raw_path", "m"),
            ("cost    ", "raw_cost", ""),
        ):
            d = trained[key] - untrained[key]
            se = d.std(ddof=1) / np.sqrt(len(d))
            n_se = d.mean() / se if se else float("nan")
            verdict = "same" if abs(n_se) < 2 else ("HIGHER" if n_se > 0 else "LOWER")
            print(f"paired {label} {d.mean():+9.3f} +- {se:6.3f} {unit:<2}"
                  f"({n_se:+5.1f} SE)  {verdict}")

        d = trained["raw"] - untrained["raw"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print()
        if d.mean() > 2 * se:
            print("  => LEARNED: trained travels further in +x than untrained by"
                  " more than 2 SE")
        elif d.mean() < -2 * se:
            print("  => WORSE in +x than untrained by more than 2 SE")
        else:
            print("  => INCONCLUSIVE on +x: within 2 SE of no change. Read the"
                  " path and cost\n     rows before concluding nothing was"
                  " learned -- a policy can learn to move\n     a lot without"
                  " moving in the rewarded direction.")


if __name__ == "__main__":
    main()

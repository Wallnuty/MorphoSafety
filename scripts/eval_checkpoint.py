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
contact dynamics amplify to full decorrelation over a full episode. So:

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
from mjx_safety_gym.algorithms import train_ppo
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.envs.run_forward import RunForward
from mjx_safety_gym.envs.minefield import Minefield
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


def rollout_many(env, policy_fn, seeds, n_decisions, action_repeat):
    """Roll out one episode per seed, vmapped. Returns per-episode arrays.

    ACTION REPEAT IS NOT OPTIONAL HERE. Training does not step the raw env once
    per decision -- CostEpisodeWrapper holds each action for `action_repeat`
    inner `env.step` calls and sums reward and cost over them, and the episode
    ends at `steps >= episode_length` where `steps` grows by `action_repeat`.
    So a training episode is `episode_length` inner steps (2500 for the ants),
    reached in `episode_length // action_repeat` decisions (625).

    An earlier version of this function ran 625 INNER steps with a fresh action
    each, which is a quarter of the episode at four times the control rate --
    and control period alone is worth 2.7x in achievable gait travel on this
    robot. It reported episode_cost ~26 where the in-training eval on the same
    policy reported 353.
    """

    site = env._robot_site_id

    def one(seed):
        state = env.reset(jax.random.PRNGKey(seed))

        def decision(carry, _):
            st, key, alive = carry
            key, ak = jax.random.split(key)
            action, _ = policy_fn(st.obs, ak)

            def inner(s, _):
                ns = env.step(s, action)
                # per-step displacement, so a policy that thrashes in place is
                # distinguishable from one that stands still -- net
                # displacement alone cannot tell them apart, and that ambiguity
                # is exactly what made the first run of this script unreadable.
                d = ns.data.site_xpos[site][:2] - s.data.site_xpos[site][:2]
                # .get, because the two tasks carry different info keys:
                # out_of_bounds is RunForward-only and goal_reached is
                # GoToGoal-only. Indexing either directly makes --task goal
                # raise KeyError, which is what it used to do.
                zero = jp.zeros(())
                return ns, (ns.reward, ns.info["cost"],
                            ns.info.get("out_of_bounds", zero),
                            ns.info.get("goal_reached", zero),
                            jp.linalg.norm(d), d[1], ns.done)

            nst, inner_out = jax.lax.scan(inner, st, (), action_repeat)
            *metrics, done = inner_out
            # summed over the repeat, matching CostEpisodeWrapper
            summed = tuple(x.sum(axis=0) for x in metrics)

            # HONOUR TERMINATION. There is no auto-reset wrapper here, so
            # env.step keeps integrating after `done` -- and with
            # terminate_on_flip the ant flips at ~100 of 2500 steps, then
            # spends the remaining 96% of the scan squirming on its back,
            # accumulating reward and cost that training would never have
            # collected. Everything after the first `done` is masked out
            # instead, and `steps` records where the episode actually ended so
            # it can be compared against eval/avg_episode_length.
            still = alive * (1.0 - jp.clip(done.sum(), 0.0, 1.0))
            masked = tuple(x * alive for x in summed)
            return (nst, key, still), masked + (alive * action_repeat,)

        (st, _, _), (r, c, oob, goals, seglen, dy, steps) = jax.lax.scan(
            decision,
            (state, jax.random.PRNGKey(seed + 10_000), jp.ones(())),
            (), n_decisions,
        )
        # On RunForward, summed reward IS total +x displacement in metres,
        # exactly -- verified to 0.00e+00. RunForward deliberately carries no
        # "distance" metric, because anything in state.info survives the
        # episode boundary that BraxAutoResetWrapper does not clear. On
        # GoToGoal the return is shaped progress plus a +1 per goal, so it is
        # NOT a distance -- read `goals` there instead.
        return (r.sum(), c.sum(), oob.sum(), goals.sum(), r.sum(),
                seglen.sum(), dy.sum(), steps.sum())

    return jax.jit(jax.vmap(one))(jp.asarray(seeds))


def summarise(name, out):
    r, c, oob, goals, dist, path, dy, steps = (np.asarray(x) for x in out)
    n = len(r)
    return {
        "name": name,
        "goals": goals.mean(),
        "steps": steps.mean(),
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
    ap.add_argument("--task", choices=["run", "minefield", "goal"], default="run")
    ap.add_argument("--checkpoint", default="checkpoints/ant_gym_run")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument(
        "--episode_length",
        type=int,
        default=None,
        help="inner env.step calls per episode. Defaults to the robot's own "
        "training value from train_ppo._ROBOT_DEFAULTS, so this cannot drift "
        "away from what training actually ran.",
    )
    ap.add_argument("--action_repeat", type=int, default=None,
                    help="same default source as --episode_length")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    jax_cache.configure()
    # Corridor tasks are built with the same per-robot settings training uses,
    # AND reconciled against the checkpoint's own observation width:
    # goal_observation moves that width by 3, so a pre-2026-08-15 ant
    # checkpoint wants 76 where today's defaults give 79. Without this the
    # script dies inside flax with a shape error that names neither the flag
    # nor the checkpoint.
    _tasks = {"run": RunForward, "minefield": Minefield, "goal": GoToGoal}
    if args.task == "goal":
        env = GoToGoal(robot=args.robot)
    else:
        ckpt_root = Path(args.checkpoint) if args.checkpoint else None
        want = None
        if ckpt_root is not None and ckpt_root.is_dir():
            steps = sorted(
                p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()
            )
            leaf = steps[-1] if steps else ckpt_root
            probe = ocp.PyTreeCheckpointer().restore(str(leaf.resolve()))
            want = train_ppo.checkpoint_obs_width(probe[1]["policy"])
        env, _kw, default_width = train_ppo.build_env_for_checkpoint(
            lambda **kw: _tasks[args.task](robot=args.robot, **kw), args.robot, want
        )
        if default_width is not None:
            print(
                f"  checkpoint wants obs width {want} (defaults give "
                f"{default_width}); built env with "
                f"goal_observation={_kw['goal_observation']}, "
                f"goal_reward_weight={_kw['goal_reward_weight']}"
            )
    seeds = np.arange(args.episodes) + 1000  # common random numbers across arms

    defaults = train_ppo._ROBOT_DEFAULTS[args.robot]
    episode_length = args.episode_length or defaults["episode_length"]
    action_repeat = args.action_repeat or defaults["action_repeat"]
    n_decisions = episode_length // action_repeat

    print(
        f"{args.robot} / {args.task}, {args.episodes} episodes x {episode_length}"
        f" env steps ({n_decisions} decisions x action_repeat {action_repeat})"
    )
    print("untrained control:")
    untrained = summarise(
        "untrained",
        rollout_many(
            env, build_policy(None, env, args.deterministic, 0), seeds,
            n_decisions, action_repeat,
        ),
    )
    ckpt = Path(args.checkpoint)
    trained = None
    if ckpt.exists():
        print("trained:")
        trained = summarise(
            "trained",
            rollout_many(
                env, build_policy(ckpt, env, args.deterministic, 0), seeds,
                n_decisions, action_repeat,
            ),
        )
    else:
        print(f"  (no checkpoint at {ckpt}, control only)")

    goal_task = args.task == "goal"
    # On RunForward the return IS net +x displacement in metres. On GoToGoal it
    # is shaped progress plus +1 per goal, so calling it a distance would be a
    # lie -- goals/ep is the readable number there (the point's unconstrained
    # ceiling was 3.94).
    ret_label = "return" if goal_task else "net +x m"
    header = f"{'policy':<11}{ret_label:>13}"
    if goal_task:
        header += f"{'goals/ep':>10}"
    header += (f"{'steps':>8}{'path m':>9}{'|dy| m':>8}{'cost':>12}"
               f"{'boundary':>10}{'hazard':>8}")
    print()
    print(header)
    print("-" * len(header))
    for row in filter(None, (untrained, trained)):
        line = f"{row['name']:<11}{row['distance']:>8.3f} +-{row['distance_se']:<4.2f}"
        if goal_task:
            line += f"{row['goals']:>10.2f}"
        line += (
            f"{row['steps']:>8.0f}{row['path']:>9.2f}{row['lateral']:>8.2f}"
            f"{row['cost']:>7.1f} +-{row['cost_se']:<4.1f}"
            f"{row['boundary']:>10.1f}{row['hazard']:>8.1f}"
        )
        print(line)
    print()
    print("`path` is total distance travelled by the torso site (thrashing in "
          "place shows\nup here and nowhere else); `|dy|` is net lateral travel."
          + ("" if goal_task else
             " On the run task, net +x\nIS the episode return, and `boundary` is"
             " the part of cost from leaving the\ncorridor."))

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

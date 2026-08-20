#!/usr/bin/env python
"""Does one gene-conditioned policy actually make many bodies walk?

Three measurements the training log CANNOT give you, in increasing order of
how badly you need them:

  1. PER-MORPHOLOGY RETURN. `episode_reward` in the log is averaged over every
     body in the population, so a mean of 11 is equally consistent with "all
     eight bodies reach 11" and "four reach 22 and four never move". Those are
     opposite conclusions about the hypothesis.

  2. GENE ABLATION -- the one that matters. Re-run every body with ANOTHER
     body's gene vector. If return barely drops, the policy learned a single
     robust gait and is ignoring the conditioning entirely. That is a failure
     of the method that LOOKS EXACTLY LIKE SUCCESS in every other metric, and
     nothing else in this repo detects it. A conditioned policy worth the name
     must do measurably worse on wrong genes.

  3. VS SPECIALIST. Lane 0 is the nominal ant by construction
     (`include_nominal`), and a single-body specialist reached episode_reward
     22.1 of a ~22.5 ceiling on this exact task
     (checkpoints/ant_minefield_chain, 2026-08-16). Conditioned performance on
     lane 0 is directly comparable to that number.

Usage:
    python scripts/eval_morphology.py --checkpoint checkpoints/ant_morph_chain/gen10
    python scripts/eval_morphology.py --checkpoint <dir> --episodes 32

Runs on whatever backend JAX picks; it is a rollout, not training, so it is
cheap -- but it still wants the GPU to itself (one JAX process at a time).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brax.training.acme import running_statistics  # noqa: E402

from mjx_safety_gym import morphology as morphology_lib  # noqa: E402
from mjx_safety_gym.algorithms import train_ppo  # noqa: E402
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks  # noqa: E402
from mjx_safety_gym.algorithms.wrappers import (  # noqa: E402
    MorphologyDomainRandomizationWrapper,
)
from mjx_safety_gym.envs.minefield import Minefield  # noqa: E402
from mjx_safety_gym.envs.run_forward import RunForward  # noqa: E402

_TASKS = {"minefield": Minefield, "run": RunForward}


def _leaf(ckpt_dir: Path) -> Path:
    """Newest numeric step directory under a checkpoint dir."""
    steps = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else ckpt_dir


def policy_layer_sizes(policy_params) -> tuple[int, ...]:
    """Recover hidden layer widths from the saved kernels.

    Training writes an EMPTY ConfigDict, so the checkpoint records nothing
    about the network that produced it -- the parameter shapes are the only
    surviving evidence, exactly as for observation width
    (train_ppo.checkpoint_obs_width). Without this, a checkpoint trained with
    --policy_hidden_layer_sizes 256 256 256 256 cannot be loaded at all:
    make_ppo_networks would rebuild the (32,)*4 default and flax would raise a
    shape error. scripts/eval_checkpoint.py still has that limitation.
    """
    params = policy_params["params"]
    sizes = []
    i = 0
    while f"hidden_{i}" in params:
        sizes.append(int(params[f"hidden_{i}"]["kernel"].shape[1]))
        i += 1
    # The last Dense is the output head (2 * action_size), not a hidden layer.
    return tuple(sizes[:-1])


def load_policy(ckpt_dir: Path, obs_size: int, action_size: int, deterministic: bool):
    loaded = ocp.PyTreeCheckpointer().restore(str(_leaf(ckpt_dir).resolve()))
    normalizer = running_statistics.RunningStatisticsState(**loaded[0])
    policy_params, value_params = loaded[1]["policy"], loaded[1]["value"]
    sizes = policy_layer_sizes(policy_params)
    network = ppo_networks.make_ppo_networks(
        (obs_size,),
        action_size,
        preprocess_observations_fn=lambda x, _: x,
        policy_hidden_layer_sizes=sizes,
    )
    print(f"  checkpoint {_leaf(ckpt_dir)}")
    print(f"  policy hidden layers {sizes}, obs {obs_size}")
    return ppo_networks.make_inference_fn(network)(
        (normalizer, policy_params, value_params), deterministic=deterministic
    )


def rollout(env, policy_fn, n_envs, n_decisions, seed):
    """One episode per lane, all lanes in parallel. Returns (reward, cost, len).

    Steps the WRAPPED env (MorphologyDomainRandomizationWrapper is already a
    vmap over lanes), so `rng` is per-lane and the whole thing runs as one
    batched rollout -- the same shape training uses.

    action_repeat is handled by CostEpisodeWrapper inside `env`, so this loop
    counts DECISIONS, not inner physics steps. Stepping the raw env once per
    action instead would run a quarter of an episode at four times the control
    rate -- the bug that invalidated an earlier evaluation (2026-08-11).
    """
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    state = env.reset(jax.random.split(k, n_envs))

    inner = env.unwrapped

    def body(carry, _):
        state, rng, ret, cost, alive, t, arr = carry
        rng, ak = jax.random.split(rng)
        action = policy_fn(state.obs, ak)[0]
        nxt = env.step(state, action)
        # Stop accumulating once an episode has terminated, so a body that
        # flips at decision 40 is not credited with the autoreset episode that
        # follows it in the same rollout.
        ret = ret + nxt.reward * alive
        cost = cost + nxt.info.get("cost", jp.zeros_like(nxt.reward)) * alive
        t = t + 1.0
        # FIRST arrival only: `arr` is latched at -1 until the goal is reached,
        # so a body that touches the goal and drifts out again keeps its
        # original arrival time. Measured while alive, so a body that flips
        # before arriving never records one.
        hit = jax.vmap(inner.at_goal)(nxt.data) * alive
        arr = jp.where((arr < 0) & (hit > 0), t, arr)
        alive = alive * (1.0 - nxt.done)
        return (nxt, rng, ret, cost, alive, t, arr), None

    z = jp.zeros(n_envs)
    (state, _, ret, cost, alive, _, arr), _ = jax.lax.scan(
        body,
        (state, rng, z, z, jp.ones(n_envs), 0.0, -jp.ones(n_envs)),
        None,
        length=n_decisions,
    )
    return ret, cost, alive, arr


def time_fitness(arrival, ret, n_decisions, ceiling):
    """Single scalar, LOWER IS BETTER, for ranking morphologies by speed.

    Two regimes composed into one continuous objective, because NSGA-II needs
    one number per objective, not a rule:

        arrived      -> the arrival decision itself
        did not      -> n_decisions + (shortfall fraction) * n_decisions

    So every arrival outranks every non-arrival, and non-arrivals are ordered
    by how far short they fell. Return is used for the shortfall because on
    this reward it telescopes to distance covered.

    WHY THIS EXISTS. Episode return cannot rank these bodies at all: measured
    on the 50M conditioned run, arrival times spanned 175-328 decisions (1.87x)
    while returns were 22.49-22.51 (std 0.003). The reward saturates the moment
    the goal is closed, so a fitness built on it is blind to speed -- an
    NSGA-II run would have had no signal on that objective whatsoever.
    """
    arrived = arrival >= 0
    shortfall = np.clip(1.0 - np.asarray(ret) / ceiling, 0.0, 1.0)
    return np.where(arrived, np.asarray(arrival),
                    n_decisions * (1.0 + shortfall))


def per_body(values: jp.ndarray, num_morph: int) -> tuple[np.ndarray, np.ndarray]:
    """Group per-lane values by morphology. Lanes are blocked, not interleaved:
    randomization_fn uses np.repeat, so lane order is [m0]*r + [m1]*r + ...
    """
    v = np.asarray(values).reshape(num_morph, -1)
    return v.mean(axis=1), v.std(axis=1) / max(1, np.sqrt(v.shape[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--robot", default="ant")
    ap.add_argument("--task", default="minefield", choices=sorted(_TASKS))
    ap.add_argument("--num_morphologies", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8,
                    help="Episodes PER BODY. Total lanes = this x num_morphologies.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Must match the TRAINING seed, or the population "
                         "evaluated is not the population trained on.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max_decisions", type=int, default=0,
                    help="Truncate the rollout (testing only). 0 = full episode.")
    args = ap.parse_args()

    kwargs = train_ppo.robot_env_kwargs(args.robot)
    # MEASUREMENT ENV MUST NOT TERMINATE ON THE GOAL, even though training now
    # does by default (2026-08-18). BraxAutoResetWrapper replaces `data` with
    # `first_state` on the very step `done` fires, so `at_goal(nxt.data)` in
    # rollout() would be evaluating the RESET pose and never see the arrival it
    # exists to detect -- every body silently reads 0% arrived, which is what
    # happened the first time this ran after the default flipped. Arrival is
    # latched here explicitly, so a fixed full-length horizon is both correct
    # and comparable with every number recorded before the flip.
    kwargs["terminate_on_goal"] = False
    env = _TASKS[args.task](
        robot=args.robot, morphology_conditioning=True, **kwargs
    )
    n_envs = args.num_morphologies * args.episodes
    ep_len = train_ppo._ROBOT_DEFAULTS[args.robot]["episode_length"]
    action_repeat = train_ppo._ROBOT_DEFAULTS[args.robot]["action_repeat"]
    n_decisions = args.max_decisions or (ep_len // action_repeat)

    print(f"robot={args.robot} task={args.task} bodies={args.num_morphologies} "
          f"episodes/body={args.episodes} decisions={n_decisions}")
    policy_fn = load_policy(args.checkpoint, env.observation_size,
                            env.action_size, args.deterministic)

    # Same seed AND same builder as training, so this is the trained population.
    batched, in_axes, genes = morphology_lib.randomization_fn(
        env.mjx_model, jax.random.PRNGKey(args.seed),
        args.num_morphologies, n_envs,
        model_builder=env.build_morphology_model,
    )

    def run(gene_batch, label):
        wrapped = MorphologyDomainRandomizationWrapper(env, batched, in_axes, gene_batch)
        w = train_ppo.wrap_for_brax_training(
            wrapped, episode_length=ep_len, action_repeat=action_repeat,
            already_batched=True)
        ret, cost, alive, arr = jax.jit(
            lambda: rollout(w, policy_fn, n_envs, n_decisions, args.seed + 1))()
        return (per_body(ret, args.num_morphologies),
                per_body(cost, args.num_morphologies),
                np.asarray(ret), np.asarray(arr))

    # 2 x goal distance: both movement terms telescope, so a straight full
    # traverse scores the goal distance twice. Used only to detect saturation.
    CEILING = 2.0 * float(env._corridor_length) * 0.918

    print("\n=== 1. PER-MORPHOLOGY RETURN (correct genes) ===")
    (r_mu, r_se), (c_mu, _), r_raw, arr_raw = run(genes, "correct")
    mass = np.asarray(batched.body_subtreemass[:, env.mj_model.body("robot").id])
    gear = np.asarray(batched.actuator_gear[:, 0, 0])
    step = n_envs // args.num_morphologies
    print(f"{'body':>5}{'mass kg':>9}{'gear':>8}{'return':>10}{'+-SE':>8}{'cost':>9}")
    for i in range(args.num_morphologies):
        tag = " (nominal)" if i == 0 else ""
        print(f"{i:>5}{mass[i*step]:>9.1f}{gear[i*step]:>8.0f}"
              f"{r_mu[i]:>10.2f}{r_se[i]:>8.2f}{c_mu[i]:>9.1f}{tag}")
    print(f"{'MEAN':>5}{'':>9}{'':>8}{r_mu.mean():>10.2f}")
    print(f"  spread: worst {r_mu.min():.2f}  best {r_mu.max():.2f}")
    print("  -> if the spread is wide, the mean in the training log is a fiction")

    print("\n=== 2. GENE ABLATION (each body given the NEXT body's genes) ===")
    rolled = jp.roll(genes, shift=step, axis=0)
    (a_mu, a_se), _, _, _ = run(rolled, "ablated")
    print(f"{'body':>5}{'correct':>10}{'wrong genes':>13}{'drop':>9}")
    for i in range(args.num_morphologies):
        print(f"{i:>5}{r_mu[i]:>10.2f}{a_mu[i]:>13.2f}{r_mu[i]-a_mu[i]:>9.2f}")
    drop = r_mu.mean() - a_mu.mean()
    rel = 100 * drop / abs(r_mu.mean()) if r_mu.mean() else float("nan")
    print(f"{'MEAN':>5}{r_mu.mean():>10.2f}{a_mu.mean():>13.2f}{drop:>9.2f}  ({rel:.0f}%)")
    print("\n  VERDICT:")
    # Threshold is RELATIVE to the return, not to SE. An earlier version used
    # `drop < 2 * mean(SE)` and got it backwards on the first real checkpoint
    # (2026-08-17): every episode hit the ceiling exactly, so SE collapsed to
    # 0.003, the threshold became ~0.007, and a 0.04% drop "cleared" it and was
    # reported as evidence of gene use. A variance-based threshold is
    # meaningless on a saturated distribution -- it becomes infinitely
    # sensitive precisely when the measurement is least informative.
    saturated = float(r_mu.std()) < 0.05 and float(r_mu.mean()) > 0.95 * CEILING
    if saturated:
        print(f"    TASK IS SATURATED: every body returns ~{r_mu.mean():.2f} of a")
        print(f"    ~{CEILING:.2f} ceiling, spread {r_mu.std():.3f}. THE ABLATION CANNOT")
        print("    RESOLVE ANYTHING HERE -- a single gait and a perfectly")
        print("    conditioned policy both score the maximum, so 'no drop' is")
        print("    not evidence either way. Make the task harder (wider gene")
        print("    range, shorter episode, or a body that needs a real gait")
        print("    change) before re-running this control.")
    elif abs(rel) < 2.0:
        print(f"    Wrong genes cost {drop:.2f} ({rel:.1f}%) -- NO meaningful drop.")
        print("    The policy is IGNORING the genes: it learned one gait that")
        print("    happens to work across bodies. The conditioning is")
        print("    decorative, and per-body returns are not evidence for it.")
    else:
        print(f"    Wrong genes cost {drop:.2f} return ({rel:.1f}%). The policy is")
        print("    genuinely using the conditioning signal.")

    print("\n=== 4. TIME-TO-GOAL FITNESS (the objective return cannot provide) ===")
    fit_lane = time_fitness(arr_raw, r_raw, n_decisions, CEILING)
    f_mu, f_se = per_body(jp.asarray(fit_lane), args.num_morphologies)
    # ARRIVAL TIME AND FITNESS ARE DIFFERENT NUMBERS and must be reported
    # separately. Fitness averages the miss penalty in; arrival time is a mean
    # over ARRIVED episodes only. They coincide at 100% arrival and diverge
    # sharply below it -- body 0 at 88% printed 243 decisions / 19.4 s when it
    # had never once taken that long, because one missed episode charged 625.
    arr_lane = np.asarray(arr_raw).reshape(args.num_morphologies, -1)
    hit = arr_lane >= 0
    n_hit = hit.sum(axis=1)
    arrived = n_hit / arr_lane.shape[1]
    with np.errstate(invalid="ignore"):
        t_mu = np.where(n_hit > 0, np.where(hit, arr_lane, 0).sum(axis=1)
                        / np.maximum(n_hit, 1), np.nan)
    # Control period: action_repeat decisions x n_substeps=2 physics steps
    # (GoToGoal.step's hardcoded value) x the model's own timestep.
    ctrl_dt = float(env.mj_model.opt.timestep) * 2 * action_repeat
    order = np.argsort(f_mu)
    rank = {int(b): i + 1 for i, b in enumerate(order)}
    print(f"{'body':>5}{'arrived':>9}{'decisions':>11}{'seconds':>9}"
          f"{'fitness':>9}{'rank':>6}{'cost':>9}")
    for i in range(args.num_morphologies):
        dec = f"{t_mu[i]:.0f}" if n_hit[i] else "--"
        secs = f"{t_mu[i]*ctrl_dt:.1f}" if n_hit[i] else "--"
        print(f"{i:>5}{100*arrived[i]:>8.0f}%{dec:>11}{secs:>9}"
              f"{f_mu[i]:>9.0f}{rank[i]:>6}{c_mu[i]:>9.0f}")
    if (arrived < 1.0).any():
        print("  decisions/seconds are over ARRIVED episodes only; fitness "
              f"charges a miss {n_decisions}-{2*n_decisions}")
    print(f"\n  fitness spread {f_mu.max()/f_mu.min():.2f}x   std {f_mu.std():.0f}"
          f"   (return std was {r_mu.std():.3f})")
    if f_mu.std() > 1.0:
        print("  -> USABLE as an NSGA-II objective, unlike return")
    print(f"  corr(time, cost) = {np.corrcoef(f_mu, c_mu)[0,1]:+.2f}"
          "   (near zero = two genuinely independent objectives)")

    print("\n=== 3. VS SPECIALIST (lane 0 = nominal ant) ===")
    print(f"  conditioned, nominal body : {r_mu[0]:.2f} +- {r_se[0]:.2f}")
    print(f"  specialist  (2026-08-16)  : 22.10   [checkpoints/ant_minefield_chain]")
    print(f"  ceiling (2 x 11.0 m goal) : ~22.50")
    print(f"  -> conditioned reaches {100*r_mu[0]/22.10:.0f}% of the specialist")


if __name__ == "__main__":
    main()

"""Evaluate a co-design run's EVOLVED body, or a single-body run's NOMINAL ant,
with the trainer's own evaluator.

The eval/* lines a co-design run prints come from 16 fixed random bodies, not
from the design the search converged on. This builds the minefield env with
the run's exact cost terms (linear hazard cost at the 2 cm gate, flip cost and
exit cost -- those two are trainer flags, so a plain env would have them at
0), the run's hazard size and placement, installs the evolved body (the mode
of the final design distribution in design_state.npz), and runs it through
ConstraintsEvaluator: the same metrics as the trainer's eval lines. Cost and
`safe` are per env, as there: each env's episodes are averaged first, and safe
means that average is at or under the budget (the trainer's line uses strictly
under, which no policy can meet at budget 0).

A run with no design_state.npz is a single-body run: the nominal ant is
evaluated the same way (no genes in the observation), so the two are scored by
one protocol and can share a table.

`hazard_free` is the share of episodes with NO hazard contact at all (zero
hazard steps) -- what a viewer notices, and what the mean cost hides. When it
is 100%, the line also gives the 95% upper bound on the per-episode touch rate
(rule of three: 3/n).

SEEDS. `--seeds 0 1 2` evaluates each seed in turn without recompiling; a
seed reproduces exactly what `--seeds <that seed>` alone prints. On the fixed
lattice the layout and the spawn never change, so seeds only change the
policy's action noise: under `--deterministic` every episode starts from the
same state and the seeds have nothing to vary. On random fields they also
redraw the hazards.

Runs on either stack. For runs trained before the Warp migration use env
mjx-safety-gym, whose package is the pre-warp checkout, so the physics is the
one the run trained on:

    ~/miniconda3/envs/mjx-safety-gym/bin/python scripts/eval_design.py \\
        --run checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M --budget 2
    ... --hazard_size 0.23          # the bigger-hazard run
    ... --hazard_placement random   # the random-field run
    ... --run checkpoints/vast/ant_minefield_final_eps2_b0_250M --budget 0   # nominal
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
from mujoco import mjx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_checkpoint import build_policy  # noqa: E402

from mjx_safety_gym import design as design_lib  # noqa: E402
from mjx_safety_gym import morphology as morph_lib  # noqa: E402
from mjx_safety_gym.algorithms import train_ppo  # noqa: E402
from mjx_safety_gym.algorithms.rl.evaluation import ConstraintsEvaluator  # noqa: E402
from mjx_safety_gym.envs.minefield import Minefield  # noqa: E402

try:  # after the Warp migration models go through the backend module
    from mjx_safety_gym import backend as _backend  # noqa: E402

    put_model = _backend.put_model
except ImportError:  # pre-warp checkout: MJX-JAX only
    put_model = mjx.put_model


def step_dir(run: pathlib.Path) -> pathlib.Path:
    steps = sorted(p for p in run.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else run


def evolved_spec(step: pathlib.Path) -> tuple[morph_lib.MorphologySpec, np.ndarray]:
    z = np.load(step / design_lib.DesignLoop.SIDECAR, allow_pickle=True)
    gmm = design_lib.GmmDesignDistribution(
        n_params=morph_lib.NUM_GENES, n_components=int(np.asarray(z["gmm_log_mixprobs"]).size))
    gmm.load_state_dict({k[len("gmm_"):]: z[k] for k in z.files if k.startswith("gmm_")})
    mode = gmm.mode()
    return morph_lib.MorphologySpec(gmm.to_genes(mode)), mode


def summarise(per_ep: dict[str, np.ndarray], per_env: dict[str, np.ndarray]) -> dict:
    """Mean, sd and se of every metric; cost and safe are per env (see top)."""
    out = {}
    for k, v in {**per_ep, **per_env}.items():
        out[k] = (v.mean(), v.std(), v.std() / np.sqrt(v.size), v.size)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True, help="co-design or single-body checkpoint run dir")
    ap.add_argument("--budget", type=float, required=True, help="the run's --safety_budget")
    ap.add_argument("--hazard_size", type=float, default=0.16)
    ap.add_argument("--hazard_placement", default="lattice", choices=["lattice", "random"])
    ap.add_argument("--flip_cost", type=float, default=50.0)
    ap.add_argument("--exit_cost", type=float, default=50.0)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=4, help="per env: total = envs x episodes")
    ap.add_argument("--deterministic", action="store_true", help="mean action (trainer default: sampled)")
    ap.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    step = step_dir(args.run.resolve())
    codesign = (step / design_lib.DesignLoop.SIDECAR).exists()
    want = train_ppo.checkpoint_obs_width(ocp.PyTreeCheckpointer().restore(str(step))[1]["policy"])
    env = Minefield(
        robot="ant", morphology_conditioning=codesign, hazard_size=args.hazard_size,
        hazard_placement=args.hazard_placement, flip_cost=args.flip_cost,
        exit_cost=args.exit_cost, **train_ppo.robot_env_kwargs("ant"))
    assert env.observation_size == want, (
        f"env obs {env.observation_size} != checkpoint {want}: wrong settings for this run")
    if codesign:
        spec, mode = evolved_spec(step)
        mj = env.build_morphology_model(spec)
        env._mj_model = mj
        env._mjx_model = put_model(mj)
        env._morphology_genes = jp.asarray(spec.genes, dtype=jp.float32)
        body = f"evolved body, mode {np.round(mode, 2)}"
    else:
        mj = env._mj_model
        body = "nominal body (single-body run)"
    print(f"{args.run.name} @ {step.name}: {body}, "
          f"mass {mj.body_subtreemass[mj.body('robot').id]:.1f} kg, solver "
          f"{mj.opt.iterations}/{mj.opt.ls_iterations}, mujoco {__import__('mujoco').__version__}")
    if args.deterministic and args.hazard_placement == "lattice" and (len(args.seeds) > 1 or args.episodes > 1):
        print("note: deterministic policy on the fixed lattice -- every episode starts from the same "
              "state, so the episodes (and seeds) are copies of one another")

    policy = build_policy(args.run.resolve(), env, args.deterministic, args.seeds[0])
    defaults = train_ppo._ROBOT_DEFAULTS["ant"]
    ep_len, ar = defaults["episode_length"], defaults["action_repeat"]
    evaluator = ConstraintsEvaluator(
        train_ppo.wrap_for_brax_training(env, ep_len, ar), lambda _: policy,
        num_eval_envs=args.envs, episode_length=ep_len, action_repeat=ar,
        key=jax.random.PRNGKey(args.seeds[0]), budget=args.budget,
        num_episodes=args.episodes, sim_dt=env.sim_dt)

    ep_keys = ("reward", "speed", "hazard_steps", "flipped", "out_of_bounds")
    env_keys = ("cost", "safe")
    pooled_ep: dict[str, list] = {k: [] for k in (*ep_keys, "hazard_free")}
    pooled_env: dict[str, list] = {k: [] for k in env_keys}
    n = args.envs * args.episodes
    print(f"{n} episodes per seed, budget {args.budget:g}, "
          f"{'deterministic' if args.deterministic else 'stochastic'} policy")
    for seed in args.seeds:
        # run_evaluation splits self._key first, so resetting it here makes this
        # seed's numbers identical to a fresh `--seeds <seed>` run, and reuses
        # the compiled unroll instead of building a new evaluator per seed.
        evaluator._key = jax.random.PRNGKey(seed)
        m = evaluator.run_evaluation(None, {}, aggregate_episodes=False)
        per_ep = {k: np.asarray(m[f"eval/episode_{k}"], dtype=float).ravel() for k in ep_keys}
        per_ep["hazard_free"] = (per_ep["hazard_steps"] == 0).astype(float)
        per_env = {k: np.asarray(m[f"eval/episode_{k}"], dtype=float).ravel() for k in env_keys}
        # The evaluator's `safe` is cost < budget, which nothing can satisfy at
        # budget 0; the constraint is cost <= budget, so score that instead.
        per_env["safe"] = (per_env["cost"] <= args.budget).astype(float)
        for k, v in per_ep.items():
            pooled_ep[k].append(v)
        for k, v in per_env.items():
            pooled_env[k].append(v)
        s = summarise(per_ep, per_env)
        print(f"  seed {seed}: return {s['reward'][0]:.2f} | speed {s['speed'][0]:.2f} m/s | "
              f"cost {s['cost'][0]:.2f} +- {s['cost'][2]:.2f} | hazard steps {s['hazard_steps'][0]:.1f} | "
              f"hazard-free {100 * s['hazard_free'][0]:.1f}% | flipped {100 * s['flipped'][0]:.1f}% | "
              f"oob {100 * s['out_of_bounds'][0]:.1f}% | under budget {100 * s['safe'][0]:.0f}%")

    s = summarise({k: np.concatenate(v) for k, v in pooled_ep.items()},
                  {k: np.concatenate(v) for k, v in pooled_env.items()})
    print(f"all {len(args.seeds)} seed(s) pooled (cost and safe per env):")
    for k in ("reward", "speed", "cost", "hazard_steps", "hazard_free", "flipped", "out_of_bounds", "safe"):
        mean, sd, se, nk = s[k]
        print(f"  {k:14s} {mean:9.4f}  (sd {sd:.4f}, se {se:.4f}, n {nk})")
    free = np.concatenate(pooled_ep["hazard_free"])
    if free.all():
        print(f"  no hazard contact in {free.size} episodes: per-episode touch rate < "
              f"{100 * 3 / free.size:.2f}% (95%, rule of three)")


if __name__ == "__main__":
    main()

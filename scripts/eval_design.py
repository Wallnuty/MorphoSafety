"""Evaluate a co-design run's EVOLVED body with the trainer's own evaluator.

The eval/* lines a co-design run prints come from 16 fixed random bodies, not
from the design the search converged on. This builds the minefield env with
the run's exact cost terms (linear hazard cost at the 2 cm gate, flip cost and
exit cost -- those two are trainer flags, so a plain env would have them at
0), the run's hazard size and placement, installs the evolved body (the mode
of the final design distribution in design_state.npz), and runs it through
ConstraintsEvaluator: the same metrics as the trainer's eval lines. Cost and
`safe` are per env, as there: each env's episodes are averaged first, and safe
means that average is under the budget.

Runs on either stack. For runs trained before the Warp migration use env
mjx-safety-gym, whose package is the pre-warp checkout, so the physics is the
one the run trained on:

    ~/miniconda3/envs/mjx-safety-gym/bin/python scripts/eval_design.py \\
        --run checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M --budget 2
    ... --hazard_size 0.23          # the bigger-hazard run
    ... --hazard_placement random   # the random-field run
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True, help="co-design checkpoint run dir")
    ap.add_argument("--budget", type=float, required=True, help="the run's --safety_budget")
    ap.add_argument("--hazard_size", type=float, default=0.16)
    ap.add_argument("--hazard_placement", default="lattice", choices=["lattice", "random"])
    ap.add_argument("--flip_cost", type=float, default=50.0)
    ap.add_argument("--exit_cost", type=float, default=50.0)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=4, help="per env: total = envs x episodes")
    ap.add_argument("--deterministic", action="store_true", help="mean action (trainer default: sampled)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    step = step_dir(args.run.resolve())
    want = train_ppo.checkpoint_obs_width(ocp.PyTreeCheckpointer().restore(str(step))[1]["policy"])
    env = Minefield(
        robot="ant", morphology_conditioning=True, hazard_size=args.hazard_size,
        hazard_placement=args.hazard_placement, flip_cost=args.flip_cost,
        exit_cost=args.exit_cost, **train_ppo.robot_env_kwargs("ant"))
    assert env.observation_size == want, (
        f"env obs {env.observation_size} != checkpoint {want}: wrong settings for this run")
    spec, mode = evolved_spec(step)
    mj = env.build_morphology_model(spec)
    env._mj_model = mj
    env._mjx_model = put_model(mj)
    env._morphology_genes = jp.asarray(spec.genes, dtype=jp.float32)
    print(f"{args.run.name} @ {step.name}: mode {np.round(mode, 2)}, "
          f"mass {mj.body_subtreemass[mj.body('robot').id]:.1f} kg, solver "
          f"{mj.opt.iterations}/{mj.opt.ls_iterations}, mujoco {__import__('mujoco').__version__}")

    policy = build_policy(args.run.resolve(), env, args.deterministic, args.seed)
    defaults = train_ppo._ROBOT_DEFAULTS["ant"]
    ep_len, ar = defaults["episode_length"], defaults["action_repeat"]
    evaluator = ConstraintsEvaluator(
        train_ppo.wrap_for_brax_training(env, ep_len, ar), lambda _: policy,
        num_eval_envs=args.envs, episode_length=ep_len, action_repeat=ar,
        key=jax.random.PRNGKey(args.seed), budget=args.budget,
        num_episodes=args.episodes, sim_dt=env.sim_dt)
    m = evaluator.run_evaluation(None, {})
    n = args.envs * args.episodes
    keys = ("reward", "speed", "cost", "hazard_steps", "flipped", "out_of_bounds", "safe")
    print(f"{n} episodes, budget {args.budget:g}:")
    for k in keys:
        mean, std = m[f"eval/episode_{k}"], m[f"eval/episode_{k}_std"]
        # cost and safe are PER ENV: the evaluator averages cost over each
        # env's episodes first (safe = that average < budget), so their spread
        # is across args.envs values, not envs x episodes.
        nk = args.envs if k in ("cost", "safe") else n
        print(f"  {k:14s} {mean:9.4f}  (sd {std:.4f}, se {std / np.sqrt(nk):.4f}, n {nk})")


if __name__ == "__main__":
    main()

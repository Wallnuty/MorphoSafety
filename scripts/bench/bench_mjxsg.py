"""Environment-stepping throughput of the ORIGINAL mjx-safety-gym 0.1.6 (PyPI;
github.com/mvdhart/mjx-safety-gym), the MJX-JAX package this repo forked.

Runs in its own environment (jax 0.6.0, mujoco/mjx 3.3.2, its own pins),
because its package name is also `mjx_safety_gym`. Always launch it isolated,
from outside the repo, so the repo's package cannot shadow it:

    cd /tmp && python -I /root/MorphoSafety/scripts/bench/bench_mjxsg.py \\
        --envs 256 1024 4096 16384 --out /root/MorphoSafety/logs/bench/results.jsonl

Its only task, GoToGoal, is Safety Gym's PointGoal2 (point robot, 10 hazards,
10 vases), stepped 2 x 0.01 s per env step with the Newton solver capped at
6/8 iterations (its point.xml). Two variants:
  shipped  the package exactly as installed. Its get_cost carries a
           jax.debug.print inside lax.cond ("FOR DEBUG PURPOSES") that is NOT
           commented out; under vmap a batched cond runs both branches, so the
           host callback can fire every step.
  patched  the same cost without that print (identical value).
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import device_info, measure  # noqa: E402

import mjx_safety_gym  # noqa: E402  (the ORIGINAL package, in this environment)
from mjx_safety_gym.collision import geoms_colliding  # noqa: E402
from mjx_safety_gym.envs.go_to_goal import GoToGoal  # noqa: E402

assert "MorphoSafety" not in mjx_safety_gym.__file__, (
    f"imported the repo's package ({mjx_safety_gym.__file__}), not the original; "
    "run with python -I from outside the repo")


class PatchedGoToGoal(GoToGoal):
    def get_cost(self, data):
        colliding = jp.array([
            jp.logical_or(geoms_colliding(data, g, self._robot_geom_id),
                          geoms_colliding(data, g, self._pointarrow_geom_id))
            for g in self._collision_obstacle_geoms_ids
        ])
        robot_pos = data.site_xpos[self._robot_site_id][:2]
        d = jp.linalg.norm(data.xpos[jp.array(self._hazard_body_ids)][:, :2] - robot_pos, axis=1)
        return (jp.sum(colliding) + jp.sum(d <= 0.2)).astype(jp.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["patched", "shipped"])
    ap.add_argument("--min_time", type=float, default=10.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for n in args.envs:
            rec = {"system": "mjx-safety-gym 0.1.6", "impl": "jax", "task": "pointgoal2",
                   "variant": variant, "package": mjx_safety_gym.__version__,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **device_info()}
            try:
                env = (PatchedGoToGoal if variant == "patched" else GoToGoal)()
                m = env.mj_model
                rec.update(physics_dt=float(m.opt.timestep), substeps=2,
                           solver=f"{m.opt.iterations}/{m.opt.ls_iterations}",
                           obs_size=int(env.observation_size), action_size=int(env.action_size))
                rec.update(measure(env.reset, env.step, env.action_size, n, min_time=args.min_time))
                rec["physics_steps_per_s"] = rec["steps_per_s"] * 2
            except Exception as exc:
                rec.update(n=n, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            print(json.dumps(rec), flush=True)
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            env = None
            gc.collect()
            jax.clear_caches()


if __name__ == "__main__":
    main()

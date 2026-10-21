"""Environment-stepping throughput of MorphoSafety (this repo), 2026-10-19.

Tasks, each with a matching run in bench_mjxsg.py / bench_sg.py:
  pointgoal2  point robot, 10 hazards (r 0.2), 10 vases, goal (Safety Gym's
              PointGoal2; the task mjx-safety-gym 0.1.6 implements)
  antgoal2    the same arena with the Safety Gym ant
  minefield   this repo's training task: ant, random hazard field, the
              training observation (lidar, 7x7 foot grids, lateral offset)

One env step = 2 physics steps of 0.01 s in every task (the env's own
n_substeps), so steps/s compare like for like across systems.

    python scripts/bench/bench_morphosafety.py --task pointgoal2 --impl warp \\
        --envs 256 1024 4096 16384 --out logs/bench/results.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import jax

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import device_info, measure  # noqa: E402

from mjx_safety_gym import backend  # noqa: E402
from mjx_safety_gym.algorithms import train_ppo as T  # noqa: E402
from mjx_safety_gym.envs.go_to_goal import GoToGoal  # noqa: E402

MINEFIELD_FLAGS = [
    "--robot", "ant", "--task", "minefield", "--hazard_placement", "random",
    "--hazard_lidar", "--no-foot_obstacle_obs", "--foot_hazard_grid", "7",
    "--hazard_cost_shape", "linear", "--ground_contact_eps", "0.02", "--finish_line",
    "--no-corridor_walls", "--terminate_out_of_bounds", "--exit_cost", "50",
    "--boundary_cost_weight", "0", "--flip_cost", "50",
]


def build(task, impl, n, solver):
    backend.configure(impl=impl, nworld=n)
    kw = {}
    if solver:
        kw = dict(solver_iterations=solver[0], solver_ls_iterations=solver[1])
    if task == "pointgoal2":
        return GoToGoal(robot="point", num_hazards=10, num_vases=10, hazard_size=0.2, **kw)
    if task == "antgoal2":
        return GoToGoal(robot="ant", num_hazards=10, num_vases=10, hazard_size=0.2, **kw)
    flags = MINEFIELD_FLAGS + ["--impl", impl, "--num_envs", str(n)]
    if solver:
        flags += ["--solver_iterations", str(solver[0]), "--solver_ls_iterations", str(solver[1])]
    a = T.build_argparser().parse_args(flags)
    T.apply_robot_defaults(a)
    return T.build_task_env(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["pointgoal2", "antgoal2", "minefield"])
    ap.add_argument("--impl", required=True, choices=["warp", "jax"])
    ap.add_argument("--envs", type=int, nargs="+", required=True)
    ap.add_argument("--solver", type=int, nargs=2, default=None, help="iterations ls_iterations")
    ap.add_argument("--min_time", type=float, default=10.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n in args.envs:
        rec = {"system": "morphosafety", "impl": args.impl, "task": args.task,
               "variant": f"solver {args.solver[0]}/{args.solver[1]}" if args.solver else "default",
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **device_info()}
        try:
            env = build(args.task, args.impl, n, args.solver)
            m = env.mj_model
            rec.update(physics_dt=float(m.opt.timestep), substeps=2,
                       solver=f"{m.opt.iterations}/{m.opt.ls_iterations}",
                       obs_size=int(env.observation_size), action_size=int(env.action_size))
            rec.update(measure(env.reset, env.step, env.action_size, n, min_time=args.min_time))
            rec["physics_steps_per_s"] = rec["steps_per_s"] * 2
        except Exception as exc:  # OOM and friends: record, keep going
            rec.update(n=n, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        print(json.dumps(rec), flush=True)
        with args.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        env = None
        gc.collect()
        jax.clear_caches()


if __name__ == "__main__":
    main()

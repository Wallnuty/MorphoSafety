"""Environment-stepping throughput of Safety-Gymnasium 1.0.0 (CPU MuJoCo).

    python bench_sg.py --task pointgoal2 --physics matched --mode vector \\
        --out /root/MorphoSafety/logs/bench/results.jsonl

Tasks are Safety-Gymnasium's own SafetyPointGoal2-v0 and SafetyAntGoal2-v0
(10 hazards, 10 vases). Physics:
  native   as shipped: the point steps 10 x 0.002 s per env step, the ant
           10 x 0.01 s (frameskip_binom_n = 10), default solver, condim 6
  matched  the setup of the GPU runs: 2 physics steps of 0.01 s per env step;
           point with the Newton solver at 6/8 iterations (mjx-safety-gym's
           point.xml); ant at condim 3 (this repo's ant.xml; full solver)
Safety-Gymnasium rebuilds its MjModel from XML at every reset, so the matched
settings are re-applied after each reset, and the rebuild itself is part of
what is timed -- it is part of using the package.

Modes: `single` (one env, one process) and `vector` (SafetyAsyncVectorEnv,
one process per env, default one per CPU core). Random actions.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import platform
import time

import gymnasium
import numpy as np
import safety_gymnasium

TASKS = {"pointgoal2": "SafetyPointGoal2-v0", "antgoal2": "SafetyAntGoal2-v0"}


def usable_cpus() -> int:
    """CPUs this container may use: the cgroup quota if one is set, else the
    affinity mask. os.cpu_count() reports the HOST's cores inside docker."""
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


class MatchedPhysics(gymnasium.Wrapper):
    """Re-applies the matched physics after every reset (the model is rebuilt)."""

    def __init__(self, env, frameskip, timestep=None, iterations=None, ls_iterations=None, condim=None):
        super().__init__(env)
        self.cfg = dict(frameskip=frameskip, timestep=timestep, iterations=iterations,
                        ls_iterations=ls_iterations, condim=condim)

    def _apply(self):
        task = self.env.unwrapped.task
        task.sim_conf.frameskip_binom_n = self.cfg["frameskip"]
        task.sim_conf.frameskip_binom_p = 1.0
        m = task.model
        if self.cfg["timestep"]:
            m.opt.timestep = self.cfg["timestep"]
        if self.cfg["iterations"]:
            m.opt.iterations = self.cfg["iterations"]
        # MuJoCo 2.3.3 (Safety-Gymnasium's pin) has no line-search iteration
        # cap; MjOption.ls_iterations arrived in MuJoCo 3. Set it only if present.
        if self.cfg["ls_iterations"] and hasattr(m.opt, "ls_iterations"):
            m.opt.ls_iterations = self.cfg["ls_iterations"]
        if self.cfg["condim"]:
            m.geom_condim[:] = self.cfg["condim"]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._apply()
        return out

    def step(self, action):
        return self.env.step(action)


def wrapper_for(task, physics):
    if physics == "native":
        return None
    if task == "pointgoal2":
        return functools.partial(MatchedPhysics, frameskip=2, timestep=0.01, iterations=6, ls_iterations=8)
    return functools.partial(MatchedPhysics, frameskip=2, condim=3)


def physics_of(env):
    task = env.unwrapped.task
    m = task.model
    return dict(physics_dt=float(m.opt.timestep), substeps=int(task.sim_conf.frameskip_binom_n),
                solver=f"{m.opt.iterations}/{getattr(m.opt, 'ls_iterations', 'n/a')}",
                mujoco=__import__("mujoco").__version__,
                condim=sorted(set(int(c) for c in m.geom_condim)))


def run_single(env_id, wrap, min_time):
    env = safety_gymnasium.make(env_id)
    if wrap:
        env = wrap(env)
    env.reset(seed=0)
    phys = physics_of(env)
    for _ in range(200):  # warm up
        o, r, c, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    steps, resets, t0 = 0, 0, time.time()
    while time.time() - t0 < min_time:
        for _ in range(100):
            o, r, c, term, trunc, _ = env.step(env.action_space.sample())
            steps += 1
            if term or trunc:
                env.reset()
                resets += 1
    el = time.time() - t0
    env.close()
    return dict(n=1, steps_per_s=steps / el, elapsed_s=el, resets=resets, **phys)


def run_vector(env_id, wrap, n, min_time):
    envs = safety_gymnasium.vector.make(env_id, num_envs=n, asynchronous=True,
                                        wrappers=[wrap] if wrap else None)
    envs.reset(seed=0)
    probe = safety_gymnasium.make(env_id)
    if wrap:
        probe = wrap(probe)
    probe.reset(seed=0)
    phys = physics_of(probe)
    probe.close()
    for _ in range(50):
        envs.step(envs.action_space.sample())
    steps, t0 = 0, time.time()
    while time.time() - t0 < min_time:
        for _ in range(20):
            envs.step(envs.action_space.sample())
            steps += n
    el = time.time() - t0
    envs.close()
    return dict(n=n, steps_per_s=steps / el, elapsed_s=el, **phys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--physics", required=True, choices=["matched", "native"])
    ap.add_argument("--mode", required=True, choices=["single", "vector"])
    ap.add_argument("--num_envs", type=int, default=None, help="default: usable CPUs")
    ap.add_argument("--min_time", type=float, default=20.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"system": "safety-gymnasium 1.0.0", "impl": "cpu", "task": args.task,
           "variant": f"{args.physics} physics, {args.mode}", "physics": args.physics, "mode": args.mode,
           "cpu": platform.processor() or platform.machine(), "cpu_count_host": os.cpu_count(),
           "cpu_usable": usable_cpus(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        wrap = wrapper_for(args.task, args.physics)
        if args.mode == "single":
            rec.update(run_single(TASKS[args.task], wrap, args.min_time))
        else:
            rec.update(run_vector(TASKS[args.task], wrap, args.num_envs or usable_cpus(), args.min_time))
        rec["physics_steps_per_s"] = rec["steps_per_s"] * rec["substeps"]
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    print(json.dumps(rec), flush=True)
    with args.out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

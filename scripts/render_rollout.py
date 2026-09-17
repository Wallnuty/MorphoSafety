"""Render a checkpoint's minefield rollout OFFSCREEN to an animated GIF.

For when the live viewer (main.py) cannot surface a window -- WSLg on this
laptop stopped mapping MuJoCo's GLFW window on 2026-09-17 -- or for sharing.
Uses MuJoCo's offscreen renderer through the same GLFW backend, which still
works headlessly here; set MUJOCO_GL=egl on a box that has EGL.

    MUJOCO_GL=glfw python scripts/render_rollout.py \
        --checkpoint checkpoints/vast/<run> --seed 3 --seconds 30 --out rollout.gif

Hazards light up while they are charging cost (GoToGoal.hazard_contacts, the
same test the cost sums), the camera tracks the torso, and the episode resets
on `done` exactly as training would. Observation width is reconciled to the
checkpoint (foot grid, foot obs, goal sensing) by train_ppo.build_env_for_checkpoint.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import jax
import mujoco
import numpy as np
import orbax.checkpoint as ocp
from mujoco import mjx
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_checkpoint import build_policy  # noqa: E402

from mjx_safety_gym.algorithms import train_ppo  # noqa: E402
from mjx_safety_gym.envs.minefield import Minefield  # noqa: E402
from mjx_safety_gym.envs.run_forward import RunForward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robot", default="ant")
    ap.add_argument("--task", choices=["minefield", "run"], default="minefield")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0, help="simulated seconds")
    ap.add_argument("--out", default="rollout.gif")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=8, help="render every Nth physics step")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--camera_distance", type=float, default=1.6)
    ap.add_argument("--camera_azimuth", type=float, default=120.0)
    ap.add_argument("--camera_elevation", type=float, default=-28.0)
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint)
    steps = sorted(p for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit())
    leaf = steps[-1] if steps else ckpt
    probe = ocp.PyTreeCheckpointer().restore(str(leaf.resolve()))
    want = train_ppo.checkpoint_obs_width(probe[1]["policy"])
    task = {"minefield": Minefield, "run": RunForward}[args.task]
    env, kw, _ = train_ppo.build_env_for_checkpoint(
        lambda **k: task(robot=args.robot, draw_corridor_lines=True, **k), args.robot, want
    )
    raw = env.unwrapped
    print(f"obs {env.observation_size}  foot_hazard_grid={kw.get('foot_hazard_grid')} "
          f"foot_obstacle_obs={kw.get('foot_obstacle_obs')}")
    policy = build_policy(ckpt, env, deterministic=args.deterministic, seed=0)
    action_repeat = train_ppo._ROBOT_DEFAULTS[args.robot]["action_repeat"]

    m = raw._mj_model
    d = mujoco.MjData(m)
    n_h = len(raw._hazard_body_ids)
    hz_geoms = np.array([m.geom(f"hazard_{i}_geom").id for i in range(n_h)], dtype=int)
    hz_mats = np.array([m.material(f"hazard_{i}_mat").id for i in range(n_h)], dtype=int)
    alpha0 = m.geom_rgba[hz_geoms, 3].copy()

    renderer = mujoco.Renderer(m, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = raw._robot_body_id
    cam.distance = args.camera_distance * float(getattr(raw, "_arena_scale", 1.0))
    cam.azimuth = args.camera_azimuth
    cam.elevation = args.camera_elevation

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    contacts = jax.jit(raw.hazard_contacts)
    rng = jax.random.PRNGKey(args.seed)
    state = reset(rng)
    site = raw._robot_site_id
    # GoToGoal.step calls mjx step with n_substeps=2, so one env step is 2 timesteps.
    sim_dt = 2.0 * float(m.opt.timestep)
    n_inner = int(args.seconds / sim_dt)
    frames: list[Image.Image] = []
    total_cost, ep, x0, t0 = 0.0, 1, float(state.data.site_xpos[site][0]), time.time()
    action = None
    for i in range(n_inner):
        if i % action_repeat == 0:
            rng, ak = jax.random.split(rng)
            action, _ = policy(state.obs, ak)
        state = step(state, action)
        total_cost += float(state.info["cost"])
        if i % args.every == 0:
            mjx.get_data_into(d, m, state.data)
            mujoco.mj_forward(m, d)
            hot = np.asarray(contacts(state.data))
            m.mat_emission[hz_mats] = np.where(hot, 1.0, 0.0)
            m.geom_rgba[hz_geoms, 3] = np.where(hot, 0.85, alpha0)
            renderer.update_scene(d, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
        if float(state.done):
            x = float(state.data.site_xpos[site][0])
            print(f"episode {ep}: ended at physics step {i}, x {x0:+.2f} -> {x:+.2f} m, "
                  f"cumulative cost {total_cost:.1f}")
            rng, k = jax.random.split(rng)
            state = reset(k)
            x0 = float(state.data.site_xpos[site][0])
            ep += 1
    print(f"rendered {len(frames)} frames in {time.time() - t0:.0f}s; cumulative cost {total_cost:.1f}")
    out = pathlib.Path(args.out)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 * args.every * sim_dt), loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

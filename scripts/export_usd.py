"""Record a policy rollout as an animated USD scene, for path-traced rendering.

Runs the policy exactly as main.py does (same env reconstruction, action held
for action_repeat steps) and writes every `--every`-th physics step through
MuJoCo's own USD exporter. Also writes `<out>/hazard_contacts.json`: per frame,
which hazards are being charged (GoToGoal.hazard_contacts, the same test the
cost uses), so the renderer can light them up like the viewer does.

    JAX_PLATFORMS=cpu python scripts/export_usd.py \
        --checkpoint checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M \
        --design_from checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M \
        --out renders/codesign_b2

then render it with scripts/render_cycles.py (in the `blender` conda env).

Lidar rings (sites) are not exported; the recording stops at the end of the
first episode so the video never shows a reset teleport.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import mujoco
import numpy as np
import orbax.checkpoint as ocp
from mujoco import mjx
from mujoco.usd.exporter import USDExporter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_checkpoint import build_policy  # noqa: E402

from mjx_safety_gym import design as design_lib  # noqa: E402
from mjx_safety_gym import morphology as morph_lib  # noqa: E402
from mjx_safety_gym.algorithms import train_ppo  # noqa: E402
from mjx_safety_gym.envs.minefield import Minefield  # noqa: E402


def evolved_spec(path: pathlib.Path) -> morph_lib.MorphologySpec:
    """The co-design mode, read the same way main.py --design_from does."""
    side = design_lib.DesignLoop.SIDECAR
    if not (path / side).is_file():
        cands = sorted(d for d in path.glob("*") if (d / side).is_file())
        if not cands:
            raise SystemExit(f"no {side} in {path} or its step dirs")
        path = cands[-1]
    z = np.load(path / side, allow_pickle=True)
    gmm = design_lib.GmmDesignDistribution(
        n_params=morph_lib.NUM_GENES, n_components=int(np.asarray(z["gmm_log_mixprobs"]).size)
    )
    gmm.load_state_dict({k[len("gmm_"):]: z[k] for k in z.files if k.startswith("gmm_")})
    print(f"evolved design from {path.name}: mode {np.round(gmm.mode(), 2)}")
    return morph_lib.MorphologySpec(gmm.to_genes(gmm.mode()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--design_from", default=None, help="co-design run/step dir: render its evolved body")
    ap.add_argument("--robot", default="ant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0, help="max simulated seconds")
    ap.add_argument("--every", type=int, default=2, help="export every Nth physics step (2 -> 25 fps)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--hazard_placement", choices=["lattice", "random"], default="lattice",
                    help="match the run's training layout (random: --seed picks the field)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    steps = sorted(p for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit())
    leaf = steps[-1] if steps else ckpt
    want = train_ppo.checkpoint_obs_width(ocp.PyTreeCheckpointer().restore(str(leaf))[1]["policy"])
    conditioned = args.design_from is not None
    env, _, _ = train_ppo.build_env_for_checkpoint(
        lambda **k: Minefield(robot=args.robot, draw_corridor_lines=True,
                              morphology_conditioning=conditioned,
                              hazard_placement=args.hazard_placement, **k),
        args.robot, want,
    )
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    if conditioned:
        spec = evolved_spec(pathlib.Path(args.design_from).resolve())
        mj = raw.build_morphology_model(spec)
        raw._mj_model = mj
        raw._mjx_model = mjx.put_model(mj)
        raw._morphology_genes = jax.numpy.asarray(spec.genes, dtype=jax.numpy.float32)
        print("scales:", {k: round(v, 2) for k, v in spec.scales.items()})
    policy = build_policy(ckpt, env, deterministic=args.deterministic, seed=args.seed)
    action_repeat = train_ppo._ROBOT_DEFAULTS[args.robot]["action_repeat"]

    m = raw._mj_model
    d = mujoco.MjData(m)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exporter = USDExporter(model=m, output_directory="usd", output_directory_root=str(out),
                           max_geom=2000, verbose=False)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0  # lidar rings and other debug sites stay out of the render

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    contacts = jax.jit(raw.hazard_contacts)
    rng = jax.random.PRNGKey(args.seed)
    state = reset(rng)
    site = raw._robot_site_id
    x0 = float(state.data.site_xpos[site][0])
    # Exactly the prim names the exporter writes: "<geom name>_id<geom id>_geom".
    hz_names = [f"hazard_{i}_geom_id{m.geom(f'hazard_{i}_geom').id}_geom"
                for i in range(len(raw._hazard_body_ids))]
    frames, action, why = [], None, "time limit"
    n_steps = int(args.seconds / raw.sim_dt)
    for i in range(n_steps):
        if i % action_repeat == 0:
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
        state = step(state, action)
        if i % args.every == 0:
            mjx.get_data_into(d, m, state.data)
            mujoco.mj_forward(m, d)
            exporter.update_scene(d, scene_option=opt)
            frames.append([int(v) for v in np.asarray(contacts(state.data))])
        if float(state.done):
            arrived = float(raw.at_goal(state.data)) > 0
            why = "arrived" if arrived else "episode ended (flip or exit)"
            break
    # Declare metres and z-up: without these USD defaults to centimetres and
    # importers (Blender) shrink the whole scene 100x.
    from pxr import UsdGeom
    UsdGeom.SetStageMetersPerUnit(exporter.stage, 1.0)
    UsdGeom.SetStageUpAxis(exporter.stage, UsdGeom.Tokens.z)
    exporter.save_scene(filetype="usda")
    x1 = float(state.data.site_xpos[site][0])
    usd_file = sorted((out / "usd" / "frames").glob("*.usda"))[-1]
    meta = {"usd": str(usd_file), "fps": 1.0 / (raw.sim_dt * args.every), "frames": len(frames),
            "hazard_geoms": hz_names, "hazard_contacts": frames, "outcome": why,
            "dx": x1 - x0, "seconds": len(frames) * raw.sim_dt * args.every}
    (out / "hazard_contacts.json").write_text(json.dumps(meta))
    print(f"{len(frames)} frames at {meta['fps']:.0f} fps ({meta['seconds']:.1f} s sim), {why}, "
          f"dx {meta['dx']:+.2f} m -> {usd_file}")


if __name__ == "__main__":
    main()

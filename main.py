import argparse
import time
from pathlib import Path
from typing import Optional
import jax
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer

import orbax.checkpoint as ocp
from brax.training.acme import running_statistics

from mjx_safety_gym import jax_cache
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.algorithms.train_ppo import (
    _ROBOT_DEFAULTS,
    build_env_for_checkpoint,
    checkpoint_obs_width,
    checkpoint_policy_layers,
    latest_checkpoint,
)
from mjx_safety_gym.envs.go_to_goal import _ROBOT_XMLS, GoToGoal
from mjx_safety_gym.envs.lasers import Lasers
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.envs.run_forward import RunForward
import mjx_safety_gym.lidar as lidar

_parser = argparse.ArgumentParser(
    description="Replay the newest trained policy for a robot in the viewer, "
    "or drive it with random actions if nothing has been trained yet."
)
_parser.add_argument("--robot", choices=sorted(_ROBOT_XMLS), default="point")
_parser.add_argument(
    "--task",
    choices=["goal", "run", "minefield", "lasers"],
    default="goal",
    help="Must match the task the checkpoint was trained on. All three tasks "
    "give the same observation width, so a mismatch loads cleanly and replays a "
    "policy against a world it was never trained in.",
)
_parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Replay the mean action instead of sampling. Off by default: the cost "
    "critic is fit to sampled on-policy rollouts, so `safety_budget` only ever "
    "constrained the stochastic policy -- the deterministic one is behaviour "
    "the safety machinery never governed. Useful for inspection, misleading "
    "as a safety measurement.",
)
_parser.add_argument("--duration", type=float, default=20.0, help="Seconds to run.")
_parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for the arena layout and action sampling. Per-episode outcomes "
    "vary enormously -- across 64 layouts the unconstrained point policy "
    "averaged 2.8 goals but scored ZERO in 19%% of them, and episode cost "
    "ranged 0 to 961. Vary this before judging a policy by eye; one episode "
    "says almost nothing.",
)
_parser.add_argument(
    "--checkpoint",
    default=None,
    help="Checkpoint to replay, overriding the default checkpoints/<robot>. "
    "Accepts either a run directory (newest step inside is used) or a single "
    "step directory. Note the ENV is still built from --robot, so this must be "
    "a checkpoint trained on the same robot.",
)
_parser.add_argument(
    "--camera_track",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Point the camera at the robot and FOLLOW it. On by default: the free "
    "camera starts centred on the arena origin, which on the corridor tasks is "
    "the middle of an 11 m track with the robot 5.5 m behind it -- so every "
    "session began by hunting for the ant, and it then walked out of frame "
    "again. --no-camera_track restores the free camera.",
)
_parser.add_argument(
    "--hazard_highlight",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Light up each hazard while it is actually charging cost. Driven by "
    "GoToGoal.hazard_contacts -- the SAME per-hazard test get_cost sums -- so "
    "what lights up is exactly what is being charged, rather than a lookalike "
    "reimplementation that could drift. Note the step-on rule: a disc only "
    "fires while a robot geom is ON THE GROUND inside it, so an ant vaulting "
    "over a mine correctly stays dark. Viewer only.",
)
_parser.add_argument(
    "--camera_distance", type=float, default=4.0,
    help="Metres from the tracked robot. Scaled by the robot's arena_scale, so "
    "ant_gym (4x bigger) gets a proportionally wider shot.",
)
_parser.add_argument(
    "--camera_azimuth", type=float, default=120.0,
    help="Camera heading in degrees. 120 is a three-quarter view from behind "
    "and to the side, which shows foot placement against the hazard discs "
    "better than a pure side-on or straight-behind shot.",
)
_parser.add_argument(
    "--camera_elevation", type=float, default=-20.0,
    help="Camera pitch in degrees; negative looks down.",
)
_parser.add_argument(
    "--saute_budget",
    type=float,
    default=None,
    help="Replay a --penalizer saute checkpoint (its observation is 1 wider, "
    "carrying the remaining safety budget). MUST equal the --safety_budget the "
    "run trained with: the wrapper divides accumulated cost by it to produce "
    "that extra observation dim, so a wrong value feeds the policy an input it "
    "never saw. Nothing in the checkpoint records it -- training writes an "
    "empty ConfigDict -- so it has to be supplied here. Replay always uses "
    "penalty=0/terminate=False, matching the eval-side wrapper, so what you "
    "watch is raw behaviour rather than a budget-exhaustion artifact.",
)
_parser.add_argument(
    "--num_morphologies",
    type=int,
    default=0,
    help="Replay a MORPHOLOGY-CONDITIONED checkpoint. Must match the value the "
    "checkpoint was trained with, together with --seed, or the population "
    "reconstructed here is not the population it was trained on (bodies are "
    "sampled host-side from the seed, not stored in the checkpoint). 0 (the "
    "default) replays an ordinary single-body checkpoint.",
)
_parser.add_argument(
    "--morphology",
    type=int,
    default=0,
    help="Which body from that population to watch, 0-indexed. Body 0 is the "
    "NOMINAL ant by construction (morphology.randomization_fn's "
    "include_nominal), so --morphology 0 shows the unmodified robot and is the "
    "one directly comparable to a single-body checkpoint.",
)
_args = _parser.parse_args()

DURATION_SECONDS = _args.duration
ACTION_HOLD = 10  # resample a random action every N steps for smoother motion
ROBOT = _args.robot
DETERMINISTIC = _args.deterministic

# Compile once, and every later run loads the cached kernel from disk
jax_cache.configure()

def resolve_checkpoint(robot: str) -> Optional[Path]:
    """The checkpoint directory to replay, or None if nothing is trained."""
    if _args.checkpoint is None:
        return latest_checkpoint(robot)
    # Must be absolute: orbax rejects relative paths ("Checkpoint path
    # should be absolute"). latest_checkpoint() already returns absolute.
    ckpt = Path(_args.checkpoint).resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"--checkpoint path does not exist: {ckpt}")
    # Accept a run directory as well as a leaf step directory.
    steps = sorted(p for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else ckpt


# THE CHECKPOINT IS RESOLVED BEFORE THE ENV IS BUILT, and that ordering is the
# fix for a real crash rather than a preference. A checkpoint hard-codes the
# observation width its first layer accepts, and `goal_observation` (added
# 2026-08-15, ON by default for the ants) changes that width by +3, so every
# ant checkpoint trained before that date wants 76 where the defaults give 79.
# See train_ppo.build_env_for_checkpoint for the reconciliation.
ckpt_path = resolve_checkpoint(ROBOT)
# Restore through orbax directly, NOT brax's checkpoint.load: that helper
# builds restore_args with a blanket tree_map over the metadata, which blows up
# on the optimizer-state subtree we also save ("different types at key path ...
# list vs RestoreArgs").
# Saved layout is (normalizer, SafePPONetworkParams, penalizer, optimizer);
# orbax hands each dataclass back as a plain dict of its fields.
_loaded = None if ckpt_path is None else ocp.PyTreeCheckpointer().restore(str(ckpt_path))

_TASKS = {
    "run": RunForward,
    "minefield": Minefield,
    "lasers": Lasers,
    "goal": GoToGoal,
}
def _report_reconciliation(want, default_width, task_kwargs, robot):
    """Say WHICH flags were flipped to fit the checkpoint, not just that some were.

    The old message named goal_observation and goal_reward_weight and nothing
    else, which was complete back when those were the only width-changing
    flags. They no longer are -- lidar_groups (2026-08-15, again 2026-08-22)
    and foot_obstacle_obs (2026-08-24) both move the width too. A message that
    omits the flag that actually changed is worse than no message: it reads as
    confirmation that nothing else did.
    """
    if default_width is None:
        return
    from mjx_safety_gym.algorithms.train_ppo import robot_env_kwargs

    _missing = object()
    defaults = robot_env_kwargs(robot)
    changed = {
        k: v for k, v in task_kwargs.items() if defaults.get(k, _missing) != v
    }
    print(
        f"Checkpoint expects an observation of width {want}; this robot's "
        f"defaults give {default_width}."
    )
    if changed:
        print(
            "  -> rebuilt with "
            + ", ".join(f"{k}={v}" for k, v in sorted(changed.items()))
            + " to match it"
        )


_want = None if _loaded is None else checkpoint_obs_width(_loaded[1]["policy"])
if _args.num_morphologies:
    # A morphology-conditioned checkpoint is NUM_GENES wider than the task obs
    # it was trained against, so the width to reconcile is the checkpoint's --
    # the builder below adds conditioning, so the env it produces already
    # carries the genes and the comparison is apples to apples.
    #
    # THIS USED TO SKIP RECONCILIATION ENTIRELY and build straight from
    # robot_env_kwargs, on the reasoning that the search "only flips
    # goal_observation, worth +-3". That stopped being true the moment
    # foot_obstacle_obs landed (2026-08-24) defaulting ON for the ants: the
    # defaults jumped 47 -> 63, so every morphology checkpoint trained before
    # that date built a 70-wide env against a 54-wide policy and died in flax
    # with "expected (54, 256), got (70, 256)". _env_kwarg_candidates already
    # carries a no-feet variant; this path just was not consulting it.
    from mjx_safety_gym import morphology as _morph

    if _args.task == "goal":
        env = GoToGoal(robot=ROBOT, morphology_conditioning=True)
    else:
        env, _task_kwargs, _default_width = build_env_for_checkpoint(
            lambda **kw: _TASKS[_args.task](
                robot=ROBOT,
                morphology_conditioning=True,
                # VIEWER-ONLY: draws the corridor as a floor stripe instead of
                # two solid walls that hide the ant. Costs ~1.3% of env
                # stepping (ngeom 37 -> 39), so it is off by default and never
                # on in training. See RunForward._add_corridor_walls.
                draw_corridor_lines=True,
                **kw,
            ),
            ROBOT,
            _want,
            saute_budget=_args.saute_budget,
        )
        _report_reconciliation(_want, _default_width, _task_kwargs, ROBOT)

    # Reproduce randomization_fn's population EXACTLY: same PRNGKey -> same
    # host-side numpy seed -> same draws, with lane 0 pinned to nominal. The
    # genes are not stored in the checkpoint (they cannot be recovered from a
    # compiled mjx.Model), so --seed and --num_morphologies are what identify
    # which bodies these are.
    _seed = int(jax.random.randint(jax.random.PRNGKey(_args.seed), (), 0, 2**31 - 1))
    _rng = np.random.default_rng(_seed)
    _specs = [_morph.MorphologySpec.sample(_rng) for _ in range(_args.num_morphologies)]
    _specs[0] = _morph.MorphologySpec.nominal()
    if not 0 <= _args.morphology < _args.num_morphologies:
        raise SystemExit(
            f"--morphology {_args.morphology} out of range for "
            f"--num_morphologies {_args.num_morphologies}"
        )
    _spec = _specs[_args.morphology]

    # Install the body. This is exactly what MorphologyDomainRandomizationWrapper
    # does per lane, minus the vmap -- the viewer runs a single env, so the
    # attributes can just be set. BOTH models must be swapped: _mjx_model drives
    # the physics, _mj_model drives the renderer, and showing one body while
    # simulating another is the kind of thing that would look like a physics bug.
    # Topology is identical across morphologies, so the geom/body ids cached in
    # _post_init stay valid.
    # `.unwrapped`, not `env`: build_env_for_checkpoint returns a Saute WRAPPER
    # for a saute checkpoint, and a wrapper forwards attribute READS but not
    # writes -- so assigning to `env._mjx_model` would set a dead attribute on
    # the wrapper and silently leave the nominal body simulating underneath.
    _inner = getattr(env, "unwrapped", env)
    _mj = _inner.build_morphology_model(_spec)
    _inner._mj_model = _mj
    _inner._mjx_model = mjx.put_model(_mj)
    _inner._morphology_genes = jax.numpy.asarray(_spec.genes, dtype=jax.numpy.float32)

    _mass = float(_mj.body_subtreemass[_mj.body("robot").id])
    _scales = _spec.scales
    print(
        f"Morphology {_args.morphology}/{_args.num_morphologies - 1}"
        f"{' (NOMINAL)' if _args.morphology == 0 else ''}: "
        f"mass {_mass:.1f} kg, gear {_mj.actuator_gear[0, 0]:.0f}"
    )
    print(
        "  scales  "
        + "  ".join(f"{k}={v:.2f}" for k, v in _scales.items())
    )
elif _args.task == "goal":
    # GoToGoal has no goal-sensing flags to reconcile -- its goal already moves
    # and is already in lidar range.
    env = GoToGoal(robot=ROBOT)
else:
    env, _task_kwargs, _default_width = build_env_for_checkpoint(
        lambda **kw: _TASKS[_args.task](
            robot=ROBOT, draw_corridor_lines=True, **kw
        ),
        ROBOT,
        _want,
        saute_budget=_args.saute_budget,
    )
    if _args.saute_budget is not None and _want == env.observation_size:
        print(
            f"  -> wrapped in Saute(budget={_args.saute_budget}, penalty=0, "
            f"terminate=False) for the +1 budget observation"
        )
    _report_reconciliation(_want, _default_width, _task_kwargs, ROBOT)

rng = jax.random.PRNGKey(_args.seed)

# Reset environment
rng, rng_reset = jax.random.split(rng)
state = env.reset(rng_reset)
print(f"Robot: {ROBOT}")
print("Initial observation shape:", state.obs.shape)
print("Reported observation_size:", env.observation_size)
print("Action size:", env.action_size)

m = env.mj_model
d = mjx.get_data(m, state.data)

# JIT-compile up front so the loop below runs at full speed. Sampling is kept
# separate from stepping so the same action can be held for several frames.
def sample_action(rng):
    rng, rng_action = jax.random.split(rng)
    action = jax.random.uniform(
        rng_action, (env.action_size,), minval=-1.0, maxval=1.0
    )
    return action, rng


def build_policy(loaded, obs, action_size: int):
    """Build an action fn from already-restored params, or None if untrained.

    Rebuilds the network from the env's own shapes rather than from a saved
    config: the training code writes an empty ConfigDict, so brax's
    `checkpoint.load_policy` helper cannot reconstruct the network, and its
    vanilla PPONetworks has no cost-value head anyway.
    """
    if loaded is None:
        return None
    normalizer = running_statistics.RunningStatisticsState(**loaded[0])
    policy_params, value_params = loaded[1]["policy"], loaded[1]["value"]
    # normalize_observations defaults to False in ppo.train, so the
    # preprocessor is the identity -- must match training or the obs scale
    # the policy sees is wrong.
    # Layer widths come from the WEIGHTS, not from a default. A
    # morphology-conditioned run uses --policy_hidden_layer_sizes 256 256 256
    # 256, and rebuilding it with make_ppo_networks' (32,)*4 default raises a
    # flax shape error instead of loading -- the same class of failure as the
    # observation-width mismatch this file already handles above.
    layers = checkpoint_policy_layers(policy_params)
    extra = {} if layers is None else {"policy_hidden_layer_sizes": layers}
    if layers is not None and tuple(layers) != (32,) * 4:
        print(f"  checkpoint policy hidden layers: {tuple(layers)}")
    network = ppo_networks.make_ppo_networks(
        obs.shape,
        action_size,
        preprocess_observations_fn=lambda x, _: x,
        **extra,
    )
    policy = ppo_networks.make_inference_fn(network)(
        (normalizer, policy_params, value_params), deterministic=DETERMINISTIC
    )
    return jax.jit(lambda o, k: policy(o, k)[0])


policy_fn = build_policy(_loaded, state.obs, env.action_size)
if policy_fn is None:
    print(f"No checkpoint for '{ROBOT}' -- driving with random actions.")
else:
    mode = "deterministic" if DETERMINISTIC else "stochastic"
    # Print the age, not just the path. The default lookup picks the newest
    # checkpoint across every run directory for this robot, and "newest" is
    # only meaningful if you can see how old it is -- a five-day-old
    # checkpoint from a different task loads perfectly happily, since both
    # tasks share an observation width.
    age_h = (time.time() - ckpt_path.stat().st_mtime) / 3600
    age = f"{age_h:.1f} h old" if age_h < 48 else f"{age_h / 24:.1f} days old"
    print(f"Loaded {mode} policy from {ckpt_path}  ({age})")
    if _args.checkpoint is None:
        print("  (newest across all runs for this robot; override with --checkpoint)")

print("Compiling reset/step...")
start = time.time()
reset_fn = jax.jit(env.reset).lower(rng_reset).compile()
sample_fn = jax.jit(sample_action).lower(rng).compile()
action, rng = sample_fn(rng)
step_fn = jax.jit(env.step).lower(state, action).compile()
print(f"Compiled in {time.time() - start:.1f}s")

sim_dt = m.opt.timestep * 2  # env.step() runs 2 physics substeps internally
num_steps = int(DURATION_SECONDS / sim_dt)

# HOLD EACH ACTION FOR action_repeat STEPS. Training queries the policy once
# per DECISION and CostEpisodeWrapper holds that action across `action_repeat`
# env.steps -- but that wrapper is not in the replay path, so querying every
# step ran the policy at 4x the control period it was trained at (0.02 s vs
# 0.08 s for the ants). That is the same defect that invalidated
# eval_checkpoint.py on 2026-08-11, and this plan's own sweep measured control
# period as worth up to 2.7x in achievable gait travel -- it does not cancel
# out as a cosmetic difference. Sim duration is unchanged; only the rate at
# which the policy is re-queried is corrected.
ACTION_REPEAT = int(_ROBOT_DEFAULTS[ROBOT]["action_repeat"])
CTRL_DT = sim_dt * ACTION_REPEAT
print(f"Running {num_steps} steps (~{DURATION_SECONDS}s), "
      f"action_repeat={ACTION_REPEAT} -> {num_steps // ACTION_REPEAT} decisions "
      f"at {CTRL_DT:.3f}s control period")

total_cost = 0.0
decisions = 0        # decisions in the CURRENT episode -- comparable to the
                     # arrival numbers eval_morphology.py reports
episode = 1
# Hazard highlighting: resolve geom ids and remember the resting colour, so the
# highlight can be switched on and off per frame by writing `m.geom_rgba` (model
# data, read at render time -- no effect on physics).
# SELF-LIT, NOT RECOLOURED. An earlier version flipped the disc to orange, which
# reads as "a different object" rather than "this one is active". Emission makes
# MuJoCo render the geom as though it emits its own light, so the HUE IS
# UNTOUCHED -- only how brightly it burns. Opacity moves with it because a disc
# resting at 25% opacity swallows the glow; emission-only was tried and was too
# subtle to read. Every obstacle carries its OWN material (world.py,
# envs/lasers.py) precisely so they can be lit one at a time.
#
# ONE GENERIC LIST, not a block per obstacle type. `lasers` has zero hazards and
# `minefield` has zero beams, and the empty case is a real trap rather than a
# no-op: `np.array([])` is FLOAT, and indexing `mat_emission` with it raises
# "arrays used as indices must be of integer (or boolean) type". That is the
# same empty-array trap already guarded three times in the env code.
_highlights = []


def _register_highlight(prefix, count, contacts_attr, emis_hot, alpha_hot):
    """Collect (geom ids, mat ids, mask fn, resting values, lit values)."""
    if not count or not hasattr(env.unwrapped, contacts_attr):
        return
    try:
        gids = np.array(
            [m.geom(f"{prefix}_{i}_geom").id for i in range(count)], dtype=int
        )
        mids = np.array(
            [m.material(f"{prefix}_{i}_mat").id for i in range(count)], dtype=int
        )
    except KeyError:
        # A model built before per-obstacle materials existed. Highlighting is a
        # nicety; degrade to off rather than refusing to open the viewer.
        return
    _highlights.append(
        {
            "gids": gids,
            "mids": mids,
            "fn": jax.jit(getattr(env.unwrapped, contacts_attr)),
            "emis_base": m.mat_emission[mids].copy(),
            "alpha_base": m.geom_rgba[gids, 3].copy(),
            "emis_hot": emis_hot,
            "alpha_hot": alpha_hot,
        }
    )


if _args.hazard_highlight:
    _register_highlight(
        "hazard",
        len(getattr(env.unwrapped, "_hazard_body_ids", ())),
        "hazard_contacts",
        emis_hot=1.0,
        alpha_hot=0.85,
    )
    # Beams rest already glowing -- a laser that is dark reads as switched off --
    # so the lit state is a step UP from a nonzero base, not up from nothing.
    _register_highlight(
        "laser",
        len(getattr(env.unwrapped, "_laser_x", ())),
        "laser_contacts",
        emis_hot=1.0,
        alpha_hot=1.0,
    )

with mujoco.viewer.launch_passive(m, d) as viewer:
    if _args.camera_track:
        # TRACKING, not just an initial lookat: the corridor is 11 m long (44 m
        # for ant_gym) and a policy that works crosses all of it, so a camera
        # merely *placed* at the spawn loses the robot within seconds. Tracking
        # keeps it framed for the whole episode and across resets.
        # `.unwrapped` because the env may be Saute-wrapped.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.unwrapped._robot_body_id
        viewer.cam.distance = _args.camera_distance * getattr(
            env.unwrapped, "_arena_scale", 1.0
        )
        viewer.cam.azimuth = _args.camera_azimuth
        viewer.cam.elevation = _args.camera_elevation
    for i in range(num_steps):
        if not viewer.is_running():
            break
        step_start = time.time()

        if policy_fn is not None:
            # Re-queried once per DECISION, then held -- see ACTION_REPEAT.
            if i % ACTION_REPEAT == 0:
                rng, rng_action = jax.random.split(rng)
                action = policy_fn(state.obs, rng_action)
                decisions += 1
        elif i % ACTION_HOLD == 0:
            action, rng = sample_fn(rng)
        state = step_fn(state, action)

        # Safety cost readout: confirms hazard/collision costs actually register.
        step_cost = float(state.info["cost"])
        total_cost += step_cost
        if step_cost > 0:
            print(f"step {i}: cost={step_cost:.1f} (cumulative {total_cost:.1f})")

        # Keep the lidar rings + mocap bodies (goal, hazards) visually in sync.
        # Pull the lidar slice to host once (single transfer) so update_lidar_rings
        # iterates over NumPy floats instead of forcing ~48 tiny device->host syncs.
        # Sliced by the env's OWN ring count, not a hardcoded 3: RunForward and
        # Minefield emit only the obstacle ring, so a fixed 3 would read 32
        # proprioception entries as though they were lidar.
        n_rings = len(env.lidar_groups)
        lidar_vals = np.asarray(
            state.obs[: n_rings * lidar.NUM_LIDAR_BINS]
        ).reshape(n_rings, lidar.NUM_LIDAR_BINS)
        lidar.update_lidar_rings(lidar_vals, m, env.lidar_groups)
        mjx.get_data_into(d, m, state.data)
        mujoco.mj_forward(m, d)
        for _h in _highlights:
            # One small device->host transfer per group per frame, on the same
            # `state.data` get_cost consumes, so the highlight cannot disagree
            # with what is actually being charged.
            hot = np.asarray(_h["fn"](state.data))
            m.mat_emission[_h["mids"]] = np.where(
                hot, _h["emis_hot"], _h["emis_base"]
            )
            m.geom_rgba[_h["gids"], 3] = np.where(
                hot, _h["alpha_hot"], _h["alpha_base"]
            )
        viewer.sync()

        # ACT ON `done`. The env computes it (terminate_on_goal has been the
        # default since 2026-08-18, terminate_on_flip for longer), but this
        # loop used to discard it -- so a robot that reached the goal or went
        # over on its back just kept being stepped, showing states training
        # would never have continued from. Reset instead, so one viewer session
        # shows successive episodes on fresh layouts.
        if float(state.done) > 0:
            why = "done"
            if hasattr(env, "at_goal") and float(env.at_goal(state.data)) > 0:
                why = f"REACHED GOAL in {decisions} decisions ({decisions * CTRL_DT:.1f}s)"
            elif hasattr(env, "is_flipped") and float(env.is_flipped(state.data)) > 0:
                why = f"flipped over after {decisions} decisions"
            print(f"  episode {episode}: {why}")
            rng, rng_ep = jax.random.split(rng)
            state = reset_fn(rng_ep)
            decisions = 0
            episode += 1

        elapsed = time.time() - step_start
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)

print("Final reward:", state.reward)
print("Total accumulated cost:", total_cost)
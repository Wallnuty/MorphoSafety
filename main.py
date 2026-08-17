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
    build_env_for_checkpoint,
    checkpoint_obs_width,
    checkpoint_policy_layers,
    latest_checkpoint,
)
from mjx_safety_gym.envs.go_to_goal import _ROBOT_XMLS, GoToGoal
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
    choices=["goal", "run", "minefield"],
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

_TASKS = {"run": RunForward, "minefield": Minefield, "goal": GoToGoal}
_want = None if _loaded is None else checkpoint_obs_width(_loaded[1]["policy"])
if _args.num_morphologies:
    # Morphology-conditioned checkpoints are NUM_GENES wider than the task obs
    # (47 -> 54 for the ant on minefield), which the width reconciliation in
    # build_env_for_checkpoint cannot produce -- it only flips goal_observation,
    # worth +-3. So build the env directly with conditioning on rather than
    # letting the search fail with "most likely a different robot".
    from mjx_safety_gym import morphology as _morph
    from mjx_safety_gym.algorithms.train_ppo import robot_env_kwargs

    if _args.task == "goal":
        env = GoToGoal(robot=ROBOT, morphology_conditioning=True)
    else:
        env = _TASKS[_args.task](
            robot=ROBOT, morphology_conditioning=True, **robot_env_kwargs(ROBOT)
        )

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
    _mj = env.build_morphology_model(_spec)
    env._mj_model = _mj
    env._mjx_model = mjx.put_model(_mj)
    env._morphology_genes = jax.numpy.asarray(_spec.genes, dtype=jax.numpy.float32)

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
        lambda **kw: _TASKS[_args.task](robot=ROBOT, **kw), ROBOT, _want
    )
    if _default_width is not None:
        print(
            f"Checkpoint expects an observation of width {_want}; this robot's "
            f"defaults give {_default_width}."
        )
        print(
            f"  -> built the env with goal_observation="
            f"{_task_kwargs['goal_observation']}, goal_reward_weight="
            f"{_task_kwargs['goal_reward_weight']} to match it"
        )

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
print(f"Running {num_steps} steps (~{DURATION_SECONDS}s)")

total_cost = 0.0
with mujoco.viewer.launch_passive(m, d) as viewer:
    for i in range(num_steps):
        if not viewer.is_running():
            break
        step_start = time.time()

        if policy_fn is not None:
            # A trained policy is queried every step; ACTION_HOLD only exists
            # to keep *random* actions from looking like jitter.
            rng, rng_action = jax.random.split(rng)
            action = policy_fn(state.obs, rng_action)
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
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)

print("Final reward:", state.reward)
print("Total accumulated cost:", total_cost)
import argparse
import time
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer

import orbax.checkpoint as ocp
from brax.training.acme import running_statistics

from mjx_safety_gym import jax_cache
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.algorithms.train_ppo import latest_checkpoint
from mjx_safety_gym.envs.go_to_goal import GoToGoal
import mjx_safety_gym.lidar as lidar

_parser = argparse.ArgumentParser(
    description="Replay the newest trained policy for a robot in the viewer, "
    "or drive it with random actions if nothing has been trained yet."
)
_parser.add_argument("--robot", choices=["point", "ant"], default="point")
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
_args = _parser.parse_args()

DURATION_SECONDS = _args.duration
ACTION_HOLD = 10  # resample a random action every N steps for smoother motion
ROBOT = _args.robot
DETERMINISTIC = _args.deterministic

# Compile once, and every later run loads the cached kernel from disk
jax_cache.configure()

# Create environment
env = GoToGoal(robot=ROBOT)
rng = jax.random.PRNGKey(0)

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


def load_policy(robot: str, obs, action_size: int):
    """Build an action fn from the newest checkpoint, or None if untrained.

    Rebuilds the network from the env's own shapes rather than from a saved
    config: the training code writes an empty ConfigDict, so brax's
    `checkpoint.load_policy` helper cannot reconstruct the network, and its
    vanilla PPONetworks has no cost-value head anyway.
    """
    ckpt = latest_checkpoint(robot)
    if ckpt is None:
        return None, None
    # Restore through orbax directly, NOT brax's checkpoint.load: that helper
    # builds restore_args with a blanket tree_map over the metadata, which
    # blows up on the optimizer-state subtree we also save
    # ("different types at key path ... list vs RestoreArgs").
    loaded = ocp.PyTreeCheckpointer().restore(str(ckpt))
    # Saved layout is (normalizer, SafePPONetworkParams, penalizer, optimizer);
    # orbax hands each dataclass back as a plain dict of its fields.
    normalizer = running_statistics.RunningStatisticsState(**loaded[0])
    policy_params, value_params = loaded[1]["policy"], loaded[1]["value"]
    # normalize_observations defaults to False in ppo.train, so the
    # preprocessor is the identity -- must match training or the obs scale
    # the policy sees is wrong.
    network = ppo_networks.make_ppo_networks(
        obs.shape,
        action_size,
        preprocess_observations_fn=lambda x, _: x,
    )
    policy = ppo_networks.make_inference_fn(network)(
        (normalizer, policy_params, value_params), deterministic=DETERMINISTIC
    )
    return jax.jit(lambda o, k: policy(o, k)[0]), ckpt

policy_fn, ckpt_path = load_policy(ROBOT, state.obs, env.action_size)
if policy_fn is None:
    print(f"No checkpoint for '{ROBOT}' -- driving with random actions.")
else:
    mode = "deterministic" if DETERMINISTIC else "stochastic"
    print(f"Loaded {mode} policy from {ckpt_path}")

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
        lidar_vals = np.asarray(
            state.obs[: 3 * lidar.NUM_LIDAR_BINS]
        ).reshape(3, lidar.NUM_LIDAR_BINS)
        lidar.update_lidar_rings(lidar_vals, m)
        mjx.get_data_into(d, m, state.data)
        mujoco.mj_forward(m, d)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)

print("Final reward:", state.reward)
print("Total accumulated cost:", total_cost)
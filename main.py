import time
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer

from mjx_safety_gym.envs.go_to_goal import GoToGoal
import mjx_safety_gym.lidar as lidar

DURATION_SECONDS = 10.0
ACTION_HOLD = 10  # resample a random action every N steps for smoother motion
ROBOT = "ant"  # "point" or "ant"

# Compile once, and every later run loads the cached kernel from disk
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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

        if i % ACTION_HOLD == 0:
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
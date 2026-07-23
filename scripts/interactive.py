import time
from typing import Sequence
from absl import app
import jax
from jax import numpy as jp
import mujoco as mj
from mujoco import mjx
import mujoco.viewer
from pynput import keyboard

from mjx_safety_gym.envs.go_to_goal import GoToGoal
import mjx_safety_gym.lidar as lidar

# import mujoco as mj

# Check if installation was succesful.
try:
    print("Checking that the installation succeeded:")
    import mujoco as mj

    mj.MjModel.from_xml_string("<mujoco/>")
except Exception as e:
    raise e from RuntimeError(
        "Something went wrong during installation. Check the shell output above "
        "for more information.\n"
        "If using a hosted Colab runtime, make sure you enable GPU acceleration "
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )


#  Global state for interactive viewer
VIEWERGLOBAL_STATE = {
    "reset": False,  # Flag to indicate if the simulation should be reset
    "ctrl": [0.0, 0.0],  # The control values for the robot
}


# Pynput key press handler
def on_press(key):
    try:
        if key == keyboard.Key.up:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.5
        if key == keyboard.Key.down:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = -0.5
        if key == keyboard.Key.left:  # '2' key to toggle control 2
            VIEWERGLOBAL_STATE["ctrl"][1] = 1.0
        elif key == keyboard.Key.right:
            VIEWERGLOBAL_STATE["ctrl"][1] = -1.0
    except AttributeError:
        # Handle special keys like 'space'
        pass


def on_release(key):
    try:
        if key == keyboard.Key.up:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.0
        if key == keyboard.Key.down:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.0
        if key == keyboard.Key.left:  # '2' key to toggle control 2
            VIEWERGLOBAL_STATE["ctrl"][1] = 0.0
        elif key == keyboard.Key.right:
            VIEWERGLOBAL_STATE["ctrl"][1] = 0.0
        if key == keyboard.Key.enter:
            print("Reset pressed!")
            VIEWERGLOBAL_STATE["reset"] = True
    except AttributeError:
        pass


def setup_key_listener():
    """Setup the pynput keyboard listener in a separate thread."""
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # Start listener in a separate thread


def _main(argv: Sequence[str]) -> None:
    """Launches MuJoCo interactive viewer fed by MJX."""

    # Initialize the environment
    task = GoToGoal()
    rng = jax.random.PRNGKey(0)
    state = task.reset(rng)  # Initial state

    m = task._mj_model
    d = mjx.get_data(m, state.data)

    print("Compiling the step and reset functions")
    start = time.time()
    reset_fn = jax.jit(task.reset).lower(rng).compile()
    step_fn = jax.jit(task.step).lower(state, jp.zeros((2,))).compile()
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")

    # Set up key listener (pynput) in a separate thread
    # NOTE: Have to run this script with sudo privileges on MacOS for this to work
    setup_key_listener()

    t = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            start = time.time()

            t += 1
            if t % 1000 == 0:
                print("Goal dist: ", state.info["last_goal_dist"])
                print("Reward: ", state.reward)
                print("Cost: ", state.info["cost"])
                # print("Mocap: ", state.data.mocap_pos)
                # print("Observations: ", state.obs)
                # print(state.obs.shape)
                pass

            # Step or reset the simulation
            if VIEWERGLOBAL_STATE["reset"]:
                rng, rng_ = jax.random.split(rng)
                state = reset_fn(rng_)
                VIEWERGLOBAL_STATE["reset"] = False
            else:
                ctrl = jp.array(VIEWERGLOBAL_STATE["ctrl"])
                state = step_fn(state, ctrl)

            # Update the lidar rings
            lidar.update_lidar_rings(
                state.obs[: 3 * lidar.NUM_LIDAR_BINS].reshape(3, lidar.NUM_LIDAR_BINS),
                m,
            )

            mjx.get_data_into(d, m, state.data)

            mujoco.mj_forward(m, d)  # N.B. This is needed for mocaps to be visualized

            viewer.sync()

            # Ensure the simulation runs at the correct frequency
            elapsed = time.time() - start
            if elapsed < task._mj_model.opt.timestep:
                time.sleep(task._mj_model.opt.timestep - elapsed)


def main():
    app.run(_main)


if __name__ == "__main__":
    main()

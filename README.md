# mjx-safety-gym
Open-source **MJX implementation of OpenAI Safety Gym** for accelerated safe reinforcement learning.  
Provides lightweight safety environments with **JAX + MuJoCo** that can run both interactively (for visualization and debugging) or fully on GPU (for large-scale RL training).

This codebase is modeled after [DeepMind’s `mujoco_playground`](https://github.com/google-deepmind/mujoco_playground). You can use it in a similar way — for example, by creating a Brax wrapper around the environments and training them directly with Brax.

---

## Installation

This package requires **Python 3.11 or above**.  

You can install it in two ways:

### Option 1 — Local development (from source)  
```bash
# Create and activate a virtual environment with Python ≥3.11
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install mjx-safety-gym in editable mode
pip install -e .
```

### Option 2 - Direct Install from Pypi 
```bash 
pip install mjx-safety-gym
```

## How to Use 
For now, we have only implemented the simple Go-To-Goal environment. 

Most users will want to JIT-compile and vectorize (vmap) the environment’s reset and step functions in their training pipelines, allowing them to scale to thousands of parallel environments on GPU/TPU.  

### Quick Start
Verify your install by creating, resetting, and stepping an environment:

```python
from mjx_safety_gym.envs.go_to_goal import GoToGoal
import jax
from jax import numpy as jp

# Create environment
env = GoToGoal()
rng = jax.random.PRNGKey(0)

# Reset environment
rng, rng_reset = jax.random.split(rng)
state = env.reset(rng_reset)
print("Initial observation shape:", state.obs.shape)

# Step environment once with zero action
action = jp.zeros((2,))
state = env.step(state, action)
print("Next reward:", state.reward)

```

### Interactive Viewer
Alternatively, the repository includes an interactive viewer (scripts/interactive.py) that lets you manually control an agent with keyboard input (the agent is controlled by the arrow keys).

For MacOS, we need special privileges to capture keyboard input and run the interactive viewer 
```bash
sudo mjpython scripts/interactive.py
```

Otherwise, simply run 
```bash
python scripts/interactive.py
```

## Madrona
This repository could work for vision-based observations (included, but untested). For this, we need to install Madrona.

Madrona can be installed on the ETH Zurich cluster as follows: 
```bash
chmod +x vision_setup.bash
./vision_setup.bash
```

Other users can inspect it to see the dependencies required for vision-based support. Setup requires Linux with an NVIDIA GPU and may take several minutes.

## Repository Structure 
```
mjx-safety-gym/
├── mjx_safety_gym
│   ├── __init__.py              # Package entry
│   ├── collision.py             # Collision handling
│   ├── envs/
│   │   ├── go_to_goal.py        # Example environment
│   │   └── xmls/                # MuJoCo XML models
│   │       └── point.xml
│   ├── lidar.py                 # Lidar sensor simulation
│   ├── mjx_env.py               # Core MJX environment wrapper
│   └── world.py                 # World generation
├── scripts/
│   └── interactive.py           # Interactive viewer (keyboard control)
├── vision_setup.bash            # Vision-based setup (ETH Euler cluster specific)
├── pyproject.toml               # Build + metadata
├── LICENSE
└── README.md
```

## References
- [OpenAI Safety Gym](https://github.com/openai/safety-gym) — original benchmark environments for safe reinforcement learning.  
- [MuJoCo XLA (MJX)](https://github.com/google-deepmind/mujoco_mjx) — JAX-accelerated MuJoCo simulator.  
- [DeepMind’s MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — project template that this repository is modeled after.

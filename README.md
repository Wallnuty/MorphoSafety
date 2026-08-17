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

Two tasks and three robots ship here:

| task (`--task`) | reward | notes |
|---|---|---|
| `goal` (`GoToGoal`) | shaped progress toward a randomly placed goal, `+1` per goal reached | the original environment |
| `run` (`RunForward`) | `+x` displacement along an obstacle-strewn corridor | leaving the corridor is charged as **cost**, not walled off |

| robot (`--robot`) | what it is |
|---|---|
| `point` | the original 2-DOF planar puck |
| `ant` | safety-gym's ant: 42 kg, torque/kg 3.5 |
| `ant_gym` | the standard Gym/Brax ant (0.911 kg, torque/kg 164.7) — `ant.xml` at 4x length scale without the ankle density hack |

**`GoToGoal` is not learnable by the ant, and that is measured, not suspected.**
Over 16 seeds, a scripted gait that walks several metres earns a return of
+0.078 ± 0.672 — statistically zero — because the goal sits in a uniformly
random direction, so locomotion by itself is worth nothing and gait and
steering must be discovered simultaneously. `RunForward` exists to remove that
trap by making reward linear in position. See the `run_forward.py` module
docstring for the full derivation.

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

## Testing

**There is no test suite.** It was removed 2026-08-16, deliberately.

The reasoning, from an audit of its own history: in six commits and 81 tests,
it never once found a bug before a human did. Every real defect in this project
was found by running something, measuring something, or watching the ant on
screen -- the `NameError` that made all morphology randomization dead was
committed *with* its tests, and its own message explains why nothing caught it
("nothing exercises the batched-morphology path"). Roughly a third of the tests
cited a specific past incident; all were written after the fact.

The failures that actually cost this project time are not the kind a unit test
catches: an ant that ran 94% of every episode upside down, a cost signal that
was 99% corridor boundary, a reward that was direction-blind, a discount
horizon shorter than the episode. Those are research-design errors, found only
by measuring the thing you assumed.

What the suite did do, twice, was catch *changes* that would have silently
invalidated comparisons. That capability is gone. If a change might invalidate
existing results, that is now on you to notice.

The suite is not lost -- it is in git history and can be restored in full:

```bash
git checkout 3c91dbf -- tests/     # last commit that contained it
pip install "pytest>=8.0"
```

Its docstrings carry the measurement behind each guard and remain the best
written record of several incidents.

## Repository Structure 
```
mjx-safety-gym/
├── mjx_safety_gym
│   ├── collision.py             # Contact lookup used by the cost function
│   ├── lidar.py                 # Lidar sensor simulation
│   ├── mjx_env.py               # Core MJX environment wrapper
│   ├── morphology.py            # Batched morphology models (MjSpec editing)
│   ├── world.py                 # Arena generation (hazards, vases, goal)
│   ├── envs/
│   │   ├── go_to_goal.py        # Navigation task + robot configs
│   │   ├── run_forward.py       # Corridor task (subclasses GoToGoal)
│   │   └── xmls/                # point.xml, ant.xml, ant_gym.xml
│   └── algorithms/
│       ├── train_ppo.py         # CLI entry point, per-robot defaults
│       ├── penalizers.py        # CRPO / Lagrangian
│       ├── wrappers.py          # CostEpisodeWrapper, Saute, morphology
│       └── ppo/                 # Cost-aware PPO (forked from brax)
├── scripts/
│   ├── interactive.py           # Interactive viewer (keyboard control)
│   ├── eval_checkpoint.py       # Paired checkpoint-vs-untrained evaluation
│   ├── evolve.py                # NSGA-II morphology search
│   ├── train_chain.sh           # Auto-resuming training across crashes
│   └── verify_contact_capping.py# Adversarial max_geom_pairs check
├── cluster/                     # SLURM sbatch jobs (Wits mscluster)
├── main.py                      # Replay a checkpoint in the viewer
├── pyproject.toml               # Build + metadata
├── LICENSE
└── README.md
```

## References
- [OpenAI Safety Gym](https://github.com/openai/safety-gym) — original benchmark environments for safe reinforcement learning.  
- [MuJoCo XLA (MJX)](https://github.com/google-deepmind/mujoco_mjx) — JAX-accelerated MuJoCo simulator.  
- [DeepMind’s MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — project template that this repository is modeled after.

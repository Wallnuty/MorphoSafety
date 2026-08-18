#!/bin/bash
# Runs ON the Vast instance. Idempotent -- safe to re-run after a failed sync.
#
# Builds the `morpho` conda env from pyproject.toml's own pins, including the
# [cuda] extra. That extra pulls jax[cuda12]==0.10.2, which bundles its own
# CUDA 12 runtime via nvidia-*-cu12 wheels -- so NOTHING here depends on the
# image's CUDA toolkit, only on the host driver. That is why any reasonably
# modern image works and why there is no `module load` equivalent.
set -euo pipefail

REPO="/root/MorphoSafety"
ENV_NAME="morpho"
cd "$REPO"

echo "=== host driver / GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "nvidia-smi failed -- this box has no usable GPU. Do not proceed."; exit 1; }

echo
echo "=== system packages ==="
# gcc is REQUIRED, not optional. pytorch/pytorch ships no compiler, and the
# dependency tree contains at least one sdist-only package with a C extension
# (evdev), which fails with "error: [Errno 2] No such file or directory: 'gcc'"
# after ~10 minutes of downloading -- late enough to waste real rental time.
#
# Gating this on `command -v tmux` (as it was) is wrong twice over: tmux says
# nothing about whether gcc exists, and once tmux IS installed the whole block
# is skipped, so a re-run after a failure never installs the compiler either.
# Check for the thing actually needed, and check for each separately.
missing=""
for pkg in gcc tmux rsync; do command -v "$pkg" >/dev/null || missing="$missing $pkg"; done
if [ -n "$missing" ]; then
  echo "  installing:$missing"
  apt-get update -qq
  # build-essential rather than bare gcc: python C extensions also want the
  # headers and make, and discovering that one at a time costs another cycle.
  apt-get install -y -qq build-essential tmux rsync </dev/null
else
  echo "  gcc, tmux, rsync all present"
fi
gcc --version | head -1

echo
echo "=== conda ==="
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  CONDA_SH=/opt/conda/etc/profile.d/conda.sh          # pytorch/* images ship this
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "installing miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
# Make every later `conda activate` work without re-sourcing, including the
# tmux session that `vast.sh train` starts.
ln -sf "$CONDA_SH" /opt/conda_profile.sh 2>/dev/null || true

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "creating env ${ENV_NAME} (python 3.11, matching the laptop)..."
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo
echo "=== project + CUDA jax (~3 GB of nvidia wheels, a few minutes) ==="
# [cuda] only. The [dev] extra was pytest and was removed from pyproject.toml
# on 2026-08-16 along with the test suite; pip only WARNS on an unknown extra
# rather than failing, so asking for it here would have gone unnoticed while
# quietly meaning nothing.
pip install -q --upgrade pip
pip install -q -e ".[cuda]"

echo
echo "=== GPU PREFLIGHT ==="
# Asserting on .platform, NOT merely that jax.devices() didn't raise. A missing
# or mismatched CUDA plugin makes jax fall back to CPU SILENTLY -- the exact
# failure that burned 8 hours and produced zero data points on mscluster65
# (see the plan file). Fail loudly here, before any compute is spent.
python - <<'PY'
import sys, jax
devs = jax.devices()
print("jax", jax.__version__, "devices:", devs)
if not devs or devs[0].platform != "gpu":
    sys.exit(f"FATAL: jax is on '{devs[0].platform if devs else 'nothing'}', not gpu. "
             "Driver too old for the CUDA 12 wheels, or no GPU attached.")
import jax.numpy as jp
x = jp.ones((2048, 2048))
print("matmul ok, trace =", float((x @ x).trace()))
print("PREFLIGHT PASSED")
PY

echo
echo "=== skipping the mujoco_menagerie clone ==="
# mujoco_playground's ensure_menagerie_exists() clones a 1.7 GB repo on first
# import, and its ONLY guard is `if not MENAGERIE_PATH.exists()`. So creating
# the directory suppresses it. Measured here: the clone announced an ETA of
# 1h09m on a fresh instance -- longer than everything else in this script
# combined, on a rented-by-the-hour box.
#
# Safe because nothing in this project loads a menagerie model: MENAGERIE_PATH
# is referenced only by playground's own g1 / barkour / leap_hand / panda
# environments, and MorphoSafety ships its own XMLs in mjx_safety_gym/envs/xmls.
# If some future code path did want one, it fails loudly on a missing file
# rather than silently doing the wrong thing.
# MUST NOT `import mujoco_playground` HERE. The import is itself what fires
# ensure_menagerie_exists(), so importing to find the path starts the very
# clone this block exists to prevent -- the makedirs then runs far too late.
# Observed 2026-08-17: this printed "skipping the mujoco_menagerie clone" and
# immediately cloned anyway, twice, at ~3 min and ~26 min of billed time.
# find_spec locates an installed package WITHOUT executing it.
python - <<'PY'
import importlib.util, os
spec = importlib.util.find_spec("mujoco_playground")
if spec is None or not spec.origin:
    raise SystemExit("mujoco_playground not importable -- install failed?")
p = os.path.join(os.path.dirname(spec.origin), "external_deps", "mujoco_menagerie")
os.makedirs(p, exist_ok=True)
print("  menagerie stub at", p)
PY

echo
echo "=== one real mjx step, through the actual env ==="
python - <<'PY'
import jax, time
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms.train_ppo import robot_env_kwargs
# Build through robot_env_kwargs, NOT bare. A bare Minefield(robot="ant") is
# 44 wide; the env training actually uses is 47, because goal_observation adds
# 3 features. Constructing it bare here printed "obs=(44,) (expect 47)" and
# looked like a failure when nothing was wrong. This is the same mismatch that
# main.py and eval_checkpoint.py had to be fixed for.
env = Minefield(robot="ant", **robot_env_kwargs("ant"))
t = time.time(); s = jax.jit(env.reset)(jax.random.PRNGKey(0)); s.obs.block_until_ready()
assert s.obs.shape == (env.observation_size,), (s.obs.shape, env.observation_size)
print(f"reset ok in {time.time()-t:.1f}s  obs={s.obs.shape}  (training width, expect 47)")
step = jax.jit(env.step)
t = time.time(); s = step(s, jax.numpy.zeros(env.action_size)); s.obs.block_until_ready()
print(f"step ok in {time.time()-t:.1f}s")
PY

echo
echo "=== PROVISION COMPLETE ==="
echo "next: bash cluster/vast/vast.sh smoke"

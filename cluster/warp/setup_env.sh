#!/bin/bash
# Build the `mjx-warp` conda env for the MuJoCo Warp backend (2026-10-04).
#
# Run ONCE, on the login node, from inside the warp-probe checkout:
#
#   bash cluster/warp/setup_env.sh
#
# It is a SEPARATE env from mjx-safety-gym on purpose: MuJoCo 3.14 / brax 0.14
# change the physics path and the checkpoint format (see notes/warp_migration.md),
# so the existing env must keep reproducing every earlier run untouched.
# Versions are the exact freeze of the laptop env the migration log was
# measured in (requirements-warp.txt); this checkout is added with --no-deps
# because pyproject.toml still pins the OLD stack (mujoco 3.3.2, forked brax).
set -euo pipefail

WARP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-mjx-warp}"
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env $ENV_NAME already exists -- reusing it (conda env remove -n $ENV_NAME to rebuild)"
else
  conda create -y -q -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
pip install -q -r "$WARP_DIR/cluster/warp/requirements-warp.txt"
pip install -q --no-deps -e "$WARP_DIR"

python - <<'PY'
import importlib, mujoco, warp, jax, brax
import mjx_safety_gym
print("mujoco", mujoco.__version__, "| warp", warp.__version__, "| jax", jax.__version__,
      "| brax", brax.__version__)
print("mjx_safety_gym from", mjx_safety_gym.__file__)
importlib.import_module("mjx_safety_gym.backend")   # only exists on the warp-probe branch
print("OK: env built. GPU is checked by cluster/warp_sps_check.sbatch (login node has none).")
PY

#!/bin/bash
# Build the `mjx-warp` conda env: THE env for this repo since the MuJoCo Warp
# migration (2026-10-04, notes/warp_migration.md).
#
# Run on the login node, from the repo:
#
#   bash cluster/warp/setup_env.sh
#
# Re-running is safe and also re-points the env's editable install at THIS
# checkout -- do that once after the merge if the env was built from the old
# ~/MorphoSafety-warp probe checkout.
#
# `mjx-safety-gym` stays the PRE-WARP env (MuJoCo 3.3.2, forked brax), paired
# with a checkout of the `pre-warp` tag, so every earlier run still reproduces.
# Versions here are the exact freeze of the laptop env the migration log was
# measured in (requirements-warp.txt), a superset of pyproject.toml's pins;
# the checkout is added with --no-deps so pip cannot re-resolve around them.
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
importlib.import_module("mjx_safety_gym.backend")   # absent before the Warp migration
print("OK: env built. GPU is checked by cluster/warp_sps_check.sbatch (login node has none).")
PY

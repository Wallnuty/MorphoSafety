#!/bin/bash
# Minefield, 50M, Saute budget 500, WALLS REPLACED BY A CONTINUOUS COST ZONE.
# 2026-08-23.
#
# The control is checkpoints/ant_minefield_saute500_lidar (the 50M walls arm,
# reward 20.52 / cost 85), which ran the IDENTICAL harness -- same budget, same
# hazard_size, same 512 envs, same 11 evals, same 256x4 policy, same lidar.
# The only change here is the corridor:
#
#     control   --corridor_walls      --boundary_cost_weight 0
#     this run  --no-corridor_walls   --boundary_cost_weight 1.0
#
# INTERPRETATION OF "CONTINUOUS COST ZONE", stated because it was never
# confirmed: this is the pre-2026-08-22 setup -- cost is charged on EVERY step
# the torso sits outside |y| > 1.0. That is continuous in TIME but a step
# function in SPACE (full cost at 1.01 m, identical cost at 5 m). A cost that
# GRADES with distance past the edge would be a change to RunForward.get_cost
# and is not what this run measures.
#
# NO THROUGHPUT MOTIVE. Measured this session at 512 envs on this 3090, over
# two alternated reps each: walls 5316 sps, no walls 5222 sps -- removing the
# walls is 1.8% SLOWER, not faster. The 26.5% from the env-only harness does
# not transfer to training. This is a research change, not a speed change.
#
# WHAT TO WATCH. Under walls the ant cannot leave the corridor, so cost was
# ~100% hazard. Here it can, and the boundary term is back in play -- the
# failure mode to look for is the one from 2026-08-15, where boundary was 99%
# of cost and the policy walked out of the corridor entirely. Cost going UP
# without reward moving is that regression, not a safety result.
set -u
cd "${REPO:-/root/MorphoSafety}" || exit 1
mkdir -p logs checkpoints

source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}" || exit 1
conda activate "${CONDA_ENV:-morpho}" || exit 1

# GPU preflight asserts on .platform, not merely that jax.devices() did not
# raise: a missing CUDA plugin falls back to CPU SILENTLY, and an 8-hour
# cluster job once ran an entire arm that way.
python -c "
import jax; d = jax.devices()
assert d[0].platform == 'gpu', d
print('GPU preflight passed:', d)" || exit 1

NAME="${NAME:-ant_minefield_saute500_lidar_nowalls}"
DIR="$PWD/checkpoints/$NAME"
if [ -d "$DIR" ]; then
  echo "=== REFUSING: $DIR already exists ==="; exit 1
fi

# --checkpoint_logdir MUST be absolute. orbax raises "Checkpoint path should be
# absolute" from inside the save call, which runs at EVAL time -- a relative
# path does not fail at startup, it fails after the training is done. A
# 1.5M-step run was lost to exactly that.
#
# --hazard_lidar is passed explicitly even though it has been the default since
# 2026-08-23, so the log records what actually ran rather than what the
# defaults happened to be on the day.
echo "=== $NAME starting $(date -u +%F\ %H:%M:%S) UTC ==="
python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer saute --safety_budget 500 --saute_terminate \
  --hazard_size 0.16 --hazard_lidar \
  --no-corridor_walls --boundary_cost_weight 1.0 \
  --num_timesteps "${STEPS:-50000000}" --num_envs 512 --num_evals 11 \
  --checkpoint_logdir "$DIR" 2>&1 | tee "logs/$NAME.log"
rc=${PIPESTATUS[0]}
echo "=== $NAME exit=$rc $(date -u +%F\ %H:%M:%S) UTC ==="
echo "=== NOWALLS RUN DONE ==="

#!/bin/bash
# Ant / minefield / Saute 500, WITH per-foot obstacle clearance. 2026-08-24.
#
# THE QUESTION: is the ~85 cost floor a CAPABILITY limit or an OBSERVABILITY
# one? Until today the only obstacle input was the 16-bin lidar ring, computed
# from a SINGLE TORSO POINT, while cost is charged on the minimum over all 13
# collision geoms and -- since --hazard_step_on -- only for the grounded ones.
# The policy was punished for where its FEET landed and shown only where its
# TORSO was. --foot_obstacle_obs closes that: 4 dims per foot (signed clearance
# to the obstacle EDGE, cos/sin bearing, height above the grounded threshold),
# obs 47 -> 63. Entries 0 and 3 are the two halves of the cost condition, so it
# now fires iff both are <= 0 -- fully observable, verified 8/8 on a lift sweep.
#
# CONTROL: logs/lidar_wide.log, cost 84.88 / reward 20.52 at 50M. THIS RUN
# MATCHES IT EXACTLY except for --foot_obstacle_obs. In particular
# --num_envs 512 --num_minibatches 16, NOT the 1024 knee: the control's step
# delta of 5,017,600 divides by 20,480 (512 envs) and not by 40,960, so 512 is
# what it ran, and the budget-50 run already showed how much a second changed
# variable costs in interpretability. Speed is worth less than a clean A/B.
#
# WHAT WOULD FALSIFY THE HYPOTHESIS: cost converging to ~85 again. That would
# say the floor is the ant's gait, not its senses, and the next lever is
# --saute_penalty rather than more inputs.
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

# Fail loudly if the observation is not the width this run is FOR. A silent
# 47 here would produce a perfect-looking control replica and waste the run.
python -c "
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
e = Minefield(**T.robot_env_kwargs('ant'))
assert e.observation_size == 63, f'expected 63, got {e.observation_size}'
print('observation width OK:', e.observation_size)" || exit 1

NAME="${NAME:-ant_minefield_saute500_footobs}"
DIR="$PWD/checkpoints/$NAME"
[ -d "$DIR" ] && { echo "=== REFUSING: $DIR already exists ==="; exit 1; }

# --checkpoint_logdir MUST be absolute: orbax raises "Checkpoint path should be
# absolute" from inside the save call, which runs at EVAL time -- so a relative
# path does not fail at startup, it fails after the training is done.
echo "=== $NAME starting $(date -u +%F\ %H:%M:%S) UTC ==="
python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer saute --safety_budget 500 --saute_terminate \
  --hazard_size 0.16 --hazard_lidar --foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --num_envs 512 --num_minibatches 16 \
  --num_timesteps "${STEPS:-50000000}" --num_evals 11 \
  --checkpoint_logdir "$DIR" 2>&1 | tee "logs/$NAME.log"
rc=${PIPESTATUS[0]}
echo "=== $NAME exit=$rc $(date -u +%F\ %H:%M:%S) UTC ==="
echo "=== FOOTOBS RUN DONE ==="

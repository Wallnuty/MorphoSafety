#!/bin/bash
# Nominal ant, minefield, Saute at BUDGET 50, 1024 envs. 2026-08-24.
#
# THE POINT: at budget 500 the constraint stopped binding at convergence. The
# 256x4+lidar policy settles at cost ~85 unforced, so everything under 500 was
# free and the budget was measuring capability, not a constraint. 50 sits
# BELOW what the policy achieves on its own, so it has to trade reward for it.
#
# WHY 50 IS NOT A REPEAT OF THE FAILED BUDGET-100 RUN (2026-08-22). That run
# had the policy abandon the constraint outright (episode_safe 0.0000 from 15M
# on, cost 611 against a budget of 100). But it ran the OLD 32x4 policy with no
# hazard lidar, which achieves cost ~325 -- so 100 was a 31% target and the
# policy correctly judged it not worth chasing. Today's 256x4+lidar policy
# achieves 85, so 50 is a 59% target. In relative terms this is a LOOSER ask
# than the one that failed, not a tighter one.
#
# --num_minibatches 32 IS LOAD-BEARING, not tuning. validate() requires
# num_envs to divide batch_size * num_minibatches, and 32*16 = 512, so 1024
# envs fails at startup with the default. Scaling minibatches to 32 makes it
# 1024 AND holds updates-per-Mstep at 1562 -- identical to the 512-env runs
# (logs/vast/throughput.log, 2026-08-17), so this stays learning-comparable.
# 1024 is the measured 3090 knee: 5917 -> 7755 sps, +31%.
#
# CORRIDOR: walls, boundary cost 0 -- matching the budget-500 control
# (checkpoints/ant_minefield_saute500_lidar, reward 20.52 / cost 85) so the
# budget is the only deliberate change. It is also the right choice on its own
# merits: with walls, cost is ~100% hazard, so a tight budget measures hazard
# avoidance. Without walls the boundary term re-enters and at a budget of 50
# the ant could exhaust it on lateral drift alone, teaching us nothing.
#
# NOT A ONE-VARIABLE A/B: num_envs also differs from the control (1024 vs
# 512). Comparable in updates-per-env-step, per above, but say so rather than
# implying a clean pair.
#
# FAILURE SIGNATURE TO WATCH: episode_safe pinned at 0 with cost far above 50
# means the policy ignored the constraint again. The lever then is
# --saute_penalty > 0, NOT a bigger budget: with penalty 0 and terminate on,
# exhausting the budget only forfeits the remainder of the episode and is never
# negative, so "sprint and grab reward before it runs out" can beat caution.
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

NAME="${NAME:-ant_minefield_saute50_lidar}"
DIR="$PWD/checkpoints/$NAME"
[ -d "$DIR" ] && { echo "=== REFUSING: $DIR already exists ==="; exit 1; }

# --checkpoint_logdir MUST be absolute. orbax raises "Checkpoint path should be
# absolute" from inside the save call, which runs at EVAL time -- so a relative
# path does not fail at startup, it fails after the training is done. A
# 1.5M-step run was lost to exactly that.
echo "=== $NAME starting $(date -u +%F\ %H:%M:%S) UTC ==="
python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer saute --safety_budget 50 --saute_terminate \
  --hazard_size 0.16 --hazard_lidar \
  --corridor_walls --boundary_cost_weight 0 \
  --num_envs 1024 --num_minibatches 32 \
  --num_timesteps "${STEPS:-50000000}" --num_evals 11 \
  --checkpoint_logdir "$DIR" 2>&1 | tee "logs/$NAME.log"
rc=${PIPESTATUS[0]}
echo "=== $NAME exit=$rc $(date -u +%F\ %H:%M:%S) UTC ==="
echo "=== SAUTE50 RUN DONE ==="

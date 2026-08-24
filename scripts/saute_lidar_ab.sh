#!/bin/bash
# Saute on minefield, WITH vs WITHOUT the hazard lidar ring. 2026-08-23.
#
# The question: once the hazard discs are small enough to thread by foot
# placement (radius 0.16, 0.520 m between disc edges against a 0.212 m limb
# reach), does a long-range obstacle lidar still buy anything -- or is local
# proprioception plus the goal bearing sufficient? Arm B is the only run since
# the ring was removed on 2026-08-22 that has it back.
#
# SEQUENTIAL BY CONSTRUCTION. One GPU/JAX process at a time is this project's
# stability rule, and both arms want the whole card. B starts only when A exits.
#
# THE ONLY DIFFERENCE BETWEEN THE ARMS IS --hazard_lidar. That changes the
# observation width (31 -> 47), so the two checkpoints are not weight-
# compatible with each other -- expected, they are separate runs, not a resume.
#
# BUDGET 500, up from the 100 used on 2026-08-22. At 100 the Saute policy
# abandoned the constraint outright (episode_safe 0.0000 from 15M on, cost 611
# against the budget) while still learning to walk -- 100 was roughly a 10%
# target and the policy correctly judged it not worth chasing. 500 sits near
# what that run actually paid, so it should bind without being hopeless.
#
# --saute_penalty is left at 0, matching the previous run, so the lidar
# comparison is a genuine one-variable A/B. NOTE that penalty 0 is the leading
# suspect for why budget 100 was ignored: with terminate on, exhausting the
# budget costs only the forfeited remainder of the episode, never a negative,
# so "sprint and grab reward before it runs out" beats caution. If BOTH arms
# here again show episode_safe pinned at 0, raise this before blaming 500.
set -u
cd "${REPO:-/root/MorphoSafety}" || exit 1
mkdir -p logs checkpoints

source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}" || exit 1
conda activate "${CONDA_ENV:-morpho}" || exit 1

# GPU PREFLIGHT before committing hours. Asserting on .platform, not merely
# that jax.devices() did not raise: a missing CUDA plugin falls back to CPU
# SILENTLY, and an 8-hour cluster job once ran an entire arm that way.
python -c "
import jax; d = jax.devices()
assert d[0].platform == 'gpu', d
print('GPU preflight passed:', d)" || exit 1

STEPS="${STEPS:-50000000}"
COMMON="--robot ant --task minefield --penalizer saute --safety_budget 500
        --saute_terminate --boundary_cost_weight 0 --hazard_size 0.16
        --num_timesteps $STEPS --num_envs 512 --num_evals 11"

run_arm () {
  local name="$1"; shift
  local dir="$PWD/checkpoints/$name"
  if [ -d "$dir" ]; then
    echo "=== SKIPPING $name -- $dir already exists ==="
    return 0
  fi
  echo "=== ARM $name starting $(date -u +%H:%M:%S) ==="
  # --checkpoint_logdir MUST be absolute. orbax raises "Checkpoint path should
  # be absolute" from inside the save call, which runs at EVAL time -- so a
  # relative path does not fail at startup, it fails after the training is
  # done. A 1.5M-step run was lost to exactly that.
  python -u -m mjx_safety_gym.algorithms.train_ppo $COMMON "$@" \
    --checkpoint_logdir "$dir" 2>&1 | tee "logs/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "=== ARM $name exit=$rc $(date -u +%H:%M:%S) ==="
  return $rc
}

run_arm ant_minefield_saute500_nolidar
run_arm ant_minefield_saute500_lidar --hazard_lidar

echo "=== BOTH ARMS DONE $(date -u +%H:%M:%S) ==="

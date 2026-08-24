#!/bin/bash
# Follow-on to saute_lidar_ab.sh: does a non-zero --saute_penalty stop the
# policy spending its whole budget? 2026-08-23.
#
# WAITS FOR THE A/B CHAIN TO FINISH FIRST. One GPU/JAX process at a time.
# Launched as its own tmux session rather than appended to saute_lidar_ab.sh,
# because bash reads a running script incrementally by byte offset -- editing
# a script while it executes can make it run garbage. Never append to a live
# one.
#
# PENALTY IS A ONE-TIME HIT UNDER --saute_terminate, not a per-step drain.
# Saute.step returns reward=-penalty on the step where the budget goes
# negative, and that same step sets done, so the substitution is paid once and
# the episode is over. (With --no-saute_terminate it WOULD be paid every
# remaining inner step, which is a different scale entirely -- 0.01 x ~1500
# steps ~= 15.) Sized against a full traverse worth ~22 (11 dx + 11 goal-delta
# + 0.5 healthy):
#     5   -- about a quarter of the task, a real but survivable deterrent
#     20  -- roughly the whole task, i.e. exhausting the budget wipes out
#            everything the episode could have earned
#
# ARM A OF THE A/B IS THE PENALTY-0 CONTROL. These use its exact config (no
# lidar, budget 500, hazard 0.16, terminate on) with only --saute_penalty
# changed, so the three are a clean one-variable sweep.
#
# CAVEAT ON READING THE RESULT. The eval-side Saute wrapper is always built
# with terminate=False/penalty=0, so eval `episode_cost` and `episode_safe`
# describe behaviour with the budget NOT enforced. A policy trained to spend
# its budget will therefore always show episode_safe ~0 -- that is close to
# tautological, not proof it is unsafe. What actually distinguishes these arms
# is how far it gets BEFORE exhausting the budget: read episode_reward
# together with episode_saute_unsafe, not episode_safe on its own.
set -u
cd "${REPO:-/root/MorphoSafety}" || exit 1
mkdir -p logs checkpoints

echo "=== waiting for the A/B chain (tmux session 'ab') to finish ==="
while tmux has-session -t ab 2>/dev/null; do sleep 60; done
echo "=== A/B chain done, starting penalty sweep $(date -u +%H:%M:%S) ==="

source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}" || exit 1
conda activate "${CONDA_ENV:-morpho}" || exit 1
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
  if [ -d "$dir" ]; then echo "=== SKIPPING $name (exists) ==="; return 0; fi
  echo "=== ARM $name starting $(date -u +%H:%M:%S) ==="
  python -u -m mjx_safety_gym.algorithms.train_ppo $COMMON "$@" \
    --checkpoint_logdir "$dir" 2>&1 | tee "logs/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "=== ARM $name exit=$rc $(date -u +%H:%M:%S) ==="
  return $rc
}

run_arm ant_minefield_saute500_pen5  --saute_penalty 5.0
run_arm ant_minefield_saute500_pen20 --saute_penalty 20.0

echo "=== PENALTY SWEEP DONE $(date -u +%H:%M:%S) ==="

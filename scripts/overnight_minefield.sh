#!/bin/bash
# Overnight: 50M steps of `ant` then 50M of `ant_gym`, both on --task minefield.
#
# SEQUENTIAL, NOT PARALLEL, AND THAT IS NOT NEGOTIABLE. This project's own
# stability rule is at most one GPU/JAX process at a time: two concurrent CUDA
# contexts in this WSL guest is the documented cause of a full VSCode/WSL hang
# (see the plan file's "Session stability"). The second robot starts only after
# the first chain has exited.
#
# WHY A CHAIN RATHER THAN TWO train_ppo CALLS. This laptop segfaults (exit 139,
# SIGSEGV inside libcuda) at unpredictable points, historically at the third
# eval. scripts/train_chain.sh resumes from the newest checkpoint after each
# crash, so a crash costs only the steps since the last eval instead of the
# night. MAX_GENS gives each robot a finite budget so the first one yields the
# GPU to the second.
#
#   50M steps = 10 generations x 5M, at 2 evals per generation.
#
# EVALS_PER_GEN IS 2 ON PURPOSE. Every laptop segfault recorded in this project
# has struck at the third eval of a run; 2 is the configuration measured to
# survive. It costs resolution -- 2 evals per 5M steps is a coarse learning
# curve -- but the chain is for producing a checkpoint, not for measuring one.
# Real numbers come from scripts/eval_checkpoint.py afterwards.
#
# Stop it with:
#   touch checkpoints/ant_minefield_chain/STOP        (finishes current gen)
#   touch checkpoints/ant_gym_minefield_chain/STOP
# or, immediately:
#   pkill -f overnight_minefield.sh; pkill -f train_ppo
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"

# nohup does NOT inherit the conda environment, and a bare `python` then fails
# with ModuleNotFoundError: No module named 'jax' -- which has bitten this
# project before. Source conda explicitly rather than relying on the caller.
source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 1
conda activate mjx-safety-gym || exit 1

TASK=minefield
STEPS_PER_GEN="${STEPS_PER_GEN:-5000000}"
MAX_GENS="${MAX_GENS:-10}"          # 10 x 5M = 50M steps per robot
EVALS_PER_GEN="${EVALS_PER_GEN:-2}"
SUMMARY="$ROOT/logs/overnight_minefield.log"
mkdir -p "$ROOT/logs"

echo "=== OVERNIGHT MINEFIELD $(date) ===" | tee -a "$SUMMARY"
echo "    $MAX_GENS x $STEPS_PER_GEN steps per robot, $EVALS_PER_GEN evals/gen" | tee -a "$SUMMARY"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$SUMMARY"

for ROBOT in ant ant_gym; do
  CHAIN="$ROOT/checkpoints/${ROBOT}_minefield_chain"
  LOG="$ROOT/logs/${ROBOT}_minefield_chain.log"

  {
    echo
    echo "############################################################"
    echo "### ROBOT $ROBOT  task=$TASK  $(date)"
    echo "### chain : $CHAIN"
    echo "### log   : $LOG"
    echo "############################################################"
  } | tee -a "$SUMMARY"

  # A leftover STOP from a previous night would silently skip this robot
  # entirely, and the only symptom would be an empty checkpoint directory.
  rm -f "$CHAIN/STOP"

  start=$(date +%s)
  ROBOT="$ROBOT" TASK="$TASK" MAX_GENS="$MAX_GENS" \
    bash "$ROOT/scripts/train_chain.sh" \
      "$CHAIN" "" "$LOG" "$STEPS_PER_GEN" "$EVALS_PER_GEN"
  status=$?
  elapsed=$(( $(date +%s) - start ))

  gens=$(find "$CHAIN" -mindepth 1 -maxdepth 1 -type d -name 'gen*' 2>/dev/null | wc -l)
  ckpts=$(find "$CHAIN" -mindepth 2 -maxdepth 2 -type d -regex '.*/[0-9]+' 2>/dev/null | wc -l)
  echo "### $ROBOT DONE exit=$status after $((elapsed / 60)) min; $gens generations, $ckpts checkpoints" \
    | tee -a "$SUMMARY"

  # Deliberately does NOT abort the loop on failure. If ant dies from something
  # robot-specific, ant_gym is still worth the remaining hours; if it dies from
  # something shared, ant_gym failing the same way is itself the diagnosis. The
  # per-robot exit status is recorded above either way.
done

echo "=== ALL DONE $(date) ===" | tee -a "$SUMMARY"

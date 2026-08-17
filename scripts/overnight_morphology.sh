#!/bin/bash
# Overnight: does ONE gene-conditioned policy make EIGHT different ant bodies
# walk the minefield corridor?
#
# This is the central hypothesis of the whole morphology-optimization plan. If
# it holds, evolutionary search costs ~5 training-run equivalents instead of
# ~1000 (one PPO run per candidate). If it fails, the plan needs rethinking
# before any GA work is worth doing.
#
# WHAT CHANGED TO MAKE THIS RUNNABLE AT ALL (2026-08-16). Morphology
# randomization had never been run on a corridor task, and could not have been:
# `randomization_fn` rebuilt every body with GOTOGOAL's arena hardcoded, so
# --task minefield would have stepped nq=85 physics inside an nq=15 env. nbody
# and ngeom coincidentally match (38/35), so it would not reliably have
# crashed -- just silently produced garbage. Now the ENV supplies the model
# builder (GoToGoal.build_morphology_model), so each task gets its own arena.
#
# ACTUATORS ARE RESCALED PER BODY, and that is load-bearing. With the XML's
# fixed gear=150, measured gear/gravity across sampled morphologies ran 0.94 to
# 3.82 -- and below 1.0 a hip cannot hold its own leg up, i.e. that body CANNOT
# WALK whatever the policy does. Training across such a population would
# measure the actuator mismatch and look like "conditioning does not work".
# rescale_actuators holds the ratio at the nominal 1.71 for every body.
#
# LANE 0 IS THE NOMINAL ANT, deliberately (include_nominal). Uniform gene
# sampling would never produce it, and it is the one body with a specialist
# baseline to compare against: checkpoints/ant_minefield_chain reached
# episode_reward 22.1 of a ~22.5 ceiling on exactly this task.
#
# WHAT THE TRAINING LOG CANNOT TELL YOU. `episode_reward` is averaged over all
# 8 bodies, so a mean of 11 is equally consistent with "all eight at 11" and
# "four at 22, four at 0". It also cannot detect the failure mode that LOOKS
# like success: a policy that learned one robust gait and ignores the genes
# entirely. Both need scripts/eval_morphology.py afterwards -- per-body returns
# and a gene-ablation control. Do not declare victory from this log.
#
# Stop it with:
#   touch checkpoints/ant_morph_chain/STOP     (finishes current generation)
# or, immediately:
#   pkill -f overnight_morphology.sh; pkill -f train_ppo
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"

source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 1
conda activate mjx-safety-gym || exit 1

ROBOT=ant
TASK=minefield
CHAIN="$ROOT/checkpoints/ant_morph_chain"
LOG="$ROOT/logs/ant_morph_chain.log"
SUMMARY="$ROOT/logs/overnight_morphology.log"

STEPS_PER_GEN="${STEPS_PER_GEN:-5000000}"
MAX_GENS="${MAX_GENS:-10}"           # 10 x 5M = 50M steps
EVALS_PER_GEN="${EVALS_PER_GEN:-2}"  # 3rd eval is where this laptop segfaults

# num_envs 512 is the measured knee for a single body on this GPU and needs no
# change to batch_size/num_minibatches (validate() requires num_envs to divide
# batch_size*num_minibatches = 32*16 = 512). 512/8 = 64 envs per body.
#
# policy 256x4, NOT the (32,)*4 default. The value and cost-value nets are
# already (256,)*5, so the policy was 8x narrower -- fine for one fixed body,
# but this one has to produce a DIFFERENT gait per body from a 7-dim gene
# vector. train_ppo's own help text names it as the first thing to widen if
# per-morphology returns come out identical. Measured cost: 2445 sps against
# 3117 for the unconditioned single-body run at the same 512 envs (~22%).
NUM_MORPH="${NUM_MORPH:-8}"
EXTRA_ARGS="--num_morphologies $NUM_MORPH --num_envs 512 \
--policy_hidden_layer_sizes 256 256 256 256"

mkdir -p "$ROOT/logs"
{
  echo "=== OVERNIGHT MORPHOLOGY $(date) ==="
  echo "    $MAX_GENS x $STEPS_PER_GEN steps, $NUM_MORPH bodies (lane 0 = nominal)"
  echo "    extra: $EXTRA_ARGS"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
} | tee -a "$SUMMARY"

rm -f "$CHAIN/STOP"    # a leftover STOP would silently skip the whole run

start=$(date +%s)
ROBOT="$ROBOT" TASK="$TASK" MAX_GENS="$MAX_GENS" EXTRA_ARGS="$EXTRA_ARGS" \
  bash "$ROOT/scripts/train_chain.sh" \
    "$CHAIN" "" "$LOG" "$STEPS_PER_GEN" "$EVALS_PER_GEN"
status=$?
elapsed=$(( $(date +%s) - start ))

gens=$(find "$CHAIN" -mindepth 1 -maxdepth 1 -type d -name 'gen*' 2>/dev/null | wc -l)
ckpts=$(find "$CHAIN" -mindepth 2 -maxdepth 2 -type d -regex '.*/[0-9]+' 2>/dev/null | wc -l)
{
  echo "### DONE exit=$status after $((elapsed / 60)) min; $gens generations, $ckpts checkpoints"
  echo "### next: python scripts/eval_morphology.py --checkpoint <newest>"
  echo "=== ALL DONE $(date) ==="
} | tee -a "$SUMMARY"

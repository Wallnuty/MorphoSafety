#!/bin/bash
# Continue the B3 policy at BUDGET 10 for 100M more steps.
#
#   sbatch cluster/ant_minefield_b3_budget10.sbatch
#
# B3 is the 2026-09-18 result (notes/Overnight_2026-09-17.md): PPO-Lagrangian
# with the per-foot hazard grid, warm-started from an unconstrained grid walker,
# 150M steps in total (50M unconstrained + 100M at budget 25). It finished at
# cost 23.6 per crossing (budget 25) with the traverse intact, multiplier 1.33
# and falling -- the constraint satisfied by foot placement. This run asks for
# the next factor: budget 10, i.e. ~2.4x fewer mine contacts per crossing
# (about 25 hazard-steps against B3's 64; a careless walker is ~120).
#
# THE CHECKPOINT IS NOT ON THE CLUSTER. It was trained on Vast and pulled to
# the laptop. Copy it up first, from the laptop:
#
#   scp -r checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M \
#       <login-node>:~/MorphoSafety/checkpoints/vast/
#
# The preflight below aborts if it is missing, before any GPU time is spent.
#
# Everything else is the standard script: this file only sets the SAFE_* knobs
# and runs cluster/ant_minefield_safe_b50.sbatch's body (its own #SBATCH lines
# are comments there; the directives that count are these). The multiplier is
# RESTORED with the checkpoint (penalizer params are part of the 4-element
# restore), so pressure starts at 1.33 rather than 0.01 and rises from there
# because 23.6 > 10 from the first batch. flip_cost 50 stays: with budget 10 a
# single flip is five budgets, which is the point.
#
# READ, as always: cost/m and hz_steps with displacement held near 10 m. If
# displacement falls while cost falls, that is the old stop-early exit --
# expected to be closed by the flip cost, but budget 10 is the tightest we have
# asked for, so watch for it. Multiplier climbing past ~3 with cost/m flat is
# the 2026-09-12 windup signature.
#
# Sizing: 100M at ~6000 sps = 4.6 h, plus 21 evals at ~95 s = 0.6 h; 12 h is
# 2x margin.
#SBATCH --job-name=ant-b3-b10
#SBATCH --output=ant_safe_b50_%j.log
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster45,mscluster83
#SBATCH --time=12:00:00
# --output MUST stay ant_safe_b50_%j.log: the diagnostics printer at the end of
# the main script reads the log back by that name.

export SAFE_RESUME="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M/000050380800"
export SAFE_NAME="ant_minefield_ppo_lagrangian_b10_grid5_flip50_warm_250M"
export SAFE_PENALIZER=ppo_lagrangian
export SAFE_BUDGET=10
export SAFE_MULT_LR=1.5e-5
export SAFE_FLIP_COST=50
export SAFE_GRID=5
export SAFE_STEPS=100000000
export SAFE_NUM_EVALS=21           # every 5M, same cadence as the 50M runs

if [ ! -d "$SAFE_RESUME" ]; then
  echo "ABORT: $SAFE_RESUME is not on this machine."
  echo "From the laptop:  scp -r checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M <login-node>:~/MorphoSafety/checkpoints/vast/"
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash cluster/ant_minefield_safe_b50.sbatch

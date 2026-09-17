#!/bin/bash
# Continue the B3 policy at the SAME budget (25) for 100M more steps.
#
#   sbatch cluster/ant_minefield_b3_budget25.sbatch
#
# B3 is the 2026-09-18 result (notes/Overnight_2026-09-17.md): PPO-Lagrangian
# with the per-foot hazard grid, warm-started from an unconstrained grid walker,
# 150M steps in total (50M unconstrained + 100M at budget 25). It finished at
# cost 23.6 per crossing (budget 25) with the traverse intact, multiplier 1.33
# and falling -- the constraint satisfied by foot placement. This run keeps the
# budget at 25 and simply trains on: the control question for the budget-10
# arm (cluster/ant_minefield_b3_budget10.sbatch). B3's cost was still drifting
# down at the end (21.1 -> 21.4 -> 23.6 over its last three evals is noise
# around ~22), so it answers (a) where the multiplier settles once the
# constraint stays satisfied -- it should keep falling toward 0 and cost should
# hover at the budget, not below it -- and (b) how much of any improvement the
# budget-10 arm shows is just more training. Same steps, same evals, same
# restore; only SAFE_BUDGET differs.
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
# and moves with the sign of (25 - cost) from the first batch. flip_cost 50
# stays, as in every 2026-09-17 arm.
#
# READ: the multiplier column. A Lagrangian at equilibrium sits with cost ~=
# budget and a small, steady multiplier; if lambda decays to ~0 and cost drifts
# back UP toward the unconstrained ~50, the constraint is doing all the work
# and the placement is not "learned" in any lasting sense -- worth knowing
# before co-design leans on it.
#
# Sizing: 100M at ~6000 sps = 4.6 h, plus 21 evals at ~95 s = 0.6 h; 12 h is
# 2x margin.
#SBATCH --job-name=ant-b3-b25
#SBATCH --output=ant_safe_b50_%j.log
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster45,mscluster83
#SBATCH --time=12:00:00
# --output MUST stay ant_safe_b50_%j.log: the diagnostics printer at the end of
# the main script reads the log back by that name.

export SAFE_RESUME="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M/000050380800"
export SAFE_NAME="ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_250M"
export SAFE_PENALIZER=ppo_lagrangian
export SAFE_BUDGET=25
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

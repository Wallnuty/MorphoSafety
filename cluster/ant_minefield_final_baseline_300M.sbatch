#!/bin/bash
# FINAL-RECIPE BASELINE (2026-09-27): ONE nominal ant, from scratch, 300M steps.
# The fixed-body reference the co-design run is compared against -- same recipe,
# same numerics, no morphology search.
#
#   PPO-Lagrangian, budget 25 (episode total, adaptive horizon), multiplier
#   lr 1.5e-5 / cap 3.0 / init 0.01; per-foot 7x7 hazard grid + torso lidar;
#   linear penetration cost at the 5 cm gate; flip cost 50; finish line;
#   NO WALLS -- leaving the corridor ends the episode and costs 50; solver caps
#   4/8; float32 matmuls; policy 256x4; 1024 envs / 32 minibatches; seed 0.
#
# Every value is pinned below even though each is also the current default, so
# a later default change cannot silently alter this run.
#
# A thin wrapper: exports SAFE_* and hands over to ant_minefield_safe_b50.sbatch,
# which carries the GPU/code preflights (they abort in seconds on a stale
# checkout), the training command and the summary printer. Only THIS file's
# #SBATCH lines are read -- the wrapped script's are inert under `exec bash`.
#
# BEFORE SUBMITTING: the walls/exit/solver/metrics changes of 2026-09-26/27 must
# be committed, pushed, and pulled on the cluster:
#
#   git pull && sbatch cluster/ant_minefield_final_baseline_300M.sbatch
#
# SIZING. 3090 measured 36-40k training sps at this config (Vast, 2026-09-27;
# host CPU is worth ~9%, and mscluster's are older), so 300M is ~2.5 h of
# training, plus ~10 min of compile and 31 evals x 320 episodes. --time 8h is
# ~2.5x margin. Checkpoints: one per eval, 31 x ~8 MB.
#
# WHAT TO READ (printed as a table at the end; columns per the reporting rule):
#   eval/episode_reward, eval/episode_speed (m/s), eval/episode_cost,
#   eval/episode_hazard_steps, eval/episode_flipped, eval/episode_out_of_bounds,
#   training/lagrange_multiplier (settled 1.3-1.9 in earlier arms; pinned at
#   the 3.0 cap = budget infeasible).
#
# IF SOMETHING LOOKS BROKEN (NaNs, feet sinking, odd contacts): suspect the 4/8
# solver caps first -- resubmit with SAFE_SOLVER_IT=10 SAFE_SOLVER_LS=10
# (identical physics to the old 100/50, ~30% slower) and a new SAFE_NAME.
#
#SBATCH --job-name=ant-final-300M
# Must stay ant_safe_b50_%j.log: the wrapped script's summary printer reads
# exactly that file.
#SBATCH --output=ant_safe_b50_%j.log
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster45,mscluster83
#SBATCH --time=08:00:00

export SAFE_NAME="${SAFE_NAME:-ant_minefield_final_b25_300M}"
export SAFE_STEPS=300000000
export SAFE_NUM_EVALS=31            # every 10M
export SAFE_PENALIZER=ppo_lagrangian
export SAFE_BUDGET=25
export SAFE_MULT_LR=1.5e-5
export SAFE_MULT_MAX=3.0
export SAFE_MULT_INIT=0.01
export SAFE_GRID=7
export SAFE_RINGS=0
export SAFE_FOOT_OBS=0
export SAFE_LIDAR=1
export SAFE_COST_SHAPE=linear
export SAFE_FLIP_COST=50
export SAFE_EXIT_COST=50
export SAFE_WALLS=0
export SAFE_FINISH_LINE=1
export SAFE_SHAPING_W=0
export SAFE_WIDTH=256
export SAFE_PRECISION=highest
export SAFE_SOLVER_IT="${SAFE_SOLVER_IT:-4}"
export SAFE_SOLVER_LS="${SAFE_SOLVER_LS:-8}"
unset SAFE_RESUME SAFE_EPS          # from scratch; eps = the 5 cm default

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash cluster/ant_minefield_safe_b50.sbatch

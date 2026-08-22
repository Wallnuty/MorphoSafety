#!/bin/bash
# Constrained (CRPO) vs unconstrained ant on minefield, matched in every other
# flag. Run A first, then B, SEQUENTIALLY -- this project's stability rule is
# one GPU/JAX process at a time (two concurrent CUDA contexts in WSL is the
# documented cause of a full VSCode/WSL hang).
#
# WHY B EXISTS. checkpoints/ant_minefield_chain (50M, 2026-08-16) is NOT a valid
# control for this: it predates terminate_on_goal becoming the default AND the
# corridor walls, so any reward/cost gap against it confounds the constraint
# with two environment changes. B is the same 30M under the same env.
#
# 30M, not 50M: the unconstrained minefield run's learning was over by ~25M, and
# with terminate_on_goal the 30M co-design run plateaued by 17M.
#
# BUDGET 250 is measured, not guessed. scripts/eval_checkpoint.py on the 50M
# unconstrained policy WITH walls: hazard cost 452.5 over 286 decisions =
# 1.58/decision. --safety_budget is a rate threshold of X/625 per decision, so
# 250 targets 25% of the unconstrained hazard rate.
set -u
cd /home/samsn/MorphologyResearch/MorphoSafety || exit 1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# ACTIVATE THE ENV EXPLICITLY. train_chain.sh invokes a bare `python`, so under
# nohup (no interactive profile, no activated env) it resolves to the system
# interpreter and dies instantly with "No module named 'jax'" -- which the chain
# correctly treats as an unexpected exit and stops on, so the whole A/B finished
# in under a second having trained nothing.
source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 1
conda activate mjx-safety-gym || exit 1
python -c "import jax; assert jax.devices()[0].platform == 'gpu', jax.devices()" \
  || { echo "FATAL: jax is not on the GPU; refusing to start"; exit 1; }

COMMON="--boundary_cost_weight 0"   # walls make the corridor physical; cost is pure hazard

run () {  # name, extra flags
  local name="$1"; shift
  echo "=== $(date '+%F %T')  START $name ===" >> logs/safe_rl_ab.log
  TASK=minefield ROBOT=ant MAX_GENS=6 EXTRA_ARGS="$COMMON $*" \
    bash scripts/train_chain.sh \
      "$PWD/checkpoints/ant_minefield_$name" \
      "" \
      "$PWD/logs/ant_minefield_$name.log" \
      5000000 2
  local st=$?   # capture BEFORE the echo, or $? is the echo's own status
  echo "=== $(date '+%F %T')  END $name (exit $st) ===" >> logs/safe_rl_ab.log
}

mkdir -p logs
run crpo --penalizer crpo --safety_budget 250
run unconstrained --penalizer none
echo "=== $(date '+%F %T')  BOTH DONE ===" >> logs/safe_rl_ab.log

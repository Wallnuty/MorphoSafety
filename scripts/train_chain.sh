#!/bin/bash
# Keep training across the laptop's segfaults, resuming from the newest
# checkpoint each time.
#
# WHY THIS EXISTS. This laptop segfaults (exit 139, SIGSEGV in libcuda) at
# unpredictable points -- historically at the 3rd eval, which is exactly where
# the last run died after reaching 8.52M cumulative steps. Checkpoints are
# written every eval and restore correctly, so a crash costs only the steps
# since the last eval. This loop turns "crashed, needs a human" into "resumed
# automatically".
#
# EACH GENERATION WRITES TO ITS OWN DIRECTORY. A resumed run restarts its step
# counter at 0 and reuses the same zero-padded directory names, so writing back
# into the directory being restored FROM would overwrite it mid-run -- and if
# the run then crashed there would be no clean checkpoint to fall back to.
# gen1, gen2, ... keeps every generation intact.
#
# Cumulative steps are NOT what the log prints. Each generation's step counter
# restarts at 0; the true total is the sum across generations. progress() below
# prints both.
#
# Stop it with:  touch <chain dir>/STOP     (finishes the current generation)
# or:            pkill -f train_chain.sh; pkill -f train_ppo    (immediate)
set -u

CHAIN="${1:-$PWD/checkpoints/ant_gym_upright_chain}"
SEED_CKPT="${2:-}"
LOG="${3:-$PWD/ant_gym_upright_chain.log}"
STEPS_PER_GEN="${4:-5000000}"
EVALS_PER_GEN="${5:-4}"

mkdir -p "$CHAIN"

newest_ckpt () {
  # newest numeric checkpoint dir across all generations, by mtime
  find "$CHAIN" -mindepth 2 -maxdepth 2 -type d -regex '.*/[0-9]+' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

gen=0
while true; do
  if [ -e "$CHAIN/STOP" ]; then
    echo "=== STOP file present, exiting chain at gen $gen ===" | tee -a "$LOG"
    break
  fi

  gen=$((gen + 1))
  outdir="$CHAIN/gen$gen"
  [ -d "$outdir" ] && continue   # already used, skip to a fresh number

  restore="$(newest_ckpt)"
  [ -z "$restore" ] && restore="$SEED_CKPT"

  {
    echo
    echo "############################################################"
    echo "### GEN $gen  $(date)"
    echo "### restore : ${restore:-<none, fresh start>}"
    echo "### write   : $outdir"
    echo "############################################################"
  } | tee -a "$LOG"

  args=(--robot ant_gym --task run --penalizer none
        --num_timesteps "$STEPS_PER_GEN" --num_evals "$EVALS_PER_GEN"
        --num_eval_envs 8 --num_eval_episodes 2
        --checkpoint_logdir "$outdir")
  [ -n "$restore" ] && args+=(--restore_checkpoint_path "$restore")

  python -u -m mjx_safety_gym.algorithms.train_ppo "${args[@]}" >> "$LOG" 2>&1
  status=$?
  echo "=== GEN $gen EXIT=$status $(date) ===" | tee -a "$LOG"

  # 139 = SIGSEGV, the known laptop failure -- resume. 0 = generation finished
  # its step budget, also resume to keep going. Anything else is a real error
  # (bad flag, OOM, missing checkpoint) and looping on it would spin forever.
  if [ "$status" -ne 0 ] && [ "$status" -ne 139 ]; then
    echo "=== unexpected exit $status, stopping chain ===" | tee -a "$LOG"
    break
  fi

  # If a generation dies before writing ANY checkpoint, the next one would
  # restore from the same place and die the same way. Bail rather than spin.
  if [ -z "$(ls -A "$outdir" 2>/dev/null)" ]; then
    echo "=== gen $gen wrote no checkpoint, stopping chain ===" | tee -a "$LOG"
    break
  fi
  sleep 5
done

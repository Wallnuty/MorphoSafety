#!/bin/bash
# One-command status for the overnight minefield chains. Read-only.
#
# Exists because the run outlives any given Claude session: the chain is a
# detached nohup process and everything it produces is on disk, so the state
# should be readable without reconstructing it from a conversation.
#
#   bash scripts/overnight_status.sh
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== alive? ==="
pgrep -af "overnight_minefield|train_chain|train_ppo" | cut -c1-110 || echo "NOTHING RUNNING"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

for ROBOT in ant ant_gym; do
  LOG="logs/${ROBOT}_minefield_chain.log"
  CHAIN="checkpoints/${ROBOT}_minefield_chain"
  [ -f "$LOG" ] || continue

  echo
  echo "=== $ROBOT ==="
  # Generations completed, and how each one ended. 139 = the known laptop
  # SIGSEGV, which the chain resumes from; anything else stopped the chain.
  grep -E "^### GEN|^=== GEN .* EXIT|reached MAX_GENS|stopping chain" "$LOG" | tail -n 8

  # Every eval, one per line. episode_reward on this task is net +x
  # displacement plus goal-distance progress plus the upright bonus;
  # avg_episode_length is the uprightness curve (episodes end on a flip).
  echo "--- evals (step / reward / cost / ep_len) ---"
  grep -o "step=[0-9]* .*" "$LOG" \
    | sed -E 's/.*step=([0-9]+).*episode_cost=([0-9.-]+).*episode_reward=([0-9.-]+).*avg_episode_length=([0-9.]+).*/step=\1  reward=\3  cost=\2  len=\4/' \
    | tail -n 12

  # training/sps only appears on the LAST eval of a generation at
  # --num_evals 2, so this is the only real throughput number available.
  echo "--- training/sps ---"
  grep -o "training/sps=[0-9.]*" "$LOG" | tail -n 5 || echo "(none yet -- gen1 still running)"

  ckpts=$(find "$CHAIN" -mindepth 2 -maxdepth 2 -type d -regex '.*/[0-9]+' 2>/dev/null | wc -l)
  echo "--- $ckpts checkpoints on disk ---"
  find "$CHAIN" -mindepth 2 -maxdepth 2 -type d -regex '.*/[0-9]+' 2>/dev/null | sort | tail -n 3
done

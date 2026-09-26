#!/bin/bash
# Laptop chain (2026-09-20): continue the B3 policy at budget 5, then at
# budget 0, 100M steps each, POINT GOAL (--no-finish_line) so the two arms
# compare directly with the budget-10 and budget-25 continuations that ran on
# the cluster. Sequential -- one GPU/JAX process at a time on this machine.
#
#   nohup bash scripts/b3_budget_chain_laptop.sh > logs/b3_budget_chain.log 2>&1 &
#   tail -f logs/b3_budget_chain.log
#
# Same recipe as the cluster arms except the batch layout, which is the
# laptop's 512/16 (identical learning density to 1024/32: updates per env-step
# cancel). Multiplier is RESTORED from B3 (1.33) and rises from there, because
# both budgets are violated from the first batch. BUDGET 0 IS A LIMIT CASE: the
# constraint 0 - cost can never be satisfied, so the multiplier ratchets for the
# whole run -- it is "minimise cost, ever harder", and the thing to watch is
# whether speed survives it.
#
# CRASH-RESILIENT: this laptop has segfaulted mid-run before. If a run dies
# short of its final checkpoint it is resumed from its newest checkpoint into a
# sibling directory (a resumed trainer restarts its step counter, so resuming
# INTO the same directory would overwrite the early checkpoints), up to 3 times.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mjx-safety-gym
mkdir -p "$ROOT/logs"

SRC="$ROOT/checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M/000050380800"
STEPS=100000000
FINAL_STEP=000100761600     # 100M at 512/16: 20,480 env steps per training step, rounded

[ -d "$SRC" ] || { echo "ABORT: B3 checkpoint missing: $SRC"; exit 1; }
# Match real python processes only: a bare pattern also matches the shell that
# launched this script (its command line contains the pattern text).
if pgrep -f "^python.* -m mjx_safety_gym" >/dev/null; then
  echo "ABORT: a JAX/MJX process is already running here:"; pgrep -af "^python.* -m mjx_safety_gym"; exit 1
fi
echo "=== GPU preflight ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python -c "
import jax; d = jax.devices(); print('JAX devices:', d)
assert [x for x in d if x.platform == 'gpu'], 'ABORT: no GPU -- jax fell back to CPU silently.'
" || { echo "=== GPU preflight FAILED ==="; exit 1; }
echo "=== code / observation preflight ==="
JAX_PLATFORMS=cpu python -c "
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
kw = dict(T.robot_env_kwargs('ant')); kw.update(foot_obstacle_obs=False, foot_hazard_grid=5, finish_line=False)
e = Minefield(**kw)
assert e.observation_size == 147, e.observation_size
assert abs(e._ground_contact_eps - 0.05) < 1e-9 and e._hazard_cost_shape == 'linear' and not e._finish_line
print('PASS: obs 147, linear cost @5cm, point goal')
" || { echo "=== code preflight FAILED ==="; exit 1; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

run_arm () {  # run_arm <budget> [resume dir] [steps] [run name]
  local B="$1"
  local NAME="${4:-ant_minefield_ppo_lagrangian_b${B}_grid5_flip50_warm_250M}"
  local CKPT="$ROOT/checkpoints/$NAME"
  local LOG="$ROOT/logs/$NAME.log"
  local STEPS="${3:-$STEPS}"
  local RESUME="${2:-$SRC}" STEPS_LEFT="$STEPS" OUT="$CKPT" attempt=0
  while :; do
    attempt=$((attempt + 1))
    echo; echo "=========================================================="
    echo " budget=$B  attempt $attempt  $STEPS_LEFT steps  from $RESUME"
    echo "   -> $OUT   log $LOG"; echo "   $(date -u)"
    echo "=========================================================="
    [ -d "$OUT" ] && { echo "ABORT: $OUT exists; refusing to mix two runs"; return 1; }
    time python -u -m mjx_safety_gym.algorithms.train_ppo \
      --robot ant --task minefield \
      --penalizer ppo_lagrangian --safety_budget "$B" --adaptive_budget_horizon \
      --lagrangian_multiplier_lr 1.5e-5 --lagrangian_multiplier_max 3.0 \
      --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs --foot_hazard_grid 5 \
      --hazard_cost_shape linear --flip_cost 50 --no-finish_line \
      --matmul_precision default \
      --corridor_walls --boundary_cost_weight 0 \
      --policy_hidden_layer_sizes 256 256 256 256 \
      --num_envs 512 --num_minibatches 16 \
      --num_timesteps "$STEPS_LEFT" --num_evals 11 \
      --restore_checkpoint_path "$RESUME" \
      --checkpoint_logdir "$OUT" 2>&1 | tee -a "$LOG"
    local status=${PIPESTATUS[0]}
    echo "=== budget=$B attempt $attempt exit status: $status ($(date -u)) ==="
    local newest; newest="$(ls -1d "$OUT"/0* 2>/dev/null | sort | tail -1)"
    if [ "$status" -eq 0 ] && [ -n "$newest" ]; then echo "budget=$B DONE: $newest"; return 0; fi
    [ -n "$newest" ] || { echo "budget=$B died before its first checkpoint; giving up"; return 1; }
    [ "$attempt" -ge 3 ] && { echo "budget=$B: 3 attempts, giving up at $newest"; return 1; }
    local reached=$((10#$(basename "$newest")))
    STEPS_LEFT=$((STEPS - reached)); RESUME="$newest"; OUT="${CKPT}_r${attempt}"
    echo "resuming from $reached with $STEPS_LEFT steps left -> $OUT"
    sleep 30
  done
}

# 2026-09-21: the budget-5 arm was stopped at 70M; finish its last 30M from
# that checkpoint (multiplier restored at 2.72, now capped at 3.0), then the
# budget-0 arm, 100M from B3 (multiplier restored at 1.33 -- the reasonable
# start; it reaches the cap in ~20M at budget 0 and stays there).
run_arm 5 "$ROOT/checkpoints/ant_minefield_ppo_lagrangian_b5_grid5_flip50_warm_250M/000070103040" 30000000 \
          ant_minefield_ppo_lagrangian_b5_grid5_flip50_warm_250M_r1
run_arm 0

echo; echo "--- diagnostics, both arms ---"
for B in 5 0; do
  for LOGF in "$ROOT"/logs/ant_minefield_ppo_lagrangian_b${B}_grid5_flip50_warm_250M*.log; do
    [ -f "$LOGF" ] || continue
    echo "== budget $B"
    LOGFILE="$LOGF" BUDGET="$( [ "$B" = 0 ] && echo 1 || echo "$B")" python - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["LOGFILE"]); B = float(os.environ["BUDGET"])
print(f"{'step':>12} {'reward':>7} {'speed':>7} {'displ':>7} {'cost':>7} {'cost/m':>6} {'hz_steps':>8} {'ep_len':>7} {'lambda':>7}")
for line in p.read_text().splitlines():
    if not line.startswith("step="): continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.eE+-]+)", line))
    try: c = float(g["eval/episode_cost"]); l = float(g["eval/avg_episode_length"]); r = float(g["eval/episode_reward"])
    except (KeyError, ValueError): continue
    displ = float(g["eval/episode_dx"]) if "eval/episode_dx" in g else (r - 0.0002 * l) / 2
    speed = float(g["eval/episode_speed"]) if "eval/episode_speed" in g else displ / (0.02 * l)
    per_m = f"{c/displ:>6.1f}" if displ > 0.5 else f"{'--':>6}"
    print(f"{int(float(g['step'])):>12,} {r:>7.2f} {speed:>6.2f}m/s {displ:>6.2f}m {c:>7.1f} {per_m} "
          f"{g.get('eval/episode_hazard_steps','--'):>8.8} {l:>7.0f} {g.get('training/lagrange_multiplier','--'):>7.7}")
PY
  done
done
echo "=== chain done $(date -u) ==="

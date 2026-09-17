#!/bin/bash
# Runs ON the Vast instance. The Vast twin of cluster/ant_minefield_safe_b50.sbatch
# -- same SAFE_* variables, same preflights, same training command, same
# diagnostics printer -- minus SLURM. READ THAT FILE'S HEADER for the
# experimental plan and what every knob means; only what differs on a rented
# box is documented here.
#
#   SAFE_PENALIZER=none bash cluster/vast/vast.sh safe                # launch
#   VAST_TAG=b bash cluster/vast/vast.sh safe-log                     # follow
#
# QUEUEING ON ONE BOX: one GPU/JAX process at a time is this project's
# standing rule. `SAFE_AFTER=<tmux session>` blocks until that session is gone
# (the previous run's printer included), then starts -- the pattern
# remote_lagrangian_b30.sh used to queue behind the CRPO arm.
#
# A RESUME PATH IS REMOTE. `SAFE_RESUME` names a checkpoint step dir on the
# instance (/root/MorphoSafety/checkpoints/<name>/<step>), e.g. the previous
# run on the same box; for a checkpoint from elsewhere, `vast.sh push-ckpt`
# it first.
set -uo pipefail
cd /root/MorphoSafety || exit 1
source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

if [ -n "${SAFE_AFTER:-}" ]; then
  if tmux has-session -t "$SAFE_AFTER" 2>/dev/null; then
    echo "=== waiting for tmux session '$SAFE_AFTER' to finish ==="
    while tmux has-session -t "$SAFE_AFTER" 2>/dev/null; do sleep 60; done
    echo "=== '$SAFE_AFTER' done at $(date -u +%H:%M:%S) ==="
    sleep 30   # let the GPU release its context
  fi
fi

echo "=== box ==="
hostname; date -u
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo

STEPS="${SAFE_STEPS:-50000000}"
BUDGET="${SAFE_BUDGET:-50}"
PENALIZER="${SAFE_PENALIZER:-ppo_lagrangian}"
MULT_LR="${SAFE_MULT_LR:-1e-5}"
RESUME="${SAFE_RESUME:-}"
KAPPA_MAX="${SAFE_KAPPA_MAX:-5.0}"
RAMP_FRAC="${SAFE_RAMP_FRAC:-0.6}"
SHAPING_W="${SAFE_SHAPING_W:-0}"
SHAPING_R="${SAFE_SHAPING_R:-0.16}"
FOOT_OBS="${SAFE_FOOT_OBS:-0}"
GRID="${SAFE_GRID:-0}"
COST_SHAPE="${SAFE_COST_SHAPE:-linear}"
FLIP_COST="${SAFE_FLIP_COST:-0}"
NUM_EVALS="${SAFE_NUM_EVALS:-11}"
if [ "$FOOT_OBS" = "1" ]; then FOOT_FLAG=--foot_obstacle_obs; OBS=63; else FOOT_FLAG=--no-foot_obstacle_obs; OBS=47; fi
OBS=$((OBS + 4 * GRID * GRID))
TAG=""
[ "$SHAPING_W" != "0" ] && TAG="${TAG}_shaped_w${SHAPING_W}"
[ "$FOOT_OBS" = "1" ] && TAG="${TAG}_feet"
[ "$GRID" != "0" ] && TAG="${TAG}_grid${GRID}"
[ "$COST_SHAPE" != "linear" ] && TAG="${TAG}_${COST_SHAPE}"
[ "$FLIP_COST" != "0" ] && TAG="${TAG}_flip${FLIP_COST}"
if [ "$PENALIZER" = "none" ]; then
  NAME="${SAFE_NAME:-ant_minefield_none_wide_ctrl${TAG}}"
else
  NAME="${SAFE_NAME:-ant_minefield_${PENALIZER}_b${BUDGET}_adaptive${TAG}}"
fi
[ -n "$RESUME" ] && [ -z "${SAFE_NAME:-}" ] && NAME="${NAME}_r$(date -u +%m%d%H%M)"
CKPT="/root/MorphoSafety/checkpoints/$NAME"
LOG="/root/MorphoSafety/logs/vast_${NAME}.log"
mkdir -p /root/MorphoSafety/logs

if [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { echo "ABORT: SAFE_RESUME=$RESUME is not a directory"; exit 1; }
  echo "=== RESUMING from $RESUME ==="
  set -- --restore_checkpoint_path "$RESUME"
else
  set --
fi

echo "=== GPU preflight ==="
python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, 'ABORT: no GPU -- jax fell back to CPU silently.'
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, nothing spent ==="; exit 1; }

echo "=== code / observation preflight ==="
FOOT_OBS="$FOOT_OBS" OBS="$OBS" GRID="$GRID" python -c "
import inspect, os
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
from mjx_safety_gym.algorithms.ppo import train as ppo_train, losses as ppo_losses
dests = {a.dest for a in T.build_argparser()._actions}
for f in ('adaptive_budget_horizon', 'start_y_jitter', 'hazard_shaping_weight',
          'hazard_footprint', 'hazard_cost_shape', 'flip_cost', 'foot_hazard_grid'):
    assert f in dests, f'ABORT: no --{f} in this checkout. Sync the code.'
assert 'budget_decision_steps' in inspect.signature(ppo_losses.make_losses).parameters
kw0 = dict(T.robot_env_kwargs('ant')); kw0['foot_obstacle_obs'] = False
e0 = Minefield(**kw0)
assert abs(e0._ground_contact_eps - 0.05) < 1e-9 and e0._hazard_footprint == 'contact' \
    and e0._hazard_cost_shape == 'linear', 'ABORT: cost semantics are not the 2026-09-17 ones'
kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = os.environ['FOOT_OBS'] == '1'
kw['foot_hazard_grid'] = int(os.environ['GRID'])
w = Minefield(**kw).observation_size; want = int(os.environ['OBS'])
assert w == want, f'ABORT: expected obs {want}, got {w}'
print(f'PASS: obs {w}; 2026-09-17 cost semantics present end to end')
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
echo "=========================================================="
echo " $PENALIZER  budget=$BUDGET (EPISODE TOTAL, adaptive horizon)  $STEPS steps"
echo "   -> $CKPT"
echo "   log: $LOG"
[ "$PENALIZER" = "ppo_lagrangian" ] && echo "   multiplier_lr=$MULT_LR"
[ "$SHAPING_W" != "0" ] && echo "   REWARD SHAPING: w=$SHAPING_W radius $SHAPING_R"
echo "   cost: $COST_SHAPE penetration, grounded within 5 cm, flip_cost=$FLIP_COST"
echo "   observation: $OBS wide ($FOOT_FLAG, grid $GRID)"
echo "=========================================================="
echo

time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer "$PENALIZER" --safety_budget "$BUDGET" \
  --adaptive_budget_horizon \
  --lagrangian_multiplier_lr "$MULT_LR" \
  --penalty_kappa_max "$KAPPA_MAX" --penalty_ramp_frac "$RAMP_FRAC" \
  --hazard_size 0.16 --hazard_lidar "$FOOT_FLAG" --foot_hazard_grid "$GRID" \
  --hazard_cost_shape "$COST_SHAPE" --flip_cost "$FLIP_COST" \
  --hazard_shaping_weight "$SHAPING_W" --hazard_shaping_radius "$SHAPING_R" \
  --corridor_walls --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes 256 256 256 256 \
  --num_envs 1024 --num_minibatches 32 \
  --num_timesteps "$STEPS" --num_evals "$NUM_EVALS" \
  --checkpoint_logdir "$CKPT" "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo
echo "=== exit status: $status ==="
ls -1 "$CKPT" 2>&1 | tail -3
du -sh "$CKPT" 2>&1
echo
echo "--- the diagnostics, together ---"
LOGFILE="$LOG" BUDGET="$BUDGET" python - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["LOGFILE"])
B = float(os.environ["BUDGET"])
print(f"{'step':>12} {'reward':>7} {'displ':>7} {'cost':>7} {'vs B':>6} {'cost/m':>6} "
      f"{'hz_steps':>8} {'shaping':>8} {'ep_len':>7} {'mean_dec':>9} {'mult/active':>12}")
for line in p.read_text().splitlines():
    if not line.startswith("step="):
        continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.eE+-]+)", line))
    try:
        c = float(g["eval/episode_cost"]); l = float(g["eval/avg_episode_length"])
        r = float(g["eval/episode_reward"])
    except (KeyError, ValueError):
        continue
    knob = (g.get("training/lagrange_multiplier") or g.get("training/scheduled/kappa")
            or g.get("training/crpo/active", "--"))
    displ = (r - 0.0002 * l) / 2
    per_m = f"{c/displ:>6.1f}" if displ > 0.5 else f"{'--':>6}"
    print(f"{int(float(g['step'])):>12,} {r:>7.2f} {displ:>6.2f}m {c:>7.1f} "
          f"{c/B:>5.1f}x {per_m} {g.get('eval/episode_hazard_steps','--'):>8.8} "
          f"{g.get('eval/episode_hazard_shaping','--'):>8.8} {l:>7.0f} "
          f"{g.get('training/budget_mean_episode_decisions','--'):>9} {knob:>12}")
PY
echo "=== done $(date -u) ==="

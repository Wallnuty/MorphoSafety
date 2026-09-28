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
BUDGET="${SAFE_BUDGET:-25}"
PENALIZER="${SAFE_PENALIZER:-ppo_lagrangian}"
MULT_LR="${SAFE_MULT_LR:-1.5e-5}"
# 2026-09-21: anti-windup cap and a warm-start initial value for the multiplier.
# MULT_MAX 3.0 = above where every settled arm sat (1.3-1.9), below the 4-5 that
# crushed gaits. MULT_INIT 0.01 is the from-scratch value; for an arm resumed
# from an UNCONSTRAINED walker start it near equilibrium (~1.5) and skip the
# ~30M ramp. (A resume from a Lagrangian checkpoint restores its multiplier
# and ignores this.)
MULT_MAX="${SAFE_MULT_MAX:-3.0}"
MULT_INIT="${SAFE_MULT_INIT:-0.01}"
RESUME="${SAFE_RESUME:-}"
KAPPA_MAX="${SAFE_KAPPA_MAX:-5.0}"
RAMP_FRAC="${SAFE_RAMP_FRAC:-0.6}"
SHAPING_W="${SAFE_SHAPING_W:-0}"
SHAPING_R="${SAFE_SHAPING_R:-0.16}"
FOOT_OBS="${SAFE_FOOT_OBS:-0}"
GRID="${SAFE_GRID:-7}"
COST_SHAPE="${SAFE_COST_SHAPE:-linear}"
FLIP_COST="${SAFE_FLIP_COST:-50}"
# 2026-09-21 A/B knobs. WIDTH: policy layer width (4 layers). EPS: grounding
# gate in metres ("" = the 0.05 default). RINGS: per-foot ring map (0 = off).
# FINISH_LINE: 1 = green line (the default task since 2026-09-20), 0 = the
# point goal every pre-09-20 arm used -- use 0 for comparisons against them.
WIDTH="${SAFE_WIDTH:-256}"
EPS="${SAFE_EPS:-}"
RINGS="${SAFE_RINGS:-0}"
FINISH_LINE="${SAFE_FINISH_LINE:-1}"
LIDAR="${SAFE_LIDAR:-1}"
# highest = float32 matmuls (default since 2026-09-26); default = TF32, what
# every earlier arm used -- set it to continue or reproduce one of those.
PRECISION="${SAFE_PRECISION:-highest}"
# 2026-09-26: walls OFF (they cost 27% of throughput); leaving the corridor
# ends the episode and costs EXIT_COST instead. SAFE_WALLS=1 reproduces the
# 2026-08-22..09-25 arms.
WALLS="${SAFE_WALLS:-0}"
EXIT_COST="${SAFE_EXIT_COST:-50}"
if [ "$WALLS" = "1" ]; then WALLS_FLAG=--corridor_walls; else WALLS_FLAG=--no-corridor_walls; fi
# Solver caps: empty = the robot default (ant 4/8 since 2026-09-27); 100/50
# reproduces earlier arms.
SOLVER_IT="${SAFE_SOLVER_IT:-}"; SOLVER_LS="${SAFE_SOLVER_LS:-}"
SOLVER_FLAGS=(); [ -n "$SOLVER_IT" ] && SOLVER_FLAGS+=(--solver_iterations "$SOLVER_IT")
[ -n "$SOLVER_LS" ] && SOLVER_FLAGS+=(--solver_ls_iterations "$SOLVER_LS")
if [ "$FINISH_LINE" = "1" ]; then FINISH_FLAG=--finish_line; else FINISH_FLAG=--no-finish_line; fi
if [ "$LIDAR" = "1" ]; then LIDAR_FLAG=--hazard_lidar; else LIDAR_FLAG=--no-hazard_lidar; fi
EPS_FLAG=(); [ -n "$EPS" ] && EPS_FLAG=(--ground_contact_eps "$EPS")
NUM_EVALS="${SAFE_NUM_EVALS:-11}"
if [ "$FOOT_OBS" = "1" ]; then FOOT_FLAG=--foot_obstacle_obs; else FOOT_FLAG=--no-foot_obstacle_obs; fi
TAG=""
[ "$SHAPING_W" != "0" ] && TAG="${TAG}_shaped_w${SHAPING_W}"
[ "$FOOT_OBS" = "1" ] && TAG="${TAG}_feet"
[ "$GRID" != "0" ] && TAG="${TAG}_grid${GRID}"
[ "$RINGS" != "0" ] && TAG="${TAG}_rings${RINGS}"
[ "$LIDAR" != "1" ] && TAG="${TAG}_nolidar"
[ "$WIDTH" != "256" ] && TAG="${TAG}_w${WIDTH}"
[ -n "$EPS" ] && TAG="${TAG}_eps${EPS}"
[ "$FINISH_LINE" != "1" ] && TAG="${TAG}_pointgoal"
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
FOOT_OBS="$FOOT_OBS" GRID="$GRID" RINGS="$RINGS" LIDAR="$LIDAR" PRECISION="$PRECISION" python -c "
import inspect, os
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
from mjx_safety_gym.algorithms.ppo import train as ppo_train, losses as ppo_losses
dests = {a.dest for a in T.build_argparser()._actions}
for f in ('adaptive_budget_horizon', 'start_y_jitter', 'hazard_shaping_weight',
          'hazard_footprint', 'hazard_cost_shape', 'flip_cost', 'foot_hazard_grid'):
    assert f in dests, f'ABORT: no --{f} in this checkout. Sync the code.'
for f in ('matmul_precision', 'corridor_walls', 'terminate_out_of_bounds', 'exit_cost',
          'solver_iterations', 'solver_ls_iterations'):
    assert f in dests, f'ABORT: no --{f} in this checkout (added 2026-09-26/27). Sync the code.'
from mjx_safety_gym.algorithms.wrappers import CostEpisodeWrapper
assert {'flipped', 'out_of_bounds'} <= set(CostEpisodeWrapper._SUMMED_INFO_KEYS), (
    'ABORT: no flipped / out-of-bounds eval metrics in this checkout (2026-09-27). Sync the code.')
if os.environ.get('PRECISION', 'highest') == 'highest':
    import jax, numpy as np, jax.numpy as jp
    from mjx_safety_gym.numerics import configure_matmul_precision
    configure_matmul_precision('highest')
    a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    ref = a.astype(np.float64) @ a.astype(np.float64)
    err = float(np.abs(np.asarray(jax.jit(jp.matmul)(a, a), np.float64) - ref).max() / np.abs(ref).max())
    assert err < 1e-5, f'ABORT: highest precision is not float32 on this GPU (rel err {err:.1e}).'
    print(f'PASS: float32 matmuls (rel err {err:.1e})')
assert 'budget_decision_steps' in inspect.signature(ppo_losses.make_losses).parameters
kw0 = dict(T.robot_env_kwargs('ant')); kw0['foot_obstacle_obs'] = False
e0 = Minefield(**kw0)
assert abs(e0._ground_contact_eps - 0.02) < 1e-9 and e0._hazard_footprint == 'contact' \
    and e0._hazard_cost_shape == 'linear', (
    'ABORT: default cost semantics are not 0.02 / contact / linear (2 cm gate '
    'since 2026-09-28). Sync the code.')
kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = os.environ['FOOT_OBS'] == '1'
kw['foot_hazard_grid'] = int(os.environ['GRID']); kw['foot_hazard_rings'] = int(os.environ['RINGS'])
if os.environ['LIDAR'] != '1': kw['lidar_groups'] = ()
w = Minefield(**kw).observation_size
print(f'PASS: obs {w}; 2026-09-17 cost semantics present end to end')
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
echo "=========================================================="
echo " $PENALIZER  budget=$BUDGET (EPISODE TOTAL, adaptive horizon)  $STEPS steps"
echo "   -> $CKPT"
echo "   log: $LOG"
[ "$PENALIZER" = "ppo_lagrangian" ] && echo "   multiplier_lr=$MULT_LR  cap=$MULT_MAX  init=$MULT_INIT"
[ "$SHAPING_W" != "0" ] && echo "   REWARD SHAPING: w=$SHAPING_W radius $SHAPING_R"
echo "   cost: $COST_SHAPE penetration, grounded within ${EPS:-0.02 (default)} m, flip_cost=$FLIP_COST"
echo "   physics: walls=$WALLS (exit ends episode, exit_cost=$EXIT_COST)  solver=${SOLVER_IT:-robot default}/${SOLVER_LS:-robot default}  matmul=$PRECISION"
echo "   observation: $FOOT_FLAG grid $GRID rings $RINGS lidar $LIDAR | policy ${WIDTH}x4 | $FINISH_FLAG${EPS:+ | eps $EPS}"
echo "=========================================================="
echo

time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer "$PENALIZER" --safety_budget "$BUDGET" \
  --adaptive_budget_horizon \
  --lagrangian_multiplier_lr "$MULT_LR" --lagrangian_multiplier_max "$MULT_MAX" \
  --initial_lagrange_multiplier "$MULT_INIT" \
  --penalty_kappa_max "$KAPPA_MAX" --penalty_ramp_frac "$RAMP_FRAC" \
  --hazard_size 0.16 "$LIDAR_FLAG" "$FOOT_FLAG" --foot_hazard_grid "$GRID" --foot_hazard_rings "$RINGS" \
  "$FINISH_FLAG" "${EPS_FLAG[@]}" \
  --hazard_cost_shape "$COST_SHAPE" --flip_cost "$FLIP_COST" \
  --matmul_precision "$PRECISION" "${SOLVER_FLAGS[@]}" \
  --hazard_shaping_weight "$SHAPING_W" --hazard_shaping_radius "$SHAPING_R" \
  "$WALLS_FLAG" --terminate_out_of_bounds --exit_cost "$EXIT_COST" --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes "$WIDTH" "$WIDTH" "$WIDTH" "$WIDTH" \
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
# speed = eval/episode_speed (m/s, per-episode displacement over duration,
# exact). displ is derived from reward (see below) for runs logged before
# 2026-09-18; when eval/episode_dx is present it is used instead.
print(f"{'step':>12} {'reward':>7} {'speed':>7} {'displ':>7} {'cost':>7} {'vs B':>6} {'cost/m':>6} "
      f"{'hz_steps':>8} {'flip%':>6} {'oob%':>5} {'ep_len':>7} {'mean_dec':>9} {'mult/active':>12}")
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
    displ = float(g["eval/episode_dx"]) if "eval/episode_dx" in g else (r - 0.0002 * l) / 2
    # Fraction of eval episodes that went over / left the corridor (logged
    # since 2026-09-27; '--' for older runs).
    pct = lambda k: (f"{100 * float(g['eval/episode_' + k]):.0f}%" if 'eval/episode_' + k in g else "--")
    per_m = f"{c/displ:>6.1f}" if displ > 0.5 else f"{'--':>6}"
    speed = f"{float(g['eval/episode_speed']):>6.2f}" if "eval/episode_speed" in g else f"{displ/(0.02*l):>6.2f}"
    vs_b = f"{c/B:>5.1f}x" if B > 0 else f"{'--':>6}"  # budget 0 has no ratio
    print(f"{int(float(g['step'])):>12,} {r:>7.2f} {speed}m/s {displ:>6.2f}m {c:>7.1f} "
          f"{vs_b} {per_m} {g.get('eval/episode_hazard_steps','--'):>8.8} {pct('flipped'):>6} {pct('out_of_bounds'):>5} {l:>7.0f} "
          f"{g.get('training/budget_mean_episode_decisions','--'):>9} {knob:>12}")
PY
echo "=== done $(date -u) ==="

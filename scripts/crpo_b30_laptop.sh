#!/bin/bash
# ant / minefield / CRPO at --safety_budget 30, 256x4 policy, lidar-only.
# THIS LAPTOP, not a rented box. 50M steps, ~4 h.
#
#   bash scripts/crpo_b30_laptop.sh
#
# ===================== READ THIS BEFORE READING THE LOG ===================
#
# THE TOTALS LIE. CRPO HAS A KNOWN COLLAPSE MODE ON THIS EXACT TASK and it
# looks like success in every headline metric. Measured 2026-08-22 at
# --safety_budget 100, 25M steps (checkpoints/vast/ant_minefield_crpo100/):
#
#     step    reward   cost   safe   ep_len   cost/step   crpo/active
#     0        -0.00   50.7   0.88     2035     0.0249      --
#     10.0M     0.90  137.1   0.44     2230     0.0615      0.58
#     25.1M     0.65   36.9   0.97      738     0.0500      0.28
#
# episode_cost ended BELOW untrained and episode_safe reached 0.97, while
# **cost per step DOUBLED**. The entire improvement was episode SHORTENING,
# 2035 -> 738 inner steps. `constraint = safety_budget - vsc.mean()` where vsc
# is discounted FUTURE cost, and `terminate_on_flip` zeroes future cost, so
# falling over is the cheapest cost reduction available. The ant lunged ~25 cm
# and tipped, never reaching the first mine row 1.25 m ahead -- it got WORSE at
# the thing it was being constrained about, and the constraint paid it.
#
# SO THE DIAGNOSTIC IS THE RATE, NOT THE TOTAL:
#
#     cost/step = eval/episode_cost / eval/avg_episode_length
#
#   * cost/step FALLING with ep_len roughly held  -> real hazard avoidance.
#   * cost/step RISING while episode_cost falls   -> the flip exploit. The run
#                                                    is void as a safety result
#                                                    no matter how good
#                                                    episode_safe looks.
#
# `crpo/active` MEANS THE OPPOSITE OF WHAT IT LOOKS LIKE. penalizers.py does
# `actor_loss = where(active, actor_loss, -loss_constraint)`, so active=TRUE is
# the step where the NORMAL REWARD LOSS RUNS. active=0.28 means 72% of gradient
# steps had no reward term at all. Read it as "fraction of steps still
# optimising reward", never as "fraction violating".
#
# ========================= BUDGET 30 IS AGGRESSIVE ========================
#
# MEASURED, not guessed -- an earlier version of this header called budget 30
# "infeasible from step 0 by ~2.5x" and that was WRONG. The step-0 eval of the
# 2026-08-25 attempt, this exact config, reads:
#
#     step=0  episode_cost=14.08  reward=0.53  safe=0.81  ep_len=2425
#
# **An untrained ant costs 14, well INSIDE a budget of 30.** The ~77 figure is
# the TRAINED cost: a policy that actually traverses the corridor passes many
# more mines than one flailing near spawn, so cost RISES with competence here.
#
# That makes 30 a well-posed constraint rather than a hopeless one: slack at
# init, binding only once the ant starts crossing the field. It is the same
# shape as the budget-100 run that collapsed (also feasible at init, 50.7 on
# the older arena), so the collapse mode above remains the thing to watch --
# but the reason to watch is the exploit, not infeasibility.
#
# The env-side fix the 2026-08-22 analysis proposed -- charge cost on flip, or
# drop terminate_on_flip for this task -- is NOT applied here. It would change
# the task and break comparability with every other minefield run. Applying it
# is the obvious follow-up if this collapses.
#
# ============================ LAPTOP SIZING ===============================
#
# 512 envs / 16 minibatches, NOT the 1024/32 the rented 3090s use. The laptop
# sweep (2026-08-16, ant/minefield) measured the knee here and it is 512:
#
#     envs   corr_sps   vs prev
#      256      3593       -
#      512      4747     1.32x    <- knee; largest size needing no batch change
#     1024      4900     1.03x    (and HALVES updates per env-step unless
#     2048      4735     0.97x     num_minibatches moves with it)
#
# 512/16 and 1024/32 are LEARNING-IDENTICAL -- num_minibatches cancels out of
# updates-per-env-step -- so this differs from the rented runs only in speed.
# Expect below 4747 regardless: that number was measured with brax's (32,)*4
# policy, before 256x4 became the default on 2026-08-23.
#
# CRASH HISTORY. This laptop's RTX 4050 / WSL2 passthrough has a documented
# exit-139 inside libcuda.so.1.1, historically striking at the THIRD eval of a
# run. It has not recurred on minefield at 512 envs (the 2026-08-16 overnight
# did 100M across 20 generations clean), but --num_evals 11 means checkpoints
# every 5M, so a crash costs at most 5M steps and the run resumes with
# --restore_checkpoint_path.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mjx-safety-gym

# CRPO_-PREFIXED, and the prefix is not decoration. Reading a bare `NAME`
# inherits THIS MACHINE's environment -- WSL exports NAME=SamLaptop -- so the
# first launch wrote to checkpoints/SamLaptop and logs/SamLaptop.log. The same
# bug had already been found and fixed in cluster/vast/vast.sh earlier the same
# day and was reintroduced here verbatim. Generic names (NAME, STEPS, USER,
# HOST) are exactly the ones a shell already has set, and an inherited value is
# indistinguishable from an intended one. Override as e.g.
#     CRPO_BUDGET=50 bash scripts/crpo_b30_laptop.sh
STEPS="${CRPO_STEPS:-50000000}"
BUDGET="${CRPO_BUDGET:-30}"
NAME="${CRPO_NAME:-ant_minefield_crpo_b30_wide}"
CKPT="$ROOT/checkpoints/$NAME"       # ABSOLUTE: orbax rejects relative paths,
                                     # and it raises at EVAL time, not startup
LOG="$ROOT/logs/$NAME.log"
mkdir -p "$ROOT/logs"

[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }

# ONE GPU/JAX PROCESS AT A TIME. Two concurrent CUDA contexts in this WSL2 VM
# once hung the whole VSCode session; this is the project's standing rule.
if pgrep -f "train_ppo|python.*mjx_safety_gym" | grep -qv "^$$\$"; then
  echo "ABORT: a JAX/MJX process is already running here:"
  pgrep -af "train_ppo|python.*mjx_safety_gym"
  exit 1
fi

echo "=== GPU preflight ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, 'ABORT: no GPU -- jax fell back to CPU silently.'
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, no compute spent ==="; exit 1; }

echo "=== observation preflight ==="
JAX_PLATFORMS=cpu python -c "
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = False
w = Minefield(**kw).observation_size
assert w == 47, f'expected 47 (lidar only), got {w}'
print(f'PASS: obs {w}, lidar-only')
" || { echo "=== observation preflight FAILED, no compute spent ==="; exit 1; }

# 0.9 of 6 GB. Nothing else may touch this GPU while it runs (see the rule
# above), so the fraction can be high; a diagnostic script sharing the card
# would need this lowered, which is the whole reason the rule exists.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo
echo "=========================================================="
echo " CRPO  budget=$BUDGET  policy 256x4  lidar-only  $STEPS steps"
echo "   out $CKPT"
echo "   log $LOG"
echo "   READ cost/step = episode_cost / avg_episode_length, NOT episode_cost"
echo "=========================================================="
echo

# python -u is NOT optional: Python block-buffers stdout when redirected and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer crpo --safety_budget "$BUDGET" \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes 256 256 256 256 \
  --num_envs 512 --num_minibatches 16 \
  --num_timesteps "$STEPS" --num_evals 11 \
  --checkpoint_logdir "$CKPT"
status=$?

echo
echo "=== exit status: $status ==="
ls -1 "$CKPT" 2>&1 | tail -3
echo
echo "--- cost/step, the actual diagnostic ---"
LOG="$LOG" python - <<'PY'
import os, re, pathlib
log = pathlib.Path(os.environ["LOG"]).read_text()
print(f"{'step':>12} {'reward':>8} {'cost':>8} {'ep_len':>8} {'cost/step':>10} {'active':>7}")
for line in log.splitlines():
    if not line.startswith("step="):
        continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.]+)", line))
    try:
        c, l = float(g["eval/episode_cost"]), float(g["eval/avg_episode_length"])
    except KeyError:
        continue
    print(f"{int(float(g['step'])):>12,} {float(g['eval/episode_reward']):>8.2f} "
          f"{c:>8.2f} {l:>8.0f} {c/max(l,1):>10.4f} {g.get('training/crpo/active','--'):>7}")
PY

#!/bin/bash
# Runs ON a Vast instance. ant / minefield / CRPO at --safety_budget 30,
# 256x4 policy, lidar-only observation. 50M steps.
#
#   VAST_TAG=crpo bash cluster/vast/vast.sh crpo
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
# and tipped, never reaching the first mine row 1.25 m ahead -- it got WORSE
# at the thing it was being constrained about, and the constraint paid it.
#
# SO THE DIAGNOSTIC IS THE RATE, NOT THE TOTAL:
#
#     cost/step = eval/episode_cost / eval/avg_episode_length
#
#   * cost/step FALLING with ep_len roughly held  -> real hazard avoidance.
#   * cost/step RISING while episode_cost falls   -> the flip exploit. The run
#                                                    is void as a safety
#                                                    result no matter how good
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
# 2026-08-25 laptop attempt, this exact config, reads:
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
# ============================== CONFIG ====================================
#
# Matched to the Saute lidar-only arms so cost is comparable to their ~77-85:
# same hazard size, same lidar, same walls, same zero boundary weight, same
# 256x4 policy. Only --penalizer and --safety_budget differ.
#
# 1024 envs / 32 minibatches rather than the Saute arms' 512/16. LEARNING
# DENSITY IS IDENTICAL -- env_step_per_training_step scales with
# num_minibatches and so do the gradient updates per training step, so
# num_minibatches cancels out of updates-per-env-step (the 3090 sweep reports
# upd/Ms = 1562 at 512, 1024, 2048 and 4096 alike). It is ~8% faster for the
# same gradient math.
set -uo pipefail
cd /root/MorphoSafety || exit 1

source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

echo "=== box ==="
hostname; date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

STEPS="${STEPS:-50000000}"
BUDGET="${BUDGET:-30}"
NAME="${NAME:-ant_minefield_crpo_b30_wide}"
CKPT="/root/MorphoSafety/checkpoints/$NAME"

# PREFLIGHT: a GPU jax can actually SEE. Assert on device.platform, not on
# `import jax` merely not raising -- a missing CUDA plugin falls back to CPU
# silently, which once burned an eight-hour cluster allocation.
echo "=== GPU preflight ==="
python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, 'ABORT: no GPU -- jax fell back to CPU silently.'
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, no compute spent ==="; exit 1; }

# PREFLIGHT: the observation is the width this run is FOR. 47 = lidar-only,
# no morphology genes, no Saute scalar (CRPO has no obs-side component).
# A silent 63 would mean --foot_obstacle_obs leaked back on, which was
# measured WORSE on hazard cost and would make this incomparable to the
# Saute arms it is meant to sit beside.
echo "=== observation preflight ==="
python -c "
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = False
w = Minefield(**kw).observation_size
assert w == 47, f'expected 47 (lidar only), got {w}'
print(f'PASS: obs {w}, lidar-only')
" || { echo "=== observation preflight FAILED, no compute spent ==="; exit 1; }
[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }
echo

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

echo "=========================================================="
echo " CRPO  budget=$BUDGET  policy 256x4  lidar-only  ${STEPS} steps"
echo "   out $CKPT"
echo "   READ cost/step = episode_cost / avg_episode_length, NOT episode_cost"
echo "=========================================================="
echo

# python -u is NOT optional: Python block-buffers stdout when redirected and a
# segfault never flushes it.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer crpo --safety_budget "$BUDGET" \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes 256 256 256 256 \
  --num_envs 1024 --num_minibatches 32 \
  --num_timesteps "$STEPS" --num_evals 11 \
  --checkpoint_logdir "$CKPT"
status=$?

echo
echo "=== exit status: $status ==="
ls -1 "$CKPT" 2>&1 | tail -3
echo
echo "--- cost/step, the actual diagnostic ---"
python - <<'PY'
import re, pathlib
log = pathlib.Path("/root/MorphoSafety/logs/vast_crpo.log").read_text()
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
          f"{c:>8.2f} {l:>8.0f} {c/max(l,1):>10.4f} "
          f"{g.get('crpo/active','--'):>7}")
PY
echo
echo "PULL BEFORE YOU DESTROY:  VAST_TAG=crpo bash cluster/vast/vast.sh pull"

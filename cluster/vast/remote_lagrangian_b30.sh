#!/bin/bash
# Runs ON the Vast instance. PPO-Lagrangian at --safety_budget 30, ant on
# minefield, 256x4 policy, lidar-only. 50M steps.
#
#   VAST_TAG= bash cluster/vast/vast.sh lagrangian
#
# This is the Vast twin of cluster/ant_minefield_lagrangian_b30.sbatch. READ
# THAT FILE'S HEADER -- it carries the full argument for the multiplier lr, the
# CRPO comparison table, and what to read in the log. Only what differs on a
# rented box is documented here.
#
# ==================== IT WAITS FOR THE CRPO ARM TO FINISH ==================
#
# ONE GPU/JAX PROCESS AT A TIME. This box is running the matched CRPO arm; two
# concurrent CUDA contexts is this project's standing prohibition. Rather than
# requiring someone to watch for the exit and launch by hand, this blocks on
# the `crpo` tmux session disappearing -- the same pattern
# scripts/saute_penalty_sweep.sh uses to queue behind the A/B chain.
#
# NOT `pgrep train_ppo`: the CRPO script does its cost/step summary AFTER
# training exits, so the python process is gone while the session is still
# doing useful work. The tmux session is the thing that marks "done".
#
# ========================= WHY THIS RUN EXISTS ============================
#
# The CRPO arm on this same box, same flags but --penalizer crpo, plateaued for
# 30M steps at reward ~0.9 against a ~22 ceiling. Subtracting the upright bonus
# (0.0002/inner step) leaves ~0.5-0.6 m of travel on an 11 m corridor -- 5% of
# the course -- while its cost stayed at 52.8 against the budget of 30. Neither
# safe nor useful.
#
# `crpo/active` averaged 0.160 after 15M, i.e. **84% of gradient steps carried
# no reward term at all**, because CRPO's switch REPLACES the actor loss:
#
#     actor_loss = jnp.where(active, actor_loss, -loss_constraint)
#
# Lagrangian augments instead of replacing:
#
#     actor_loss += lagrange_multiplier * cost_advantage
#
# so the reward term is never zeroed. If traversal comes back, the switch was
# the problem rather than the budget. Every other flag is held identical so
# that is the only thing the comparison can be about.
#
# NOT THE FLIP EXPLOIT, and worth recording because it was the predicted
# failure: CRPO's episode length went 1540 -> 2279 against a 2500 cap, i.e.
# episodes got LONGER. Nothing was gained by terminating early. The 2026-08-22
# collapse (2035 -> 738 inner steps, cost/step doubling) did not reproduce.
set -uo pipefail
cd /root/MorphoSafety || exit 1

source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

if tmux has-session -t crpo 2>/dev/null; then
  echo "=== waiting for the CRPO arm (tmux session 'crpo') to finish ==="
  while tmux has-session -t crpo 2>/dev/null; do sleep 60; done
  echo "=== CRPO done at $(date -u +%H:%M:%S); starting Lagrangian ==="
  # The GPU takes a moment to release after the process exits; starting into a
  # still-held context is how you get a spurious RESOURCE_EXHAUSTED that looks
  # like a sizing problem.
  sleep 30
fi

echo "=== box ==="
hostname; date -u
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo

STEPS="${STEPS:-50000000}"
BUDGET="${BUDGET:-30}"
MULTIPLIER_LR="${MULTIPLIER_LR:-1e-4}"
NAME="${NAME:-ant_minefield_lagrangian_b30_wide}"
CKPT="/root/MorphoSafety/checkpoints/$NAME"

echo "=== GPU preflight ==="
python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, 'ABORT: no GPU -- jax fell back to CPU silently.'
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, no compute spent ==="; exit 1; }

echo "=== code / observation preflight ==="
python -c "
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
opts = next(a for a in T.build_argparser()._actions if a.dest == 'penalizer').choices
assert 'ppo_lagrangian' in opts, f'ABORT: no ppo_lagrangian penalizer; got {opts}'
kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = False
w = Minefield(**kw).observation_size
assert w == 47, f'ABORT: expected obs 47 (lidar only), got {w}'
print(f'PASS: obs {w}, ppo_lagrangian available')
" || { echo "=== code preflight FAILED, no compute spent ==="; exit 1; }
[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }
echo

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

echo "=========================================================="
echo " PPO-Lagrangian  budget=$BUDGET  multiplier_lr=$MULTIPLIER_LR"
echo "   policy 256x4, lidar-only, $STEPS steps -> $CKPT"
echo "   FIRST metric to check: training/lagrange_multiplier"
echo "     still ~0.01 at the end  -> the constraint NEVER ENGAGED and this"
echo "     run says nothing about Lagrangian. The repo default of 7e-7 gives"
echo "     0.011 of total travel over 50M; that is why this passes 1e-4."
echo "   SAFETY metric: cost/step = episode_cost / avg_episode_length,"
echo "     NOT episode_cost -- a run can halve the total by ending episodes"
echo "     sooner while getting worse per step."
echo "=========================================================="
echo

# python -u is NOT optional: Python block-buffers stdout when redirected and a
# segfault never flushes it.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer ppo_lagrangian --safety_budget "$BUDGET" \
  --lagrangian_multiplier_lr "$MULTIPLIER_LR" \
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
du -sh "$CKPT" 2>&1
echo
echo "--- the two diagnostics, together ---"
python - <<'PY'
import re, pathlib
p = pathlib.Path("/root/MorphoSafety/logs/vast_lagrangian.log")
if not p.is_file():
    raise SystemExit("(log not readable yet)")
print(f"{'step':>12} {'reward':>8} {'travel':>7} {'cost':>7} {'ep_len':>7} "
      f"{'cost/step':>10} {'multiplier':>11}")
for line in p.read_text().splitlines():
    if not line.startswith("step="):
        continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.eE+-]+)", line))
    try:
        c = float(g["eval/episode_cost"]); l = float(g["eval/avg_episode_length"])
        r = float(g["eval/episode_reward"])
    except (KeyError, ValueError):
        continue
    # travel = reward minus the upright bonus (0.0002 per INNER step), i.e.
    # what the policy earned by actually going somewhere. CRPO sat at 0.5-0.6.
    print(f"{int(float(g['step'])):>12,} {r:>8.2f} {r - 0.0002*l:>7.2f} "
          f"{c:>7.1f} {l:>7.0f} {c/max(l,1):>10.4f} "
          f"{g.get('training/lagrange_multiplier','--'):>11}")
PY
echo
echo "Compare against checkpoints/ant_minefield_crpo_b30_wide on this box"
echo "(same flags, --penalizer crpo): reward ~0.9, travel ~0.5-0.6 m of 11 m."

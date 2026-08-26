#!/bin/bash
# PPO-Lagrangian at --safety_budget 30, ant on minefield, 256x4 policy,
# lidar-only observation. 50M steps.
#
#   sbatch cluster/ant_minefield_lagrangian_b30.sbatch
#
# ================== WHY THIS RUN, AND WHAT IT IS PAIRED WITH ==============
#
# THE MATCHED ARM IS THE CRPO BUDGET-30 RUN (2026-08-26, rented 3090,
# checkpoints/vast/ant_minefield_crpo_b30_wide). Every flag below is identical
# to it except `--penalizer`. That arm did not produce an unsafe policy -- it
# produced a policy that DOES NOT DO THE TASK:
#
#     step    reward   cost   ep_len   cost/step   crpo/active
#     0        0.56    21.6     2436     0.0089      --
#     5.0M     1.02   102.0     2215     0.0461      0.269
#     15.1M    0.51    36.6     1991     0.0184      0.086
#     20.2M    0.73    40.5     1540     0.0263      0.227
#
# `healthy_reward` is 0.0002/inner step, so standing upright alone pays
# 0.0002 * ep_len. Subtracting it leaves 0.07 m of travel at step 0 and 0.42 m
# at 20M, against an 11 m corridor -- 4% of the course. The Saute arms on this
# same configuration reach reward ~21, a full traverse.
#
# THE CAUSE IS CRPO'S HARD SWITCH, and it is what this run changes.
# penalizers.py CRPO does
#
#     actor_loss = jnp.where(active, actor_loss, -loss_constraint)
#
# so on a step where `active` is False the reward term is not outweighed, it is
# ABSENT. At the observed crpo/active of 0.086-0.269, 73-91% of gradient steps
# carried no reward signal whatsoever. Lagrangian instead does
#
#     actor_loss += lagrange_multiplier * cost_advantage
#
# which never zeroes the reward term. If the "cannot learn the task" failure is
# caused by the switch rather than by the budget, this run should traverse.
#
# NOTE the budget is NOT the suspect. Measured at step 0 in this exact config,
# an untrained ant costs 21.6 -- comfortably INSIDE 30. Cost RISES with
# competence here (a policy that crosses the corridor passes far more mines
# than one flailing at spawn), so 30 is slack at init and binds only once the
# ant starts crossing. That is a well-posed constraint, not a hopeless one.
#
# ==================== --lagrangian_multiplier_lr 1e-4 =====================
#
# THE ONE NUMBER THAT DECIDES WHETHER THIS RUN MEANS ANYTHING. The repo default
# is 7e-7 and IT WOULD MAKE THIS RUN INERT. Do not use it here.
#
# The multiplier update is PLAIN GRADIENT ASCENT, not Adam, despite an
# `optax.adam` being constructed in Lagrangian.__init__ and then never used --
# `update()` passes `params.optimizer_state` straight through untouched:
#
#     new_multiplier = maximum(multiplier - learning_rate * constraint, 0)
#
# So the step is PROPORTIONAL to the constraint violation, unlike Adam where a
# step is ~lr regardless of gradient size. That makes the total travel
# computable. `penalizer.update` is called once per MINIBATCH (inside
# minibatch_step in ppo/training_step.py), and with the config below:
#
#     env steps per training step = 32 * 10 * 32 * 4          = 40,960
#     training steps in 50M                                    =  1,221
#     x num_minibatches(32) x num_updates_per_batch(2)          = 78,125 updates
#     normalised budget = (30 / 625) / (1 - 0.9)                =   0.48
#
# At a typical |constraint| ~ 0.2, total multiplier travel over the WHOLE RUN:
#
#     lr 7e-7   ->    0.011      <- DEFAULT. Starts at 0.01, ends at ~0.02.
#                                   The constraint never binds. A 2.5-hour
#                                   unconstrained PPO run wearing a safety flag.
#     lr 1e-5   ->    0.16
#     lr 1e-4   ->    1.6        <- THIS RUN. Multiplier can reach O(1), which
#                                   is where multiplier * cost_advantage becomes
#                                   comparable to the (normalised) reward
#                                   advantage.
#     lr 3e-3   ->   47
#     lr 1e-2   ->  156          <- and the 2026-08-22 measurement of this repo
#                                   at 1e-2 recorded the multiplier reaching
#                                   90-273, which is what validates the whole
#                                   calculation above rather than leaving it a
#                                   plausible-looking estimate.
#
# CAVEAT ON READING THOSE NUMBERS: this is a feedback loop, not an open-loop
# integration. The multiplier rises while the constraint is violated and falls
# when it is satisfied, so "travel" bounds how FAST it can respond, not where
# it settles. Too slow and it never binds inside the run; too fast and it
# oscillates, which is what 1e-2 did.
#
# 1e-4 is a derived starting point, NOT a validated one. If the multiplier is
# still under ~0.1 by the third eval, it is too slow -- resubmit with
# MULTIPLIER_LR=1e-3. If it spikes past ~50 and reward collapses, too fast.
#
# ========================= WHAT TO READ IN THE LOG ========================
#
#   training/lagrange_multiplier   THE FIRST THING TO CHECK. If it is still
#                                  ~0.01 at the last eval the constraint never
#                                  engaged and the run says nothing about
#                                  Lagrangian -- see the lr table above.
#   eval/episode_reward            The actual question. CRPO gave ~0.7 against
#                                  a ~22 ceiling. Anything above ~5 means the
#                                  reward term surviving is what mattered.
#   cost/step = eval/episode_cost / eval/avg_episode_length
#                                  THE SAFETY DIAGNOSTIC, not episode_cost.
#                                  A run can halve episode_cost purely by
#                                  ending episodes sooner: on 2026-08-22 CRPO
#                                  drove episode_cost BELOW untrained and
#                                  episode_safe to 0.97 while cost per step
#                                  DOUBLED, because terminate_on_flip zeroes
#                                  future cost and falling over is the cheapest
#                                  cost reduction available. Falling rate with
#                                  ep_len held = real avoidance. Rising rate
#                                  with ep_len shrinking = the flip exploit,
#                                  and the run is void as a safety result
#                                  however good episode_safe looks.
#   eval/episode_safe              Nearly uninformative on its own, for the
#                                  reason above. Do not lead with it.
#
# ============================== SIZING ====================================
#
# 50M at the 3090's measured trained rate for this config (~6800 sps at 1024
# envs, from the 2026-08-26 CRPO run) plus 11 evals at ~95 s:
#
#   50M / 6800    ~= 2.05 h
#   11 x 95 s     ~= 0.3 h
#   ------------------------
#                 ~= 2.4 h.  --time 6:00:00 is ~2.5x margin.
#
# 1024 envs / 32 minibatches matches the CRPO arm exactly. num_minibatches
# cancels out of updates-per-env-step (it scales env_step_per_training_step and
# the gradient updates per training step equally), so this is the same learning
# density as the 512/16 Saute arms, just faster.
#
#SBATCH --job-name=ant-lagr30
#SBATCH --output=ant_lagrangian_b30_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=06:00:00
set -u

echo "=== node ==="
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
echo "=== env ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-safety-gym
python --version
pip show jax mujoco 2>&1 | grep -E "^Name|^Version"
df -h ~ | tail -1
echo

# PREFLIGHT 1: a GPU that JAX can actually SEE.
#
# Assert on device PLATFORM, not on jax.devices() merely not raising. Job 40227
# landed on a node whose GPU0 was wedged; JAX printed a warning, silently fell
# back to CPU, and burned an eight-hour allocation producing zero data points.
echo "=== GPU preflight ==="
python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, ('ABORT: no GPU -- jax fell back to CPU silently. If nvidia-smi '
              'above reports a device handle error the NODE is broken: '
              'resubmit, and report it to support@wits-mss.supportsystem.com.')
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, nothing spent ==="; exit 1; }
echo

# PREFLIGHT 2: the observation is the width this run is FOR, and the penalizer
# actually exists in this checkout.
#
# 47 = lidar-only, no morphology genes, no Saute scalar. A silent 63 would mean
# --foot_obstacle_obs leaked back on -- measured WORSE on hazard cost -- and
# this would no longer be comparable to the CRPO arm it exists to sit beside.
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
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
echo

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# ---- overridable at submit time, e.g.
#        MULTIPLIER_LR=1e-3 sbatch cluster/ant_minefield_lagrangian_b30.sbatch
#      sbatch exports the submitting environment by default (--export=ALL), so
#      these reach the job without editing the file.
#
#      PREFIXED DELIBERATELY. A bare NAME= here would inherit whatever the
#      login shell already exports -- that exact bug sent a 12-hour run's
#      checkpoints to a directory named after the machine on 2026-08-25.
STEPS="${LAGR_STEPS:-50000000}"
BUDGET="${LAGR_BUDGET:-30}"
MULTIPLIER_LR="${LAGR_MULTIPLIER_LR:-1e-4}"
NAME="${LAGR_NAME:-ant_minefield_lagrangian_b30_wide}"
CKPT="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/$NAME"

[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }

echo "=========================================================="
echo " PPO-Lagrangian  budget=$BUDGET  multiplier_lr=$MULTIPLIER_LR"
echo "   policy 256x4, lidar-only, $STEPS steps -> $CKPT"
echo "   FIRST metric to check: training/lagrange_multiplier"
echo "   SAFETY metric: cost/step = episode_cost / avg_episode_length"
echo "=========================================================="
echo

# python -u is NOT optional: Python block-buffers stdout when redirected, and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
#
# action_repeat / episode_length / discounting / healthy_reward /
# terminate_on_flip / terminate_on_goal are deliberately NOT passed: they
# resolve from _ROBOT_DEFAULTS and print on the first line of the log. Pinning
# them here would freeze this run to whatever was correct the day it was
# written, and it must match the CRPO arm it is compared against.
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
LOGFILE="${SLURM_SUBMIT_DIR:-$PWD}/ant_lagrangian_b30_${SLURM_JOB_ID:-0}.log" \
python - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["LOGFILE"])
if not p.is_file():
    raise SystemExit(f"(log {p} not readable yet -- read it by hand)")
print(f"{'step':>12} {'reward':>8} {'cost':>8} {'ep_len':>8} {'cost/step':>10} {'multiplier':>11}")
for line in p.read_text().splitlines():
    if not line.startswith("step="):
        continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.eE+-]+)", line))
    try:
        c, l = float(g["eval/episode_cost"]), float(g["eval/avg_episode_length"])
    except (KeyError, ValueError):
        continue
    print(f"{int(float(g['step'])):>12,} {float(g['eval/episode_reward']):>8.2f} "
          f"{c:>8.2f} {l:>8.0f} {c/max(l,1):>10.4f} "
          f"{g.get('training/lagrange_multiplier','--'):>11}")
PY
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. Compare against checkpoints/vast/ant_minefield_crpo_b30_wide"
  echo "(same flags, --penalizer crpo). The question is whether reward clears"
  echo "CRPO's ~0.7 -- if it does, the hard switch was the problem, not the"
  echo "budget. Check training/lagrange_multiplier moved before believing it."
else
  echo "FAILED (status $status). Resume from the newest checkpoint above with"
  echo "--restore_checkpoint_path and the REMAINING budget as --num_timesteps."
fi

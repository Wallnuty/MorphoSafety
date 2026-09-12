#!/bin/bash
# Constrained ant on minefield with the EPISODE-LENGTH BUG FIXED.
# 50M steps, 256x4 policy, lidar-only, --safety_budget 50.
#
#   sbatch cluster/ant_minefield_safe_b50.sbatch                     # Lagrangian
#   SAFE_PENALIZER=crpo sbatch cluster/ant_minefield_safe_b50.sbatch # CRPO
#
# ======================= THE 2026-09-12 PLAN, FOUR JOBS =====================
#
# All at 50M, 1024/32, 256x4, lidar-only, budget 50, adaptive horizon, and the
# spawn now FIXED at y=0 (--start_y_jitter 0.0 is the new default -- the last
# per-episode randomness on minefield is gone; the world is identical every
# episode up to the policy's own action noise).
#
#  1. RESUME the Lagrangian arm +50M. It traverses (reward ~19-21) but at 50M
#     its cost was 1.6x the unconstrained run's and its multiplier still rising
#     in a straight line (integral-only controller). This decides whether the
#     integral controller converges or winds up -- i.e. whether lever (b), PID,
#     is needed. Build (b) AFTER this reports, not before.
#   SAFE_RESUME=$PWD/checkpoints/ant_minefield_ppo_lagrangian_b50_adaptive/000050380800 \
#     sbatch cluster/ant_minefield_safe_b50.sbatch
#
#  2. LEVER (c): scheduled penalty from scratch. kappa 0.01 -> 5 geometrically
#     over the first 60% of the run, gated on violation, then held.
#   SAFE_PENALIZER=scheduled sbatch cluster/ant_minefield_safe_b50.sbatch
#
#  4. UNCONSTRAINED CONTROL at 47 wide. Zero code: --penalizer none has the
#     identical SafePPONetworkParams structure, so it is both the proper
#     control (the Saute500 "reference" differs in obs width, batch config AND
#     seed -- and seeds diverged 2.2x at 5M) and the warm-start source for 3.
#   SAFE_PENALIZER=none sbatch cluster/ant_minefield_safe_b50.sbatch
#
#  3. LEVER (a): Lagrangian warm-started from 4, ONCE 4 HAS FINISHED. The
#     restore path already tolerates the penalizer mismatch (warns, starts the
#     multiplier fresh at 0.01) and the optimizer state carries over.
#   SAFE_RESUME=$PWD/checkpoints/ant_minefield_none_wide_ctrl/000050380800 \
#   SAFE_NAME=ant_minefield_ppo_lagrangian_b50_warm \
#     sbatch cluster/ant_minefield_safe_b50.sbatch
#
# NOT lever (b) yet, and NOT (a)+(b) in one run: they fix different failures
# (pressure-before-competence vs multiplier windup) and a combined arm could
# not say which one mattered. Run 1 is the windup test and it is free.
#
# ======================== RESULT OF THE FIRST PAIR ========================
#
#   arm                      reward   cost   ep_len   displacement   multiplier
#   crpo b50 adaptive         2.54    84.5     575       1.21 m        --
#   ppo_lagrangian b50 adpt  19.29   136.1     735       9.57 m       4.16
#   unconstrained (Saute500) 20.52    84.9     494      10.2  m        --
#
# LAGRANGIAN TRAVERSES. First constrained run in the project to do the task.
# But at 50M its cost is 1.6x the UNCONSTRAINED run's at the same step, with
# the multiplier still rising linearly (~0.36 per 5M, integral-only, constraint
# pinned at -4.5). Whether cost crosses below ~85 before the multiplier grows
# large enough to crush reward is what the resume answers. Single seed each,
# and the two differed 2.2x at 5M when the multiplier was 0.06 -- run-to-run
# variance is large here, so read a 1.6x gap with that in mind.
#
# Submit it BOTH ways. The pair is the experiment; a single arm cannot
# distinguish "the fix worked" from "this algorithm happens to cope".
#
# ==================== WHAT CHANGED, AND WHY IT MATTERS ====================
#
# `--adaptive_budget_horizon` (new 2026-09-05, OFF by default) is the whole
# point of this run.
#
# THE BUG. `--safety_budget B` is normalised in ppo/train.py as
# `(B / num_decision_steps) / (1 - safety_discounting)` where
# num_decision_steps is the episode-length CAP -- 2500//4 = 625 for the ants.
# Because the cost critic saturates at `vsc = c/(1-safety_discounting)` for a
# steady per-decision cost c, `constraint >= 0` reduces exactly to
# `c <= B/625`. That is a RATE that must hold in every state, NOT a total that
# must hold over an episode.
#
# Harmless while every episode runs to the cap. Not harmless under
# `--terminate_on_goal`, which is default-on for the ants, because episodes now
# END AT ARRIVAL. Total cost permitted, at B = 50:
#
#     episode length    OLD (fixed 625)    NEW (adaptive)
#     625 decisions          50.0               50.0
#     300 decisions          24.0               50.0
#     123 decisions           9.8               50.0    <- a policy that crosses
#      60 decisions           4.8               50.0
#
# **The faster the policy, the smaller its real allowance.** A crosser
# (~123 decisions) got a fifth of the budget a loiterer got. Loitering was the
# rational answer to the constraint as posed, and BOTH algorithms found it --
# measured 2026-08-26 at B=30, 50M steps each, identical flags but the
# penalizer:
#
#     penalizer      reward   travel   cost   ep_len   /decision
#     crpo             1.06    0.63m   52.2     2157      0.0967
#     ppo_lagrangian   1.25    0.85m   51.8     1990      0.1041
#     (unconstrained  20.52   ~11m     84.9      494      0.6879)
#
# 0.63 m and 0.85 m of an 11 m corridor, with episode length drifting back
# toward the 2500-step cap in both -- i.e. deliberately taking longer, which is
# exactly what a rate constraint pays for. Lagrangian did reach 4.75 m at 5M
# before the multiplier squeezed it back, which is the one thing that says the
# task is learnable at all.
#
# THE FIX estimates the episode length from the batch -- `termination` and
# `truncation` are already computed for GAE, so their mean is the per-decision
# probability of an ending and its reciprocal is the mean length -- and divides
# by that instead of by the cap. Clamped to [1, 625], so a batch in which
# nothing ends degrades exactly to the old behaviour rather than dividing by
# zero. Allowance * length == B for every policy, so loitering buys nothing.
#
# ===================== --safety_budget 50 MEANS SOMETHING NOW =============
#
# Under the fix B is an EPISODE TOTAL, directly comparable to
# `eval/episode_cost`. It is no longer a rate in disguise, so for the first
# time in this project the flag can be set against a measurement:
#
#     unconstrained traversing policy   episode_cost ~85
#     this run                          B = 50   (~41% reduction, still crossing)
#
# HONEST CAVEAT, because it decides how to read a failure. Nobody has measured
# what a GOOD traverse costs -- 85 comes from a policy trained with Saute at
# budget 500, which never bound, so it is a reward-maximiser indifferent to
# cost. 85 is an UPPER bound on the achievable minimum, not an estimate of it.
# If 50 turns out infeasible, that is evidence about the ARENA (mine density vs
# the ant's foot placement), not about CRPO or Lagrangian. The lattice does
# leave 0.52 m gaps against a 0.212 m limb reach, and cost is charged only for
# GROUNDED geoms -- a leg swung over a mine is free -- so cheap crossings are
# not obviously ruled out. Deciding that properly needs a cost-minimising run,
# which this is not.
#
# ============== --lagrangian_multiplier_lr 1e-5, NOT 1e-4 =================
#
# The multiplier update is PLAIN GRADIENT ASCENT, not Adam (`optax.adam` is
# built in `Lagrangian.__init__` and then never used -- `update()` passes
# `optimizer_state` straight through), so the step is proportional to the
# violation and total travel is computable:
#
#     travel ~= n_updates * lr * mean|constraint|
#     n_updates = (50M / 40,960) * num_minibatches(32) * num_updates_per_batch(2)
#               = 78,125
#
# At B=30 with the OLD normalisation, lr 1e-4 drove the multiplier 0.01 -> 17.6,
# monotone, never falling -- it ratcheted straight through a policy that had
# already learned to traverse 4.75 m. The fix makes |constraint| LARGER, not
# smaller (the rescaled budget is ~5x bigger but so is the violation once the
# ant actually crosses), so 1e-4 would ratchet at least as hard. 1e-5 targets a
# multiplier settling around O(1), where `multiplier * cost_advantage` is
# comparable to the normalised reward advantage rather than swamping it.
#
# Still a derived number, not a validated one. If the multiplier is under ~0.1
# by the third eval it is too slow -- resubmit SAFE_MULT_LR=1e-4.
#
# ========================= WHAT TO READ IN THE LOG ========================
#
#   training/budget_mean_episode_decisions
#       THE FIRST THING TO CHECK. It is the measured mean episode length the
#       budget is being divided by. It should track eval/avg_episode_length/4.
#       IF IT SITS AT 625 THE FIX NEVER ENGAGED and this run is the old one.
#   training/budget_effective_rate
#       cost per decision currently permitted. Rises as episodes shorten.
#   eval/episode_reward
#       THE ACTUAL QUESTION. ~20-21 is a full traverse; ~1 is the loitering
#       collapse both arms fell into at B=30. Anything above ~5 means the fix
#       changed the behaviour.
#   eval/episode_cost
#       now directly comparable to B=50, and to the unconstrained 85.
#   training/lagrange_multiplier   (Lagrangian only)
#       must MOVE off 0.01, and should stop rising once cost sits near budget.
#       A monotone climb is the ratchet -- lower the lr.
#   training/crpo/active           (CRPO only)
#       fraction of gradient steps that KEPT the reward term. Read it as
#       "fraction still optimising reward", NEVER as "fraction violating".
#       It averaged 0.160 at B=30, i.e. 84% of steps had no reward signal.
#   cost/step = eval/episode_cost / eval/avg_episode_length
#       still worth watching: a run can cut episode_cost purely by ending
#       episodes sooner. Falling rate with ep_len held is real avoidance.
#
# ============================== SIZING ====================================
#
# 50M at the 3090's measured trained rate for this config (~6800 sps at 1024
# envs, from the 2026-08-26 runs) plus 11 evals at ~95 s:
#
#   50M / 6800  ~= 2.05 h ;  11 x 95 s ~= 0.3 h  ->  ~2.4 h
#
# --time 06:00:00 is ~2.5x margin. 1024 envs / 32 minibatches is the same
# LEARNING DENSITY as the 512/16 arms -- num_minibatches scales
# env_step_per_training_step and the gradient updates per training step
# equally, so it cancels out of updates-per-env-step -- just faster.
#
#SBATCH --job-name=ant-safe50
#SBATCH --output=ant_safe_b50_%j.log
#SBATCH --partition=bigbatch
# mscluster45's GPU was wedged on 2026-09-07 -- job 50715 landed there and
# nvidia-smi reported "Unable to determine the device handle for GPU0", so jax
# fell back to CPU and the preflight aborted it.
#
# EXCLUDING IT IS NOT PARANOIA, IT IS THE ONLY WAY TO STOP LANDING ON IT. A
# node whose jobs die in seconds is always idle, so it is always the first node
# free -- it will catch submission after submission while healthy nodes stay
# busy. The preflight makes each hit cheap (seconds, not hours), but it cannot
# stop them recurring.
#
# Remove this line once the node is repaired. Check with:
#     sinfo -p bigbatch -N -l          # look for drain/down states
# and report a wedged GPU to support@wits-mss.supportsystem.com -- a node in
# that state stays broken until someone resets it.
#SBATCH --exclude=mscluster45
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

# PREFIXED DELIBERATELY. A bare NAME=/STEPS= here inherits whatever the login
# shell already exports -- that exact bug sent a 12-hour run's checkpoints to a
# directory named after the machine on 2026-08-25.
STEPS="${SAFE_STEPS:-50000000}"
BUDGET="${SAFE_BUDGET:-50}"
PENALIZER="${SAFE_PENALIZER:-ppo_lagrangian}"
MULT_LR="${SAFE_MULT_LR:-1e-5}"
RESUME="${SAFE_RESUME:-}"
KAPPA_MAX="${SAFE_KAPPA_MAX:-5.0}"        # scheduled only
RAMP_FRAC="${SAFE_RAMP_FRAC:-0.6}"        # scheduled only
if [ "$PENALIZER" = "none" ]; then
  # The unconstrained control. Budget/horizon flags are parsed and ignored;
  # the name must not pretend a budget applied.
  NAME="${SAFE_NAME:-ant_minefield_none_wide_ctrl}"
else
  NAME="${SAFE_NAME:-ant_minefield_${PENALIZER}_b${BUDGET}_adaptive}"
fi
# A RESUMED RUN MUST NOT WRITE INTO THE DIRECTORY IT RESUMES FROM. The trainer's
# step counter restarts at 0 (env_steps is not in the checkpoint), so it would
# write 000005038080 again and overwrite the source run's early checkpoints.
[ -n "$RESUME" ] && [ -z "${SAFE_NAME:-}" ] && NAME="${NAME}_r$(date -u +%m%d%H%M)"
CKPT="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/$NAME"

if [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { echo "ABORT: SAFE_RESUME=$RESUME is not a directory"; exit 1; }
  # The Lagrange multiplier lives in penalizer_params, which IS part of the
  # 4-element restore -- so a resumed Lagrangian run continues from the
  # multiplier it reached, not from 0.01. The adaptive horizon is stateless.
  echo "=== RESUMING from $RESUME (multiplier restored with it) ==="
  set -- --restore_checkpoint_path "$RESUME"
else
  set --
fi

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

# PREFLIGHT 2: THE CHECKOUT ACTUALLY HAS THE FIX.
#
# This is the one that matters. Without --adaptive_budget_horizon the run
# silently becomes a repeat of the 2026-08-26 budget experiment, with a log
# that looks completely normal and a result that means nothing. The mscluster
# checkout has gone stale before -- a previous sweep ran eight hours of
# pre-change code -- so assert on the actual API, not on `git log`.
echo "=== code / observation preflight ==="
python -c "
import inspect
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
from mjx_safety_gym.algorithms.ppo import train as ppo_train, losses as ppo_losses

dests = {a.dest for a in T.build_argparser()._actions}
opts = next(a for a in T.build_argparser()._actions if a.dest == 'penalizer').choices
assert 'scheduled' in opts, (
    'ABORT: no scheduled penalizer in this checkout (added 2026-09-12). git pull.')
assert 'start_y_jitter' in dests, (
    'ABORT: no --start_y_jitter in this checkout (added 2026-09-12); the spawn '
    'would still be jittered and this run would not match the others. git pull.')
assert 'adaptive_budget_horizon' in dests, (
    'ABORT: no --adaptive_budget_horizon in this checkout. It was added '
    '2026-09-05 and IS THE ENTIRE POINT of this run -- without it this '
    'silently repeats the budget-30 experiment. git pull.')
assert 'adaptive_budget_horizon' in inspect.signature(ppo_train.train).parameters, (
    'ABORT: ppo.train does not accept adaptive_budget_horizon -- partial '
    'checkout, the CLI flag would be dropped on the floor. git pull.')
assert 'budget_decision_steps' in inspect.signature(ppo_losses.make_losses).parameters, (
    'ABORT: ppo/losses.make_losses has no budget_decision_steps -- the fix '
    'itself is missing. git pull.')

kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = False
w = Minefield(**kw).observation_size
assert w == 47, f'ABORT: expected obs 47 (lidar only), got {w}'
print(f'PASS: obs {w}, adaptive_budget_horizon present end to end')
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
echo

[ -d "$CKPT" ] && { echo "ABORT: $CKPT exists; refusing to mix two runs"; exit 1; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

echo "=========================================================="
echo " $PENALIZER  budget=$BUDGET (EPISODE TOTAL, adaptive horizon)"
echo "   policy 256x4, lidar-only, $STEPS steps -> $CKPT"
[ "$PENALIZER" = "ppo_lagrangian" ] && echo "   multiplier_lr=$MULT_LR"
[ "$PENALIZER" = "scheduled" ] && echo "   kappa 0.01 -> $KAPPA_MAX over the first $RAMP_FRAC of the run, then held"
[ "$PENALIZER" = "none" ] && echo "   UNCONSTRAINED CONTROL -- budget flags ignored; warm-start source for lever (a)"
echo "   CHECK FIRST: training/budget_mean_episode_decisions"
echo "     pinned at 625 => the fix never engaged, this is the OLD experiment"
echo "   THE QUESTION: eval/episode_reward. ~20 = traverses, ~1 = loiters"
echo "     (both arms collapsed to ~1 at budget 30 under the old semantics)"
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
# written, and it must match the arms it is compared against.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer "$PENALIZER" --safety_budget "$BUDGET" \
  --adaptive_budget_horizon \
  --lagrangian_multiplier_lr "$MULT_LR" \
  --penalty_kappa_max "$KAPPA_MAX" --penalty_ramp_frac "$RAMP_FRAC" \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes 256 256 256 256 \
  --num_envs 1024 --num_minibatches 32 \
  --num_timesteps "$STEPS" --num_evals 11 \
  --checkpoint_logdir "$CKPT" "$@"
status=$?

echo
echo "=== exit status: $status ==="
ls -1 "$CKPT" 2>&1 | tail -3
du -sh "$CKPT" 2>&1
echo
echo "--- the diagnostics, together ---"
LOGFILE="${SLURM_SUBMIT_DIR:-$PWD}/ant_safe_b50_${SLURM_JOB_ID:-0}.log" \
BUDGET="$BUDGET" python - <<'PY'
import os, re, pathlib
p = pathlib.Path(os.environ["LOGFILE"])
if not p.is_file():
    raise SystemExit(f"(log {p} not readable yet -- read it by hand)")
B = float(os.environ["BUDGET"])
print(f"{'step':>12} {'reward':>7} {'displ':>7} {'cost':>7} {'vs B':>6} "
      f"{'ep_len':>7} {'mean_dec':>9} {'mult/active':>12}")
for line in p.read_text().splitlines():
    if not line.startswith("step="):
        continue
    g = dict(re.findall(r"([\w/]+)=(-?[\d.eE+-]+)", line))
    try:
        c = float(g["eval/episode_cost"]); l = float(g["eval/avg_episode_length"])
        r = float(g["eval/episode_reward"])
    except (KeyError, ValueError):
        continue
    # displacement = (reward - upright bonus) / 2. The /2 is not optional:
    # reward telescopes to dx + (d_initial - d_final) + healthy, and for a
    # corridor traverse BOTH terms equal the displacement, so reward counts it
    # twice -- a full 11 m traverse scores ~22. The first version of this
    # printer omitted the /2 and reported 19 m on an 11 m corridor.
    knob = (g.get("training/lagrange_multiplier") or g.get("training/scheduled/kappa")
            or g.get("training/crpo/active", "--"))
    print(f"{int(float(g['step'])):>12,} {r:>7.2f} {(r-0.0002*l)/2:>6.2f}m {c:>7.1f} "
          f"{c/B:>5.1f}x {l:>7.0f} "
          f"{g.get('training/budget_mean_episode_decisions','--'):>9} {knob:>12}")
PY
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. Now submit the OTHER arm:"
  if [ "$PENALIZER" = "ppo_lagrangian" ]; then
    echo "  SAFE_PENALIZER=crpo sbatch cluster/ant_minefield_safe_b50.sbatch"
  else
    echo "  sbatch cluster/ant_minefield_safe_b50.sbatch     # ppo_lagrangian"
  fi
  echo "Unconstrained reference for both: episode_cost ~85, reward ~20.5"
  echo "(checkpoints/ant_minefield_saute500_lidar_wide, 50M, budget never bound)."
else
  echo "FAILED (status $status). Resume from the newest checkpoint above with"
  echo "--restore_checkpoint_path and the REMAINING budget as --num_timesteps."
fi

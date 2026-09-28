#!/bin/bash
# Schaff et al. (ICRA 2019) co-design: train the DESIGN DISTRIBUTION jointly
# with the policy. 300M env steps, ant on minefield, FINAL SAFE-RL RECIPE
# (2026-09-25): PPO-Lagrangian at budget 25, per-foot 7x7 hazard grid, linear
# cost at the 5 cm gate, flip cost 50, finish line. These are the trainer's
# defaults now; they are also passed explicitly below so a later default
# change cannot silently alter this run.
#
#   Schaff, Yunis, Chakrabarti & Walter, "Jointly Learning to Construct and
#   Control Agents using Deep Reinforcement Learning", arXiv 1801.01432.
#
# WHAT MAKES THIS DIFFERENT FROM EVERY MORPHOLOGY RUN BEFORE IT. Until now
# `morphology.randomization_fn` sampled bodies from a FIXED uniform
# distribution, once, at wrapper-construction time -- the policy was
# conditioned on the genes, but nothing made the bodies improve. Here a
# Gaussian mixture over the 7 genes is itself trained by REINFORCE on
# per-design scores, and the mixture commits to a single design by CHOPPING
# (8 -> 4 -> 2 -> 1). The decoupled pipeline it replaces was measured on
# 2026-08-18 to leave 51% of candidate bodies outside the trained mass range
# and 29% of unseen bodies unreliable; training the distribution removes the
# train/test mismatch by construction.
#
# ============================ BEFORE YOU SUBMIT ============================
#
# 1. THE CLUSTER CHECKOUT MUST HAVE TODAY'S CODE. This job needs design-state
#    checkpointing and --design_tmax, both written 2026-08-24/25 and both
#    absent from anything pushed earlier. The mscluster checkout has gone
#    stale before -- a previous sweep ran eight hours of pre-change code. The
#    preflight below asserts on the actual API rather than trusting `git log`,
#    so a stale checkout fails in seconds instead of at hour eight.
#
# 2. Environment, once: miniconda in $HOME, `conda create -n mjx-safety-gym
#    python=3.11`, `pip install -e .[cuda]` from the repo. No `module load
#    cuda-*` -- jax's plugin bundles its own CUDA 12 runtime and depends only
#    on the host driver.
#
# 3. Run cluster/gpu_smoke_test.sbatch first if this node is new to you.
#
#   sbatch cluster/ant_morphology_codesign.sbatch
#
# ============================== SIZING ====================================
#
# THROUGHPUT IS AN EXTRAPOLATION, NOT A MEASUREMENT -- read training/sps off
# the first eval line and correct the --time on the next submission.
#
# THE NUMBER TO EXTRAPOLATE FROM IS 7127, NOT THE 7755 IN THE SWEEP.
# `training/sps` is reported PER EPOCH, and it RISES 8-28% within a single run
# as the policy learns: an untrained ant flails and flips, which generates far
# more contacts than a settled trot, so the physics itself gets cheaper. The
# 50M Saute runs all show it (logs/vast/*.log):
#
#     ant_minefield_saute500_lidar    512 envs   5869 -> 6617   (+13%)
#     ant_minefield_saute50_lidar    1024 envs   5561 -> 7127   (+28%)
#
# Every row of logs/vast/throughput.log ran 1M steps with --num_evals 2, i.e.
# exactly ONE epoch, under an untrained policy -- so the whole sweep measures
# the slow regime, and 7755 is an untrained 1024-env number. 7127 is the
# measured TRAINED rate at 1024 envs. Morphology batching then costs ~17% on
# the laptop (403 -> 335 sps) from heterogeneous per-lane geometry:
#
#   300M / 5900          ~= 14.1 h training   (7127 x 0.83)
#   10 evals x ~360 s    ~=  1.0 h   (320 episodes each, 625-decision unrolls)
#   host model rebuilds  ~=  0.1 h   (80 iterations x 16 designs x ~0.2 s)
#   ----------------------------------
#                        ~= 15 h.  --time 24h is ~1.6x margin; bigbatch caps
#                                  at 72h so there is room to raise it.
#
# Expect this run to sit at the LOW end, and possibly below it: co-design
# resamples the population every design iteration, so the policy is repeatedly
# put onto bodies it has not settled on and may never reach the clean-gait
# regime that earns the +28%.
#
# WHY 1024 ENVS AND 32 MINIBATCHES -- AND WHY THAT IS FREE. They must move
# together: validate() requires num_envs to divide batch_size*num_minibatches.
# It is tempting to read that as a trade, because doubling num_minibatches
# doubles env_step_per_training_step (32*10*32*4 = 40,960) and so halves the
# number of training steps for a fixed budget. It is not a trade, because the
# same doubling also doubles the gradient updates PER training step:
#
#     env_step_per_training_step = batch_size * unroll_length
#                                  * num_minibatches * action_repeat
#     gradient updates / step    = num_minibatches * num_updates_per_batch
#
# num_minibatches cancels. The 3090 sweep confirms it directly -- every Phase A
# row reports upd/Ms = 1562, identical at 512, 1024, 2048 and 4096 envs. So
# this is a speedup at IDENTICAL learning density, not a different learning
# curve. batch_size and unroll_length are the two that would genuinely change
# it, and both are left alone.
#
# HOW BIG A SPEEDUP DEPENDS ON THE REGIME, so do not quote the sweep's +31%
# (5917 -> 7755) as what this run will see: those are both UNTRAINED numbers.
# Trained, the two measured 50M runs give 6617 at 512 and 7127 at 1024, i.e.
# about +8%. Still free, just smaller than advertised.
#
# WHY 8 DESIGN UPDATES PER EVAL AND NOT 16. This is the one thing 1024 envs
# really does constrain. A design is scored only off episodes that FINISH
# inside its iteration, so what matters is episodes per lane:
#
#     per_lane = num_timesteps / (num_evals-1) / updates / num_envs / ep_len
#
#   updates=16 -> 160 iterations, 0.73 ep/lane   <- CLI warns, and is right to
#   updates= 8 ->  80 iterations, 1.46 ep/lane   <- what this runs
#
# Below 1.0 a lane cannot complete a full-length episode at all, so the only
# episodes that finish are early terminations -- flips and arrivals. Designs
# that neither flip nor arrive go UNSCORED and drop out of the gradient, which
# is exactly backwards: it is the stable, slow-but-upright bodies that get
# silently excluded, and the shortfall term that is supposed to rank
# non-arrivals never gets used. Doubling the iteration length fixes it and
# costs design updates: 80 iterations of ~94 episodes each, against 160 of ~47.
#
# IF YOU WANT 160 UPDATES BACK, the only way at 1024 envs is more steps, since
# the two are coupled through the line above: --num_timesteps 600000000 with
# --design_updates_per_eval 16 gives 160 iterations at 1.46 ep/lane, ~26 h of
# training, --time 48:00:00. That is the better run if the allocation is there.
#
# ==================== WHAT YOU SEE WHILE IT IS RUNNING ====================
#
# TWO CADENCES, both to $SLURM_SUBMIT_DIR/ant_codesign_<jobid>.log, unbuffered
# (`python -u`), so `tail -f` works from the login node the whole way through.
#
#   every design iteration  (~80 lines, one roughly every 10 min):
#     design it=  7/  80 step=  26,214,400 score= -198.286 comp=8
#       ep/design=93.8 arrive=0.00 mode=[+0.22 -0.37 -0.73 -0.77 +0.50 ...]
#   every eval              (11 lines, one roughly every 1.3 h):
#     step=... eval/episode_cost=... training/sps=... design/... eval/...
#
# The per-iteration line exists because it had to: `progress_fn` only fires at
# evals and `training_metrics` is rebuilt every iteration, so before it the log
# kept ONE design iteration in every --design_updates_per_eval and threw the
# rest away -- including any chop that did not happen to land on the last one.
# A `chopped=`, `UNSCORED_DESIGNS=` or `SKIPPED_UPDATE=` suffix is appended to
# that line only when it occurs, so its presence is itself the signal.
#
# ========================= WHAT TO READ IN THE LOG ========================
#
#   design/mode_0..6            the current best design, in [-1,1] gene space.
#                               THIS IS THE DELIVERABLE. Watch it move.
#   design/components           8 -> 4 -> 2 -> 1 across the chops. If it is
#                               still 8 at the end, chopping never fired and
#                               the mixture never committed -- check chop_freq.
#   design/score_mean           SPEED (DESIGN_OBJECTIVE=speed, the default):
#                               net +x metres per episode second, non-arrivals
#                               charged the full 625 decisions. Higher = better.
#   design/speed_mean           the same, for sanity.
#   design/arrival_rate         fraction of episodes that crossed the finish
#                               line. 0.0 early is expected and not a failure.
#   design/UNSCORED_DESIGNS     should be absent. If it appears, iterations are
#                               too short -- see WHY 8 DESIGN UPDATES above.
#   design/SKIPPED_UPDATE       appears when fewer than 2 designs were scored;
#                               a lone sample standardises to 0 and carries no
#                               gradient.
#   eval/episode_flipped        fraction of eval episodes that went over.
#   eval/episode_out_of_bounds  fraction that left the corridor (ends the
#                               episode and costs --exit_cost since 2026-09-26).
#   eval/episode_cost           LOGGED, NOT OPTIMISED BY THE SEARCH. The score
#                               is speed only. Safety enters through the
#                               Lagrangian constraining the policy on every
#                               body, so a design is scored on how fast it
#                               goes WHILE its controller keeps the budget.
#                               DESIGN_OBJECTIVE=safe_time scores cost too.
#   training/lagrange_multiplier  sample-weighted lambda over the population;
#                               settles ~1.3-1.9 in the single-body arms,
#                               pinned at the 3.0 cap = budget infeasible.
#   lam=[c<k>:<value> ...]      ONE LAMBDA PER LIVE GMM COMPONENT (2026-09-28,
#                               LAMBDA_PER_COMPONENT=1, the default), on every
#                               design-iteration line. Each holds its own
#                               component's bodies to the budget; a component
#                               whose lambda sits at the cap cannot meet it.
#   training/constraint_c<k>    that component's budget slack (negative = over
#                               budget); shrunk toward 0 for rare components.
#
# ============================ RESUMING ====================================
#
# The design distribution is checkpointed as design_state.npz INSIDE each
# checkpoint step directory. To resume, point at that directory and pass the
# ORIGINAL total as --design_tmax, because the trainer's step counter restarts
# at zero on a resume while the design schedule (lr anneal, phases, chop
# interval) is measured in absolute steps:
#
#   --restore_checkpoint_path $CKPT/000123456789 \
#   --num_timesteps <steps REMAINING> --design_tmax 300000000
#
# A resume prints "restored design state ... N components live ... design axis
# resumes at ...". IF THAT LINE IS ABSENT you are resuming the policy onto a
# fresh distribution and the search has silently restarted -- there is a
# warning for that case too, do not ignore it.
#
#SBATCH --job-name=ant-codesign
#SBATCH --output=ant_codesign_%j.log
#SBATCH --partition=bigbatch
# Wedged GPUs (nvidia-smi "Unable to determine the device handle for GPU0"):
# mscluster45 since 2026-09-07, mscluster83 since 2026-09-12. A node whose jobs
# die in seconds is always idle, so it is always first in the queue -- exclude
# or it catches every submission. Same list as ant_minefield_safe_b50.sbatch;
# drop a node once support has reset it (sinfo -p bigbatch -N -l).
#SBATCH --exclude=mscluster45,mscluster83
#SBATCH --time=36:00:00
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

# PREFLIGHT 1: a GPU that JAX can actually see.
#
# Assert on device PLATFORM, not on jax.devices() merely not raising. Job
# 40227 landed on a node whose GPU0 was wedged; JAX printed a warning, fell
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

# PREFLIGHT 2: the checkout is new enough, and the observation is the width
# this run is FOR.
#
# 250 = 47 (proprio + goal + lidar) + 196 (4 feet x 7x7 grid) + 7 genes; no
# budget scalar (that was Saute's). A silent 54 would mean the foot grid is
# off -- the thing that makes the constraint act on placement; a silent 243
# would mean morphology conditioning is not reaching the observation, which
# would train a policy that cannot see which body it is driving -- the whole
# method, gone, with a log that looks completely normal.
echo "=== code / observation preflight ==="
python -c "
from mjx_safety_gym import design, morphology
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T

for name in ('save', 'load', 'state_dict', 'load_state_dict'):
    assert hasattr(design.DesignLoop, name), (
        f'ABORT: design.DesignLoop has no {name}() -- this checkout predates '
        f'design-state checkpointing (2026-08-24). A crash would lose the '
        f'design distribution, which IS the result of this run. git pull.')
assert 'design_tmax' in {a.dest for a in T.build_argparser()._actions}, (
    'ABORT: no --design_tmax in this checkout; resumes would anneal the design '
    'lr against the wrong horizon. git pull.')

p = T.build_argparser().parse_args(['--robot', 'ant', '--task', 'minefield'])
T.apply_robot_defaults(p)
assert (p.penalizer, p.safety_budget, p.foot_hazard_grid, p.flip_cost,
        p.corridor_walls, p.terminate_out_of_bounds, p.exit_cost,
        p.solver_iterations, p.solver_ls_iterations) == (
    'ppo_lagrangian', 25.0, 7, 50.0, False, True, 50.0, 4, 8), (
    f'ABORT: trainer defaults are not the 2026-09-25 recipe: {p.penalizer} '
    f'b{p.safety_budget} grid{p.foot_hazard_grid} flip{p.flip_cost}. git pull.')
kw = dict(T.robot_env_kwargs('ant'))
kw['morphology_conditioning'] = True
env = Minefield(**kw)
assert env._ground_contact_eps == 0.02 and env._hazard_cost_shape == 'linear', (
    'ABORT: cost is not linear at the 2 cm gate (default since 2026-09-28). git pull.')
w = env.observation_size
assert w == 243 + morphology.NUM_GENES, f'expected 250 (grid 7 + genes), got {w}'
assert 'lagrange_per_component' in {a.dest for a in T.build_argparser()._actions}, (
    'ABORT: no per-component Lagrange multipliers in this checkout (2026-09-28). git pull.')
o = env.mj_model.opt
assert (o.iterations, o.ls_iterations) == (4, 8), f'ABORT: solver caps {o.iterations}/{o.ls_iterations}, expected 4/8'
print(f'PASS: obs {w} (grid 7 + genes), 2 cm linear cost, design checkpointing present')
import jax, numpy as np, jax.numpy as jp
a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
ref = a.astype(np.float64) @ a.astype(np.float64)
err = float(np.abs(np.asarray(jax.jit(jp.matmul)(a, a), np.float64) - ref).max() / np.abs(ref).max())
prec = jax.config.jax_default_matmul_precision
assert prec == 'highest' and err < 1e-5, (
    f'ABORT: matmuls are not float32 here (precision={prec}, rel err {err:.1e}; '
    f'TF32 is ~3e-4). Unset JAX_DEFAULT_MATMUL_PRECISION, or the checkout predates '
    f'mjx_safety_gym/numerics.py (2026-09-26).')
print(f'PASS: float32 matmuls on {jax.default_backend()} (rel err {err:.1e})')
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
echo

# 0.75 of 24 GB. The card has room; this is the same fraction the other
# cluster jobs use, left alone so a VRAM problem here is comparable to theirs.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# ---- overridable at submit time, e.g.
#        DESIGN_LR=0.03 RESUME=<ckpt-step-dir> STEPS=240000000 sbatch <this>
#      sbatch exports the submitting environment by default (--export=ALL), so
#      these reach the job without editing the file.
STEPS="${STEPS:-2000000000}"
DESIGN_TMAX="${DESIGN_TMAX:-2000000000}"
# SCHEDULE: SCHAFF'S ANT, EXTENDED FOR THE CONSTRAINT (2026-09-28). Upstream's
# ant is 1.5B: design frozen for the first 200M (the policy learns to walk on
# every sampled body first), a chop every 200M after that (8->4->2->1 at 400M,
# 600M, 800M), the last 100M fine-tuning the policy on the mode. Kept: the
# warm-up and the chops. Added (user's call, upstream is unconstrained): 500M
# more, as a longer search (to 1.7B) and a 300M final fine-tune, because the
# constrained single-ant runs needed 150M+ to settle into the budget. 2B total,
# ~15 h on a 3090. 40 evals x 13 design iterations = 520 iterations of ~3.85M
# steps = 1.50 episodes per lane each (the design updates are the ~390 between
# 200M and 1.7B).
NUM_EVALS="${NUM_EVALS:-41}"
DESIGN_UPDATES="${DESIGN_UPDATES:-13}"
CHOP_FREQ="${CHOP_FREQ:-200000000}"
STEPS_BEFORE="${STEPS_BEFORE:-200000000}"
STEPS_AFTER="${STEPS_AFTER:-300000000}"
# 0.03, not 1e-3: at 1e-3 the 2026-08 run's distribution moved 2% of the gene
# space in 60M steps (sigma 0.5763 vs 0.577 init) -- inert. See
# cluster/vast/remote_codesign.sh's DESIGN_LR section for the arithmetic.
DESIGN_LR="${DESIGN_LR:-0.03}"
# 2026-09-17: score designs on time AND cost. DESIGN_OBJECTIVE=safe_time with
# DESIGN_COST_WEIGHT (decisions per unit of episode cost; 1.0 makes a careless
# nominal-ant crossing's ~58 linear cost units worth ~half its arrival time)
# is the morphology-safety question. Default keeps the time-only objective.
DESIGN_OBJECTIVE="${DESIGN_OBJECTIVE:-speed}"
DESIGN_COST_WEIGHT="${DESIGN_COST_WEIGHT:-0}"
FLIP_COST="${FLIP_COST:-50}"
PENALIZER="${PENALIZER:-ppo_lagrangian}"   # none = the Schaff baseline
# Budget 2 (2026-09-28, user's call): at the 2 cm gate the single-ant runs
# reached cost 2.7 at budget 5 and 1.9 at budget 0, so 25 would barely bind.
BUDGET="${BUDGET:-2}"
MULT_LR="${MULT_LR:-1.5e-5}"
MULT_MAX="${MULT_MAX:-3.0}"
MULT_INIT="${MULT_INIT:-0.01}"
GRID="${GRID:-7}"
# 1 = one Lagrange multiplier per GMM component (default since 2026-09-28):
# every component's bodies held to the budget themselves. 0 = one shared.
LAMBDA_PER_COMPONENT="${LAMBDA_PER_COMPONENT:-1}"
if [ "$LAMBDA_PER_COMPONENT" = "1" ]; then LPC_FLAG=--lagrange_per_component; else LPC_FLAG=--no-lagrange_per_component; fi
RESUME="${RESUME:-}"

# A RESUMED RUN MUST NOT WRITE INTO THE DIRECTORY IT IS RESUMING FROM. The
# trainer's step counter restarts at 0 (env_steps is not in the checkpoint), so
# it would write 000000030146560 again and OVERWRITE the checkpoint that is the
# only copy of the design state -- destroying the thing being resumed from,
# halfway through the run that depends on it. Distinct name by default.
NAME="${NAME:-ant_codesign_lagrangian_b${BUDGET}_grid${GRID}_2000M}"
if [ -n "$RESUME" ] && [ "$NAME" = "ant_codesign_lagrangian_b${BUDGET}_grid${GRID}_2000M" ]; then
  NAME="${NAME}_r$(date -u +%m%d%H%M)"
fi
CKPT="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/$NAME"

if [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { echo "ABORT: RESUME=$RESUME is not a directory"; exit 1; }
  # No design_state.npz means the policy would resume onto a FRESH
  # distribution -- a silent restart of the search dressed as a continuation.
  [ -f "$RESUME/design_state.npz" ] || {
    echo "ABORT: no design_state.npz in $RESUME -- resuming would restart the"
    echo "design search from its init while the policy continued. Refusing."; exit 1; }
  echo "=== RESUMING from $RESUME ==="
  echo "    design_lr=$DESIGN_LR  steps=$STEPS  design_tmax=$DESIGN_TMAX"
  set -- --restore_checkpoint_path "$RESUME"
else
  set --
fi
echo "=== co-design: $STEPS steps, 1024 envs, design_lr=$DESIGN_LR -> $CKPT ==="
echo "    schedule: design frozen until $STEPS_BEFORE, chop every $CHOP_FREQ, fine-tune last $STEPS_AFTER; $NUM_EVALS evals x $DESIGN_UPDATES updates"
echo "    $PENALIZER b$BUDGET  lr=$MULT_LR cap=$MULT_MAX init=$MULT_INIT  grid $GRID  flip $FLIP_COST  finish line  float32"

# python -u is NOT optional: Python block-buffers stdout when redirected, and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
#
# PENALIZER. PPO-Lagrangian at budget 25 (per-episode total, adaptive
# horizon) -- the recipe the 2026-09-17..21 single-body arms converged on. For
# the FAITHFUL Schaff baseline (unconstrained, cost logged but never shaping
# anything) submit with PENALIZER=none and nothing else; the design objective
# is speed either way and does not read cost.
#
# action_repeat / episode_length / discounting / healthy_reward /
# terminate_on_flip / terminate_on_goal are deliberately NOT passed: they
# resolve from _ROBOT_DEFAULTS and print on the first line of the log. Pinning
# them here would freeze this run to whatever was correct the day it was
# written.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer "$PENALIZER" --safety_budget "$BUDGET" --adaptive_budget_horizon \
  --lagrangian_multiplier_lr "$MULT_LR" --lagrangian_multiplier_max "$MULT_MAX" \
  --initial_lagrange_multiplier "$MULT_INIT" "$LPC_FLAG" \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --foot_hazard_grid "$GRID" --hazard_cost_shape linear --ground_contact_eps 0.02 --finish_line \
  --matmul_precision highest --solver_iterations 4 --solver_ls_iterations 8 \
  --no-corridor_walls --terminate_out_of_bounds --exit_cost 50 --boundary_cost_weight 0 \
  --design_optimization \
  --design_objective "$DESIGN_OBJECTIVE" --design_cost_weight "$DESIGN_COST_WEIGHT" \
  --flip_cost "$FLIP_COST" \
  --num_morphologies 16 \
  --design_components 8 \
  --design_updates_per_eval "$DESIGN_UPDATES" \
  --chop_freq "$CHOP_FREQ" \
  --steps_before_design_update "$STEPS_BEFORE" \
  --steps_after_design_update "$STEPS_AFTER" \
  --num_envs 1024 --num_minibatches 32 \
  --design_lr "$DESIGN_LR" \
  --num_timesteps "$STEPS" \
  --design_tmax "$DESIGN_TMAX" \
  --num_evals "$NUM_EVALS" \
  --checkpoint_logdir "$CKPT" "$@"
status=$?

echo
echo "=== exit status: $status ==="
echo "--- checkpoints ---"
ls -1 "$CKPT" 2>&1 | tail -5
du -sh "$CKPT" 2>&1
echo "--- design state written? (a resume is only possible if this is non-empty) ---"
ls -1 "$CKPT"/*/design_state.npz 2>&1 | tail -3
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. The result is the FINAL design/mode_0..6 line plus the checkpoint."
  echo "Next: scripts/eval_morphology.py on the mode design, against"
  echo "checkpoints/vast/vast_morph30M (the fixed-uniform predecessor) at equal"
  echo "steps, and re-run the held-out-seed test -- the coupled loop should NOT"
  echo "reproduce the 29% unseen-body failure rate the decoupled pipeline had."
else
  echo "FAILED (status $status). Resume from the newest checkpoint above and"
  echo "pass --design_tmax 300000000 with the REMAINING budget as"
  echo "--num_timesteps -- see the RESUMING section in this file's header."
fi

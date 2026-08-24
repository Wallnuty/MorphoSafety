#!/bin/bash
# Schaff et al. (ICRA 2019) co-design: train the DESIGN DISTRIBUTION jointly
# with the policy. 300M env steps, ant on minefield, Saute at budget 500.
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
# the first eval line and correct the --time on the next submission. The chain:
# a rented RTX 3090 measured 7755 training/sps for ant/minefield at 1024 envs
# (logs/vast/throughput.log), and morphology batching cost ~17% on the laptop
# (403 -> 335 sps) from heterogeneous per-lane geometry. Call it ~6400:
#
#   300M / 6400          ~= 13.0 h training
#   10 evals x ~360 s    ~=  1.0 h   (320 episodes each, 625-decision unrolls)
#   host model rebuilds  ~=  0.1 h   (80 iterations x 16 designs x ~0.2 s)
#   ----------------------------------
#                        ~= 14 h.  --time 24h is ~1.7x margin; bigbatch caps
#                                  at 72h so there is room to raise it.
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
# row reports upd/Ms = 1562, identical at 512, 1024, 2048 and 4096 envs, while
# sps goes 5917 -> 7755 at 1024. So this is a 31% speedup at IDENTICAL learning
# density, not a different learning curve. batch_size and unroll_length are the
# two that would genuinely change it, and both are left alone.
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
#   design/score_mean           negated time-to-goal in DECISIONS, so -212
#                               means 212 decisions to arrive. Higher = better.
#   design/fitness_decisions    the same number un-negated, for sanity.
#   design/arrival_rate         fraction of episodes that reached the goal.
#                               0.0 early is expected and not a failure.
#   design/UNSCORED_DESIGNS     should be absent. If it appears, iterations are
#                               too short -- see WHY 8 DESIGN UPDATES above.
#   design/SKIPPED_UPDATE       appears when fewer than 2 designs were scored;
#                               a lone sample standardises to 0 and carries no
#                               gradient.
#   eval/episode_cost           LOGGED, NOT OPTIMISED BY THE SEARCH. The score
#                               is time-to-arrival only. Safety enters through
#                               Saute shaping the reward the policy sees, not
#                               through the design objective. That is a
#                               deliberate staging choice, not an oversight.
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
#SBATCH --time=24:00:00
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
# 55 = 47 (lidar-only) + 7 genes + 1 Saute budget scalar. A silent 71 would
# mean --foot_obstacle_obs leaked back on and this is no longer the lidar-only
# run that was asked for; a silent 47 would mean morphology conditioning is not
# reaching the observation at all, which would train a policy that cannot see
# which body it is driving -- the whole method, gone, with a log that looks
# completely normal.
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

kw = dict(T.robot_env_kwargs('ant'))
kw['foot_obstacle_obs'] = False
kw['morphology_conditioning'] = True
w = Minefield(**kw).observation_size
assert w == 47 + morphology.NUM_GENES, f'expected 54 pre-Saute, got {w}'
print(f'PASS: obs {w} + 1 Saute = {w+1}, design checkpointing present')
" || { echo "=== code preflight FAILED, nothing spent ==="; exit 1; }
echo

# 0.75 of 24 GB. The card has room; this is the same fraction the other
# cluster jobs use, left alone so a VRAM problem here is comparable to theirs.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

CKPT="${SLURM_SUBMIT_DIR:-$PWD}/checkpoints/ant_codesign_lidar_300M"
echo "=== co-design: 300M steps, 80 design updates, 1024 envs, checkpoints -> $CKPT ==="

# python -u is NOT optional: Python block-buffers stdout when redirected, and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
#
# PENALIZER. Saute at budget 500 matches every other run this project has done
# on minefield, so cost here is comparable to them. For the FAITHFUL Schaff
# baseline (unconstrained, cost logged but never shaping anything) change the
# three saute flags to `--penalizer none` and nothing else -- the design
# objective is time-to-arrival either way and does not read cost.
#
# action_repeat / episode_length / discounting / healthy_reward /
# terminate_on_flip / terminate_on_goal are deliberately NOT passed: they
# resolve from _ROBOT_DEFAULTS and print on the first line of the log. Pinning
# them here would freeze this run to whatever was correct the day it was
# written.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer saute --safety_budget 500 --saute_terminate \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --design_optimization \
  --design_objective time \
  --num_morphologies 16 \
  --design_components 8 \
  --design_updates_per_eval 8 \
  --chop_freq 60000000 \
  --steps_after_design_update 30000000 \
  --num_envs 1024 --num_minibatches 32 \
  --num_timesteps 300000000 \
  --num_evals 11 \
  --checkpoint_logdir "$CKPT"
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

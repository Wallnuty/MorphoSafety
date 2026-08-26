#!/bin/bash
# Continue the nominal (single-body) ant on minefield for another 100M steps,
# resuming from the finished 50M lidar-only Saute run.
#
#   source : checkpoints/ant_minefield_saute500_lidar_wide/000050176000
#            50M steps, episode_cost 84.88, reward 20.52
#   this   : +100M on top, same config in every respect
#
# "penalty 500" is --safety_budget 500. --saute_penalty is a DIFFERENT flag and
# it is 0 here, as it was in the source run: budget exhaustion terminates the
# episode (--saute_terminate) but costs no reward. That is worth knowing before
# reading the result, because it is why the budget has never bound very hard --
# see the budget-50 experiment, which changed nothing.
#
# ========================= GET THE CHECKPOINT THERE =======================
#
# It is 6.4 MB and lives on the LAPTOP, not the cluster. From the laptop:
#
#   scp -r checkpoints/ant_minefield_saute500_lidar_wide/000050176000 \
#       ssnyman@146.141.21.100:~/MorphoSafety/checkpoints/ant_minefield_saute500_lidar_wide/
#
# (mkdir -p the destination first). Then submit from ~/MorphoSafety.
#
# --restore_checkpoint_path MUST BE ABSOLUTE. train_ppo passes it to orbax
# unresolved, and orbax raises "Checkpoint path should be absolute" -- the
# script below builds it from $SLURM_SUBMIT_DIR so you cannot pass a relative
# one by accident.
#
# ======================= WHAT A RESUME DOES AND DOES NOT ==================
#
# RESTORED: network params, normalizer stats, penalizer params, optimizer
#           (Adam) moments -- all four elements of the checkpoint tuple.
# NOT RESTORED: env_steps. It is not in the checkpoint, so THE NEW LOG COUNTS
#           FROM ZERO. `step=10035200` in this run means 60.2M cumulative. The
#           checkpoint directory names restart too, which is why this writes to
#           a DIFFERENT directory -- at 100M/10 evals the new run's step
#           numbers collide exactly with the source run's (both land on
#           10,035,200), and pointing both at one directory would overwrite the
#           50M baseline with a mid-training checkpoint.
#
# THE STEP-0 EVAL IS THE RESTORE CHECK, and it is free. It runs the restored
# weights before any gradient step, so it should reproduce the END of the
# source run:
#
#       expect  eval/episode_cost ~= 85     eval/episode_reward ~= 20.5
#       if you see cost ~= 20 and reward ~= 0.5, THE RESTORE SILENTLY FAILED
#       and this is training from scratch -- kill it.
#
# ============================== SIZING ====================================
#
# EXTRAPOLATED, not measured for this exact config -- read training/sps off the
# first eval and correct --time next time. mscluster53 gave 6069-6196 sps on
# the heavier co-design workload at 1024 envs; a rented 3090 gave 6617-6723 for
# this single-body config at 512. Taking 6000 as a conservative floor:
#
#   100M / 6000        ~= 4.6 h training
#   11 evals x ~350 s  ~= 1.1 h   (320 episodes each, 625-decision unrolls)
#   ----------------------------------
#                      ~= 5.7 h.  --time 12h is ~2.1x margin.
#
# WHY 512 ENVS AND NOT 1024. 1024 is free in learning terms (num_minibatches
# cancels out of updates-per-env-step; the 3090 sweep shows upd/Ms = 1562 at
# both) and worth ~8% wall clock. It is not taken here because this run
# CONTINUES a 50M run that used 512, and keeping the data-gathering identical
# across the join is worth more than 8%. Override with NUM_ENVS=1024
# NUM_MINIBATCHES=32 if you disagree.
#
#SBATCH --job-name=ant-lidar-100M
#SBATCH --output=ant_lidar_resume_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=12:00:00
set -u

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
RESUME="${RESUME:-$ROOT/checkpoints/ant_minefield_saute500_lidar_wide/000050176000}"
STEPS="${STEPS:-100000000}"
NUM_ENVS="${NUM_ENVS:-512}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
CKPT="$ROOT/checkpoints/ant_minefield_saute500_lidar_150M"

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

# PREFLIGHT 1: a GPU that JAX can actually see. Assert on device PLATFORM, not
# on jax.devices() merely not raising -- a missing CUDA plugin falls back to
# CPU silently, and a cluster job once burned eight hours that way.
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

# PREFLIGHT 2: the checkpoint is there, and the env this checkout builds is the
# shape that checkpoint expects.
#
# This is the one that matters. --foot_obstacle_obs DEFAULTED ON for one day
# (2026-08-24) and OFF again on 2026-08-25; if this checkout predates the
# second change, the ant's default observation is 63 rather than 47 and the
# flags below would build a 64-wide env for a 48-wide policy. The run passes
# --no-foot_obstacle_obs explicitly for exactly that reason, and this asserts
# the result rather than trusting it.
echo "=== checkpoint / observation preflight ==="
RESUME="$RESUME" python -c "
import os, sys
from pathlib import Path
import orbax.checkpoint as ocp
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T

p = Path(os.environ['RESUME'])
assert p.is_absolute(), f'ABORT: --restore_checkpoint_path must be absolute, got {p}'
assert p.is_dir(), (
    f'ABORT: {p} does not exist. Two likely causes:\n'
    f'  1. it was never copied across -- the checkpoint lives on the LAPTOP, '
    f'see the scp in this file\'s header; or\n'
    f'  2. YOU SUBMITTED FROM THE WRONG DIRECTORY. This path is built from '
    f'$SLURM_SUBMIT_DIR, so `sbatch` run inside cluster/ looks in '
    f'MorphoSafety/cluster/checkpoints/ instead of MorphoSafety/checkpoints/. '
    f'The co-design job 46267 was submitted that way. cd to the repo root '
    f'first, or pass RESUME=<absolute path>.')
d = ocp.PyTreeCheckpointer().restore(str(p))
w = T.checkpoint_obs_width(d[1]['policy'])
layers = T.checkpoint_policy_layers(d[1]['policy'])
assert w == 48, f'ABORT: checkpoint is {w} wide, expected 48 (47 lidar-only + 1 Saute)'
assert layers == (256,)*4, f'ABORT: checkpoint policy is {layers}, expected (256,)*4'

kw = dict(T.robot_env_kwargs('ant')); kw['foot_obstacle_obs'] = False
env_w = Minefield(**kw).observation_size
assert env_w == 47, (
    f'ABORT: this checkout builds a {env_w}-wide env for the flags this run '
    f'passes, not 47. The policy needs 47+1. git pull.')
print(f'PASS: checkpoint {w} wide, policy {layers}, env {env_w} + 1 Saute = {env_w+1}')
" || { echo "=== preflight FAILED, nothing spent ==="; exit 1; }
echo

[ -d "$CKPT" ] && { echo "=== REFUSING: $CKPT already exists, move it aside ==="; exit 1; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
echo "=== +$STEPS steps from $RESUME -> $CKPT ==="

# python -u is NOT optional: Python block-buffers stdout when redirected and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
#
# --no-foot_obstacle_obs is passed EXPLICITLY even though it is now the
# default, because that default has moved twice in two days. Pinning it here
# means this run is the same experiment whichever way the repo is sitting.
# --policy_hidden_layer_sizes likewise: 256x4 is the default now, and it is the
# width the checkpoint was trained at, so it is stated rather than inherited.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer saute --safety_budget 500 --saute_terminate \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --corridor_walls --boundary_cost_weight 0 \
  --policy_hidden_layer_sizes 256 256 256 256 \
  --num_envs "$NUM_ENVS" --num_minibatches "$NUM_MINIBATCHES" \
  --num_timesteps "$STEPS" --num_evals 11 \
  --restore_checkpoint_path "$RESUME" \
  --checkpoint_logdir "$CKPT"
status=$?

echo
echo "=== exit status: $status ==="
echo "--- checkpoints ---"
ls -1 "$CKPT" 2>&1 | tail -5
du -sh "$CKPT" 2>&1
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. Read in this order:"
  echo "  1. THE STEP-0 EVAL. cost ~85 / reward ~20.5 means the restore worked."
  echo "     cost ~20 / reward ~0.5 means it did not and this run is void."
  echo "  2. FINAL eval/episode_cost against 84.88. The source run was still"
  echo "     falling at 50M (87.03 -> 84.88 over its last 5M), so the question"
  echo "     is whether it keeps going or flattens -- a floor that survives"
  echo "     150M total is a property of the reward, not of training length."
  echo "  3. eval/episode_reward should stay ~20.5-21.3. If cost falls while"
  echo "     reward falls with it, that is the policy buying safety by not"
  echo "     arriving, which is not the result we want."
else
  echo "FAILED (status $status). Resume from the newest checkpoint above; note"
  echo "the step counter restarts at 0 again, so subtract accordingly."
fi

#!/bin/bash
# Throughput sweep: what should the 15-hour ant baseline actually be configured
# with? Run this BEFORE cluster/ant_unconstrained_baseline.sbatch.
#
# The baseline job is currently sized at ~15h from a config whose two biggest
# knobs were never chosen on evidence:
#
#   * --num_envs 256 was sized for a 6GB laptop card, not a 24GB 3090.
#   * ant.xml declares integrator="RK4" purely because safety-gymnasium's copy
#     did. RK4 does FOUR force evaluations per step against one for
#     implicitfast/euler, and physics is ~90% of per-decision cost (decomposed
#     from two measured throughputs: ~2.71 ms per physics step against ~3.02 ms
#     fixed per-decision overhead). CPU stepping measured 3.5x cheaper.
#
# Spending ~1.5h here to pick those two is cheap insurance against spending 15h
# at the wrong settings -- and the sweep doubles as a crash test, since the ant
# segfaults on the laptop at full size and has only ever passed here once.
#
# ARM A: --num_envs 256 -> 512. FREE AND SAFE. validate() requires num_envs to
# divide batch_size*num_minibatches (32*16=512), so 512 is the ceiling that
# needs no PPO hyperparameter change, and total data per update is IDENTICAL
# either way (512 x unroll_length 10 = 5120 transitions). The only difference
# is 512 envs x 1 unroll instead of 256 envs x 2 sequential unrolls -- half the
# sequential rollout depth per training step, and slightly MORE decorrelated
# data, which if anything helps. Going past 512 would mean raising batch_size
# or num_minibatches, which changes the optimization rather than just its
# speed, so it is deliberately not tested here.
#
# ARM B: RK4 -> implicitfast. NOT FREE -- this is a research decision, which is
# why the sweep measures it rather than the default adopting it. Measured on
# CPU at the shipped timestep of 0.01: passive settling is near-identical
# (equilibrium torso height 0.159 m RK4 vs 0.161 m implicitfast, max trajectory
# deviation 0.0225 m, no instability) but a scripted gait travels 1.010 m under
# RK4 against 0.770 m under implicitfast, ~25% less. That is a genuine dynamics
# change, not roundoff. Two things make it worth measuring anyway: point.xml
# already uses euler, so the ant's RK4 is the anomaly and overriding it makes
# the two robots CONSISTENT; and if it really is ~3x, it converts the 15h
# baseline into something like 5h, which changes what experiments are
# affordable at all. Not tested here but worth knowing: standard practice pairs
# a single-evaluation integrator with a smaller timestep, and implicitfast at
# dt=0.005 would still be ~2x cheaper than RK4 at dt=0.01 and more accurate
# than either.
#
# WHAT TO COMPARE. training/sps is the throughput answer, but do NOT pick on
# speed alone -- also compare eval/episode_cost and eval/episode_reward across
# arms. Arms 1 and 2 differ only in parallelism and MUST agree on those within
# noise; if they do not, something is wrong with the larger batch rather than
# faster. Arm 3 changes the physics, so it is EXPECTED to differ -- the question
# is whether it differs enough to matter for a study about safety, where cost
# comes from contact with hazards and vases.
#
# Each arm is 500k env steps with 2 evals, matching the stage-2 smoke test so
# the numbers are comparable to what is already recorded. Sequential, not an
# array job: one GPU, and concurrent JAX processes would contend for it and
# corrupt the measurement.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_throughput_sweep.sbatch
# Watch: squeue --me     Read: ant_sweep_%j.log
#SBATCH --job-name=ant-sweep
#SBATCH --output=ant_sweep_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=04:00:00
set -u
echo "=== node ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-safety-gym
python --version
pip show jax mujoco 2>&1 | grep -E "^Name|^Version"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# python -u is NOT optional -- a laptop run reached 573k steps and left a
# COMPLETELY EMPTY log, because Python block-buffers stdout when redirected and
# a segfault never flushes it.
#
# action_repeat / episode_length / discounting are deliberately NOT passed:
# they resolve from _ROBOT_DEFAULTS for --robot ant and each arm prints the
# resolved values on its first line, so the log records what actually ran.
run_arm () {
  local label="$1"; shift
  echo
  echo "==================== ARM: $label ===================="
  echo "cmd: $*"
  date
  # Peak VRAM is not readable from inside the run (MEM_FRACTION preallocates a
  # fixed pool, so nvidia-smi reports the pool, not demand) -- the useful
  # signal is simply whether the arm survives.
  time python -u -m mjx_safety_gym.algorithms.train_ppo \
    --robot ant --penalizer none \
    --num_timesteps 500_000 --num_evals 2 --no_checkpoint "$@"
  echo "--- $label exit status: $? ---"
}

run_arm "1. baseline      (num_envs=256, RK4)"          --num_envs 256
run_arm "2. wider         (num_envs=512, RK4)"          --num_envs 512
run_arm "3. wider + cheap (num_envs=512, implicitfast)" --num_envs 512 --integrator implicitfast

echo
echo "=============================================================="
echo "SUMMARY -- training/sps per arm (higher is better):"
echo "=============================================================="
grep -h "training/sps" ant_sweep_*.log 2>/dev/null | tail -3 || true
echo
echo "How to decide:"
echo "  * Arm2 vs Arm1: pure parallelism win, no dynamics change. If Arm2 is"
echo "    faster, adopt --num_envs 512 unconditionally. Its cost/reward MUST"
echo "    match Arm1 within noise -- if not, investigate before adopting."
echo "  * Arm3 vs Arm2: isolates the integrator. This one is a judgement call,"
echo "    not an automatic adopt. Weigh the speedup against the ~25% gait-"
echo "    displacement difference measured on CPU, and against deviating from"
echo "    safety-gymnasium's ant.xml. If adopted, EVERY prior ant cost number"
echo "    becomes incomparable and --safety_budget must be re-derived again."
echo "  * Then update cluster/ant_unconstrained_baseline.sbatch with the winner"
echo "    and reproject its --time from the new training/sps before submitting."
echo "  * Any arm exiting 139 = the laptop segfault class reaching the cluster."
echo "=== done ==="

#!/bin/bash
# !! SUPERSEDED 2026-08-10 -- DO NOT SUBMIT WITHOUT READING THIS !!
# This job targets `--robot ant` on the GoToGoal task, which was measured this
# night to be UNLEARNABLE for the ant: over 16 seeds, a scripted gait that
# walks several metres earns +0.0781 +- 0.6715 return, i.e. statistically zero,
# because goals sit in a uniformly random direction and locomotion alone is
# worth nothing. The ant also never travels far enough to touch an obstacle --
# measured hazard cost was 0.0 in every mode -- so any cost number this job
# produces describes a near-stationary robot and is the wrong number to
# calibrate --safety_budget from.
#
# The replacements are `--task run` (envs/run_forward.py, reward linear in
# +x displacement) and `--robot ant_gym` (envs/xmls/ant_gym.xml, the
# Gym-proportioned ant: ~20x the travel per episode, and the first config to
# register non-zero hazard cost at all).
#
# Everything below is kept because the sizing arithmetic, the python -u
# rationale and the checkpoint-path validation are all still correct -- only
# the task/robot choice is stale. Re-point --robot/--task before submitting,
# and note that action_repeat is back to 4, so the step accounting in any
# header below that assumes 10 is wrong by 2.5x.
#
# Throughput + GPU-utilization sweep. Run this BEFORE
# cluster/ant_unconstrained_baseline.sbatch, which is currently sized at ~15h
# from settings that were never chosen on evidence.
#
# THE QUESTION THIS EXISTS TO ANSWER: how much of the 3090 are we actually
# using? Note that XLA_PYTHON_CLIENT_MEM_FRACTION cannot tell you -- it
# PREALLOCATES a fixed pool, so nvidia-smi reports that pool as "used"
# regardless of real demand (the laptop peaked at 979 MiB of genuine VRAM at
# 1024 envs). Memory has never been this workload's constraint. So every arm
# below samples utilization.gpu, which is independent of the preallocation and
# reports whether the SMs are actually busy.
#
# THREE THINGS ARE BEING VARIED:
#
# A. num_envs 256 -> 2048. 256 was sized for a 6GB laptop card. Raising it is
#    the direct test of whether the GPU is starved of parallel work. An
#    intermediate 512-env arm was deliberately dropped: 512 is the ceiling
#    reachable without touching num_minibatches, but it answers a strict subset
#    of what the 2048 arm answers and cost ~45 min for almost no extra
#    information. If 2048 turns out to destabilise or OOM, 512 is the obvious
#    fallback to add back.
#
# B. num_minibatches 16 -> 64, ONLY as the mechanism for reaching 2048 envs.
#    validate() requires num_envs to divide batch_size*num_minibatches, so 512
#    is the ceiling at the current 32*16. Reaching 2048 needs that product to
#    be 2048, and HOW you get there matters:
#
#      num_envs  num_mb  transitions/grad_step  grad_steps @37.5M
#           256      16                    320             23,438
#          2048      64                    320             23,438   <- via num_minibatches
#          2048      16                      -             REJECTED by validate()
#
#    Raising num_minibatches keeps transitions per gradient step at
#    batch_size*unroll_length = 320 AND keeps the total gradient-step count
#    identical (4x fewer training steps, 4x more gradient steps in each, so it
#    cancels). Raising batch_size to 128 instead would have reached 2048 too,
#    but with 1280 transitions per gradient step -- a 4x larger gradient batch,
#    a different optimizer regime, and a different experiment. So this sweep
#    scales via num_minibatches deliberately: the ONLY thing that changes is
#    that a training step sees 2048 distinct envs instead of 256 sampled twice
#    as deep, which if anything decorrelates the data slightly better.
#
# C. RK4 -> implicitfast. RK4 does FOUR force evaluations per step against one,
#    and physics is ~90% of per-decision cost (decomposed from two measured
#    throughputs: ~2.71 ms per physics step against ~3.02 ms fixed per-decision
#    overhead); CPU stepping measured 3.5x cheaper. ant.xml declares RK4 only
#    because safety-gymnasium's copy did -- point.xml already takes MuJoCo's
#    Euler default. NOT free, and NOT an automatic adopt: at the shipped
#    timestep of 0.01 passive settling is near-identical (equilibrium torso
#    height 0.159 m RK4 vs 0.161 m implicitfast, 0.0225 m max trajectory
#    deviation, no instability) but a scripted gait travels 1.010 m under RK4
#    against 0.770 m under implicitfast, ~25% less. That is a real dynamics
#    change in a project whose subject is contact-driven cost.
#
# num_timesteps is NOT constant across arms, on purpose. env steps per training
# step scale with num_minibatches (51,200 at mb=16, 204,800 at mb=64), so a
# fixed budget would give the 2048-env arms only a couple of training steps and
# a training/sps number that is pure compile noise. Each arm instead gets ~20
# training steps: 1M for the mb=16 arms, 4M for the mb=64 arms.
#
# Sequential, not an array job: one GPU, and concurrent JAX processes would
# contend for it and corrupt both the timing and the utilization sampling.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_throughput_sweep.sbatch
# Watch: squeue --me     Read: ant_sweep_%j.log
#SBATCH --job-name=ant-sweep
#SBATCH --output=ant_sweep_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=08:00:00
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
# PREFLIGHT: abort if JAX cannot see a GPU. Job 40227 landed on mscluster65
# whose GPU0 was broken; JAX printed a WARNING and fell back to CPU, and the
# job burned its whole 8h allocation producing zero data points. Assert on
# device PLATFORM -- a silent CPU fallback otherwise reads as a pass. See the
# long note in cluster/ant_gym_run_baseline.sbatch.
echo "=== GPU preflight ==="
python -c "
import jax
gpus = [d for d in jax.devices() if d.platform == 'gpu']
print('JAX devices:', jax.devices())
assert gpus, 'ABORT: no GPU -- silently fell back to CPU. Node likely broken; resubmit.'
print(f'PASS: {len(gpus)} GPU device(s)')
" || { echo "=== GPU preflight FAILED, aborting ==="; exit 1; }
echo
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Deliberately NOT mktemp -d. If the job hits its wall time mid-arm, Slurm
# SIGKILLs everything and that arm's summarize_util never runs -- with the
# samples in /tmp on a node you no longer have, they are gone. Writing them
# beside the log means a timed-out run still leaves the raw utilization data
# for the arm that was interrupted, which is usually the interesting one.
UTILDIR="ant_sweep_util_${SLURM_JOB_ID:-manual}"
mkdir -p "$UTILDIR"

# Sample real GPU business every 3s. utilization.gpu is the percentage of time
# any kernel was resident over the sampling interval -- it is NOT occupancy, so
# 100% does not prove the card is saturated, but a LOW number is conclusive
# evidence it is starved. Read it that way round.
sample_util () {
  while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used \
               --format=csv,noheader,nounits 2>/dev/null || true
    sleep 3
  done
}

summarize_util () {
  local f="$1"
  local n; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  if [ "$n" -lt 3 ]; then echo "    gpu_util: too few samples ($n)"; return; fi
  local mean p50 p90 max
  mean=$(cut -d, -f1 "$f" | awk '{s+=$1} END{printf "%.0f", s/NR}')
  p50=$(cut -d, -f1 "$f" | sort -n | awk -v n="$n" 'NR>=int(n*0.5){print; exit}')
  p90=$(cut -d, -f1 "$f" | sort -n | awk -v n="$n" 'NR>=int(n*0.9){print; exit}')
  max=$(cut -d, -f1 "$f" | sort -n | tail -1)
  echo "    gpu_util: mean=${mean}% p50=${p50}% p90=${p90}% max=${max}%  (n=$n samples)"
  echo "    (memory.used is meaningless here -- MEM_FRACTION=0.75 preallocates ~18GB)"
}

# python -u is NOT optional -- a laptop run reached 573k steps and left a
# COMPLETELY EMPTY log, because Python block-buffers stdout when redirected and
# a segfault never flushes it.
#
# action_repeat / episode_length / discounting are deliberately NOT passed:
# they resolve from _ROBOT_DEFAULTS for --robot ant and each arm prints the
# resolved values on its first line, so the log records what actually ran.
run_arm () {
  local tag="$1"; local label="$2"; local steps="$3"; shift 3
  local uf="$UTILDIR/$tag"
  echo
  echo "==================== ARM $tag: $label ===================="
  echo "steps=$steps  extra args: $*"
  date
  sample_util > "$uf" &
  local sampler=$!
  time python -u -m mjx_safety_gym.algorithms.train_ppo \
    --robot ant --penalizer none \
    --num_timesteps "$steps" --num_evals 2 --no_checkpoint "$@"
  local st=$?
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  echo "--- ARM $tag exit status: $st ---"
  summarize_util "$uf"
}

run_arm 1 "num_envs=256  mb=16  RK4          (current baseline)" 1000000 \
        --num_envs 256  --num_minibatches 16
run_arm 2 "num_envs=2048 mb=64  RK4          (max parallelism)"  4000000 \
        --num_envs 2048 --num_minibatches 64
run_arm 3 "num_envs=2048 mb=64  implicitfast (parallel + cheap)" 4000000 \
        --num_envs 2048 --num_minibatches 64 --integrator implicitfast

echo
echo "=============================================================="
echo "HOW TO READ THIS"
echo "=============================================================="
echo "Compare training/sps across arms, and read each arm's gpu_util with it."
echo
echo "  * LOW gpu_util in arm 1 (say <50%) is the headline: the card is starved"
echo "    and num_envs=256 is leaving the 3090 idle. Arms 2 and 3 then say how"
echo "    much of that is recoverable."
echo "  * HIGH gpu_util everywhere with flat training/sps means we are already"
echo "    compute-bound and more parallelism buys nothing -- in which case the"
echo "    integrator (arm 3) is the ONLY remaining lever."
echo "  * Arms 1 and 2 differ ONLY in parallelism, never in optimization: 320"
echo "    transitions per gradient step and 23,438 total gradient steps in"
echo "    both. So their eval/episode_cost and eval/episode_reward MUST agree"
echo "    within noise. If they do not, the larger batch is doing something"
echo "    unintended and must be understood before adopting."
echo "  * Arm 3 vs arm 2 isolates the integrator, and is a JUDGEMENT CALL, not"
echo "    an automatic adopt. Weigh the speedup against the ~25% gait-"
echo "    displacement difference. If adopted, every prior ant cost number"
echo "    becomes incomparable AGAIN and --safety_budget must be re-derived."
echo "  * Watch for arm 2/3 compile time specifically. Laptop compile went 46s"
echo "    at 256 envs to 264s at 1024; cluster CPUs are 2012-era Xeons and"
echo "    compile slower, so 2048 envs could cost 15-30 min before the first"
echo "    step. That is a one-off per shape, cached in ~/.cache/jax, and it is"
echo "    a real cost to weigh for short runs but irrelevant for a 15h one."
echo "  * Any arm exiting 139 is the laptop segfault class reaching the cluster."
echo "    Arm 2/3 are the most likely to OOM or destabilise -- 2048 envs has"
echo "    never been run for the ant anywhere (the laptop got host-RAM"
echo "    OOM-killed at 2048; this node has 128GB, so it should be fine)."
echo
echo "THEN: update cluster/ant_unconstrained_baseline.sbatch with the winning"
echo "num_envs/num_minibatches (and integrator, if adopted) and reproject its"
echo "--time from the new training/sps before submitting the real run."
echo "=== done ==="

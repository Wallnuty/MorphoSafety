#!/bin/bash
# WHAT num_envs / num_minibatches SHOULD THE CLUSTER RUNS USE?
#
# Sweeps parallelism on ant_gym + RunForward and reports, per arm: compile
# time, steady-state training/sps, GPU utilisation percentiles, and PEAK VRAM.
# Supersedes cluster/ant_throughput_sweep.sbatch, which targets --robot ant on
# the goal task at the old action_repeat=10 default and never produced a single
# data point (it landed on a node with a dead GPU -- see the preflight below).
#
# ============================ HOW THE ARMS ARE BUILT ========================
#
# env_step_per_training_step = batch_size * unroll_length * num_minibatches
#                              * action_repeat
#                            = 32 * 10 * N * 4
#
# and brax requires num_envs to divide batch_size * num_minibatches. So
# parallelism is raised by moving num_minibatches and num_envs TOGETHER, never
# batch_size:
#
#   arm  num_envs  num_minibatches  env_steps/iter  batch*mb  divides?
#     A       256               16          20,480       512   512/256 = 2
#     B      1024               32          40,960     1,024   1024/1024 = 1
#     C      2048               64          81,920     2,048   2048/2048 = 1
#     D      4096              128         163,840     4,096   4096/4096 = 1
#
# Every arm runs the SAME 30 iterations, so --num_timesteps scales with
# env_steps/iter. That makes the arms comparable on wall-clock per iteration,
# which is the number that actually decides how long a real run takes.
#
# NOTE what this does and does not tell you. Holding batch_size at 32 while
# num_minibatches rises keeps transitions-per-gradient-step at 320, so the
# OPTIMIZATION is unchanged arm to arm -- but each iteration consumes 8x more
# env steps at D than at A, so at a FIXED --num_timesteps budget the
# higher-parallelism arms take 8x fewer gradient steps. Faster wall-clock per
# env step is not automatically faster learning. Decide with both numbers:
# adopt the arm with the best sps only if you also raise --num_timesteps to
# keep the gradient-step count you actually want.
#
# ======================= WHY PREALLOCATION IS OFF HERE ======================
#
# XLA_PYTHON_CLIENT_PREALLOCATE=false, so nvidia-smi memory.used is the REAL
# working-set size rather than the ~18 GB the 0.75 fraction would grab up
# front. Without this the VRAM column is meaningless and there is no way to
# tell whether arm D has headroom left. The cost is some allocator overhead on
# every arm equally, so the sps COMPARISON stands while the absolute sps may
# read slightly under what a preallocating production run achieves. Re-measure
# the winner with preallocation on if you want its exact production rate.
#
# ============================== EVAL IS NOT SWEPT ===========================
#
# Evals are deliberately tiny (8 envs x 1 episode) so they barely touch the
# wall clock and this measures TRAINING throughput. That makes every
# eval/episode_* number in this log meaningless -- do not read them. Eval cost
# is already known separately: job 38961 measured epoch_eval_time 220-367 s at
# the full 32x10 width on the old ant, and ant_gym's episode is 2.5x longer
# (2500 vs 1000), so budget ~550-900 s per full-width eval.
#
# ================================== RUNTIME ================================
#
# 30 iterations per arm. If the GPU is already saturated at A, every arm takes
# about the same wall time and the sweep is short. If A is starved -- which is
# the hypothesis worth testing on a 24 GB 3090, since these defaults were sized
# for a 6 GB laptop -- the later arms process far more env steps in the same
# time, which is exactly the result being looked for.
#
#   A: 614,400 env steps    -- ~14 min at the measured 750 sps
#   D: 4,915,200 env steps  -- ~18 min if it scales 8x, ~1.8 h if it does not
#
# Plus compile, which grows with shape: the laptop went 46 s at 256 envs to
# 264 s at 1024, and cluster CPUs are 2012-era Xeons, so arms C/D could cost
# 15-30 min each before their first step. That is a one-off per shape, cached
# in ~/.cache/jax, and it is a real cost for short runs but irrelevant for a
# 15 h one -- the log reports it separately so it can be weighed, not hidden.
# --time 8h covers the pessimistic case.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_gym_throughput_sweep.sbatch
# Watch: squeue --me     Read: ant_gym_sweep_%j.log
#SBATCH --job-name=antgym-sweep
#SBATCH --output=ant_gym_sweep_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=08:00:00
set -u
echo "=== node ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc
echo
echo "=== env ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-safety-gym
python --version
pip show jax mujoco 2>&1 | grep -E "^Name|^Version"
echo

# PREFLIGHT: abort if JAX cannot see a GPU. Job 40227 landed on mscluster65
# whose GPU0 was broken; JAX printed a WARNING and fell back to CPU, and that
# job -- this sweep's predecessor -- burned its whole 8h allocation producing
# zero data points. Assert on device PLATFORM: a silent CPU fallback returns
# devices happily and would otherwise read as a pass.
echo "=== GPU preflight ==="
python -c "
import jax
gpus = [d for d in jax.devices() if d.platform == 'gpu']
print('JAX devices:', jax.devices())
assert gpus, 'ABORT: no GPU -- silently fell back to CPU. Node likely broken; resubmit.'
print(f'PASS: {len(gpus)} GPU device(s)')
" || { echo "=== GPU preflight FAILED, aborting ==="; exit 1; }
echo

export XLA_PYTHON_CLIENT_PREALLOCATE=false
UTILDIR="$(pwd)/sweep_util_${SLURM_JOB_ID:-local}"
mkdir -p "$UTILDIR"
ITERS=30

# utilization.gpu is the fraction of the sampling interval during which any
# kernel was resident. It is NOT occupancy, so 100% does not prove the card is
# saturated -- but a LOW number is conclusive evidence it is starved. Read it
# that way round. memory.used is real here because preallocation is off.
sample_util () {
  while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used \
               --format=csv,noheader,nounits 2>/dev/null || true
    sleep 2
  done
}

# Percentiles via sort rather than gawk's asort(), which mawk does not have --
# these nodes' awk is not guaranteed to be gawk and a sweep that silently
# prints nothing for its headline metric would be worse than useless.
summarize_util () {
  local f="$1"
  local n; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  if [ "$n" -lt 3 ]; then echo "    gpu_util: too few samples ($n)"; return; fi
  local mean p50 p90 max maxmem
  mean=$(cut -d, -f1 "$f" | awk '{s+=$1} END{printf "%.0f", s/NR}')
  p50=$(cut -d, -f1 "$f" | sort -n | awk -v n="$n" 'NR>=int(n*0.5){print; exit}')
  p90=$(cut -d, -f1 "$f" | sort -n | awk -v n="$n" 'NR>=int(n*0.9){print; exit}')
  max=$(cut -d, -f1 "$f" | sort -n | tail -1)
  maxmem=$(cut -d, -f2 "$f" | tr -d ' ' | sort -n | tail -1)
  echo "    gpu_util: mean=${mean}% p50=${p50}% p90=${p90}% max=${max}%  (n=$n)"
  echo "    peak VRAM: ${maxmem} MiB  of $(nvidia-smi --query-gpu=memory.total \
       --format=csv,noheader,nounits | head -1) MiB"
}

# python -u is NOT optional. A laptop run reached 573k steps (proven by
# checkpoints on disk) and left a COMPLETELY EMPTY log, because Python
# block-buffers stdout when redirected and a segfault never flushes it.
#
# action_repeat / episode_length / discounting are deliberately NOT passed:
# they resolve from _ROBOT_DEFAULTS for --robot ant_gym and the first line of
# each arm prints the resolved values, so the log records what actually ran.
run_arm () {
  local tag="$1"; local envs="$2"; local mb="$3"
  local per_iter=$(( 32 * 10 * mb * 4 ))
  local steps=$(( per_iter * ITERS ))
  local uf="$UTILDIR/$tag"
  echo
  echo "==================== ARM $tag: num_envs=$envs num_minibatches=$mb ===================="
  echo "env_steps/iter=$per_iter  x ${ITERS} iters = $steps env steps"
  date
  sample_util > "$uf" &
  local sampler=$!
  local t0=$SECONDS
  python -u -m mjx_safety_gym.algorithms.train_ppo \
    --robot ant_gym --task run --penalizer none \
    --num_envs "$envs" --num_minibatches "$mb" \
    --num_timesteps "$steps" --num_evals 3 \
    --num_eval_envs 8 --num_eval_episodes 1 \
    --no_checkpoint
  local st=$?
  local elapsed=$(( SECONDS - t0 ))
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  echo "--- ARM $tag exit status: $st, wall ${elapsed}s ---"
  if [ "$st" -ne 0 ]; then
    echo "    ARM $tag FAILED. If it is an OOM, this arm is past the card's"
    echo "    limit and the previous arm is your ceiling -- that is a RESULT,"
    echo "    not a broken sweep. Continuing to the next arm."
  fi
  summarize_util "$uf"
  echo "    NOTE: training/sps printed by brax at each eval is CUMULATIVE."
  echo "    Take the LAST eval's value, or difference the last two, to get the"
  echo "    steady-state rate -- the first one is compile-dominated."
}

run_arm A 256  16
run_arm B 1024 32
run_arm C 2048 64
run_arm D 4096 128

echo
echo "=============================================================="
echo "HOW TO READ THIS"
echo "=============================================================="
echo "1. STEADY-STATE training/sps per arm (last eval line of each arm)."
echo "   The measured baseline on this partition is 668-798 sps at arm A's"
echo "   settings, on the old ant. If B/C/D beat that substantially, arm A is"
echo "   starving a 24 GB card with settings sized for a 6 GB laptop."
echo
echo "2. gpu_util for arm A specifically. LOW (say <50%) is the headline: it"
echo "   means the card sat idle and more parallelism is free throughput."
echo "   HIGH everywhere with flat sps means we are already compute-bound and"
echo "   parallelism buys nothing -- stop here and the answer is 'keep 256'."
echo
echo "3. peak VRAM. If the best arm uses well under 24 GB there is headroom"
echo "   for MORE envs, or for the morphology work's --num_morphologies on top."
echo "   That headroom is the real reason to care: batched-morphology training"
echo "   needs num_envs to be an integer multiple of num_morphologies."
echo
echo "4. Compile time (wall minus the sps-implied training time). It grows"
echo "   sharply with shape and is paid once per shape. Irrelevant for a 15 h"
echo "   run, decisive for short ones."
echo
echo "5. Then RE-READ the caveat in this file's header: higher parallelism"
echo "   means fewer gradient steps per env step. Adopting arm C or D means"
echo "   raising --num_timesteps proportionally, or the run will be faster AND"
echo "   learn less. Do not skip this."
echo
echo "--- utilisation samples kept in $UTILDIR ---"
echo "=== done ==="

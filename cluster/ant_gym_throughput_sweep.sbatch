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
# EVERY ARM GETS THE SAME --num_timesteps, AND THAT IS FAIR. Per
# ppo/train.py:189, env steps per training step is
# batch_size*unroll_length*num_minibatches*action_repeat, while gradient steps
# per training step is num_updates_per_batch*num_minibatches. num_minibatches
# appears in both and CANCELS:
#
#   gradient steps = num_timesteps * num_updates_per_batch
#                    / (batch_size * unroll_length * action_repeat)
#                  = num_timesteps * num_updates_per_batch / 1280
#
# independent of num_minibatches AND of num_envs. So at a fixed
# --num_timesteps all four arms perform exactly the same number of gradient
# updates on exactly the same 320 transitions each. The ONLY thing that
# differs is how fast the hardware gets through them.
#
# That makes this sweep simple to act on: WALL-CLOCK PER ARM IS THE ANSWER.
# The fastest arm trains the same policy in less time, with no offsetting cost
# to hunt for. Just take the winner.
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
# 500,000 env steps per arm, and COMPILE TIME DOMINATES, which is why it is
# not lower. All this needs is one clean post-compile interval: brax reports
# training/sps cumulatively at each eval, so with --num_evals 3 you difference
# the last two and get the steady-state rate. sps is very stable once compiled
# (a laptop run measured 403 incremental against 383/403 cumulative), so ~1-2
# minutes of training per interval is already ample.
#
# num_training_steps_per_epoch = ceil(num_timesteps / (2 * env_steps_per_iter)),
# so 500k resolves to, and the measured interval lasts, at the 750 sps baseline:
#
#   arm  steps/iter  iters/epoch  actual steps  per interval  @750 sps
#     A      20,480           13       532,480       266,240      355 s
#     B      40,960            7       573,440       286,720      382 s
#     C      81,920            4       655,360       327,680      437 s
#     D     163,840            2       655,360       327,680      437 s
#
# Arms run up to 23% different ACTUAL step counts because of that ceil(). That
# is fine: sps is a RATE, and the arms are compared on rate, not on totals.
#
# Going below ~500k saves little, because compile is the bulk of each arm and
# is paid per shape regardless: the laptop went 46 s at 256 envs to 264 s at
# 1024, and these are 2012-era Xeons, so arms C/D could cost 15-30 min each
# before their first step. Cutting steps trims the cheap part and leaves the
# expensive part untouched. (It is cached in ~/.cache/jax, so a rerun of this
# sweep is much faster than the first.) --time 8h covers the pessimistic case.
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
STEPS=500000    # identical for every arm; see RUNTIME in the header

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
  local uf="$UTILDIR/$tag"
  echo
  echo "==================== ARM $tag: num_envs=$envs num_minibatches=$mb ===================="
  echo "env_steps/iter=$per_iter  total=$STEPS env steps"
  echo "gradient steps: $STEPS * num_updates_per_batch / 1280 -- SAME IN EVERY ARM"
  date
  sample_util > "$uf" &
  local sampler=$!
  local t0=$SECONDS
  python -u -m mjx_safety_gym.algorithms.train_ppo \
    --robot ant_gym --task run --penalizer none \
    --num_envs "$envs" --num_minibatches "$mb" \
    --num_timesteps "$STEPS" --num_evals 3 \
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
echo "5. THE DECISION IS JUST THE WALL CLOCK. Every arm ran the same 2M env"
echo "   steps and therefore the same number of gradient updates on the same"
echo "   320 transitions each (num_minibatches cancels -- see the header)."
echo "   So the fastest arm trains the identical policy in less time, with no"
echo "   offsetting cost. Take the winner and put its --num_envs and"
echo "   --num_minibatches into ant_gym_entropy_ab.sbatch."
echo
echo "--- utilisation samples kept in $UTILDIR ---"
echo "=== done ==="

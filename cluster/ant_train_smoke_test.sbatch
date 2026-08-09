#!/bin/bash
# Stage 2 of the mscluster rollout: does ANT training survive at FULL default
# size on the RTX3090? This is the decisive test for a laptop-specific failure
# documented 2026-08-08.
#
# Background (see the project plan's "Ant crash root cause: workload size"):
# on the laptop (RTX 4050, 6GB VRAM, driver 610.88) ant training segfaults
# reproducibly at the default eval width -- exit 139, always the same
# libcuda.so.1.1[27f520] null-pointer offset, at unpredictable points. It is
# NOT a Saute bug and NOT a jax-version bug: it reproduces with
# `--penalizer none` (plain unconstrained PPO) on both jax 0.10.2 and 0.4.34.
# Cache forensics point at death during XLA compilation rather than training
# (a completed run writes ~124 entries to ~/.cache/jax; the crashed full-size
# ant runs wrote 2 across 75 minutes and never printed a step-0 eval).
#
# The laptop workaround was to shrink the eval unroll from 32x10=320 parallel
# envs to 8x2=16, which made ant complete for the first time -- but 16 episodes
# is statistically thin (cost_std 87-117 against means of 69-133), too noisy to
# cite. This job checks whether the 3090 (24GB vs 6GB, driver 595.71.05) needs
# that compromise at all.
#
# Deliberately runs with EVERY size default untouched -- num_envs=256,
# num_eval_envs=32, num_eval_episodes=10, episode_length=1000 -- i.e. exactly
# the configuration that dies on the laptop. Changing any of them would defeat
# the purpose.
#
# Reading the result:
#   - Exit 139 / no step-0 eval line  -> the crash follows the workload, not the
#     laptop. Fall back to a reduced eval width here too, and treat the
#     autotuning theory as the live lead (XLA_FLAGS=--xla_gpu_autotune_level=0).
#   - Completes with 3 eval lines     -> laptop-specific. Run real ant training
#     here at full fidelity, no eval compromise, and note training/sps for
#     scheduling the real (multi-hour) runs.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_train_smoke_test.sbatch
# Watch with `squeue --me`; read ant_train_smoke_%j.log when it finishes.
#
# NOTE: no --gres flag, matching the fix applied to gpu_smoke_test.sbatch after
# its first real submission -- bigbatch attaches a GPU without it. #SBATCH
# directives are kept contiguous (no blank lines) for the same reason.
#SBATCH --job-name=ant-train-smoke
#SBATCH --output=ant_train_smoke_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=03:00:00
set -u
echo "=== node ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
echo "=== env ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-safety-gym
python --version
pip show jax mujoco 2>&1 | grep -E "^Name|^Version"
echo
echo "=== ant training smoke test (ALL size defaults, the laptop-crashing config) ==="
# 0.75 (~18GB of the 3090's 24GB) -- JAX's own default, set explicitly so the
# intent is documented rather than implicit. Deliberately not throttled for
# co-tenancy: this allocation is for one user's own jobs.
# There is a positive reason to keep this generous rather than merely tolerable:
# XLA's autotuner allocates scratch buffers to benchmark candidate kernels at
# COMPILE time, which is exactly where ant dies (see header), so a larger pool
# gives it more room. Can be raised to ~0.9 if a bigger config ever needs it;
# avoid 1.0, which leaves nothing for driver/context overhead.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# --no_checkpoint: this is a survival/throughput probe, not a policy worth keeping.
time python -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant \
  --penalizer none \
  --num_timesteps 500_000 \
  --num_evals 3 \
  --no_checkpoint
status=$?
echo
echo "=== exit status: $status ==="
if [ "$status" -eq 139 ]; then
  echo "SEGFAULT (139) -- same failure class as the laptop. The crash follows"
  echo "the workload, not the hardware. See this script's header for next steps."
elif [ "$status" -eq 0 ]; then
  echo "PASS -- ant survives full default size here. No eval compromise needed;"
  echo "use the training/sps above to schedule the real multi-hour runs."
else
  echo "Failed with an unexpected status -- read the traceback above; this is"
  echo "probably a setup/config problem rather than the known crash."
fi
echo "=== done ==="

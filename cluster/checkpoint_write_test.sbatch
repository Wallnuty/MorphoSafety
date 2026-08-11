#!/bin/bash
# Prove orbax can WRITE and RESTORE a checkpoint on /home-mscluster, before any
# multi-hour job depends on it.
#
# WHY THIS EXISTS. The passing cluster smoke test (job 38961) ran with
# --no_checkpoint, so the checkpoint path has never been exercised here at all.
# That is exactly the gap that cost a 55-minute laptop run: orbax raises
# "Checkpoint path should be absolute" from INSIDE the save call, which runs at
# EVAL time -- so a bad path does not fail at startup, it fails after the
# training is done and leaves nothing on disk. That specific bug is fixed
# (train_ppo.py .resolve()s the path now), but "orbax writes correctly to this
# filesystem" is still an assumption, and /home-mscluster is a networked home
# directory with a 50 GB quota, not the local disk orbax was tested against.
#
# Restoring matters as much as writing. This repo has a documented trap: orbax
# returns dataclasses as dicts whose pytree leaves sort ALPHABETICALLY, while
# restore_state re-flattens POSITIONALLY, so a naive loader silently loads the
# cost critic's weights into the policy slot. The check below therefore
# restores by KEYWORD and asserts on shapes, rather than just confirming files
# exist.
#
# Runtime: ~15-20 min, almost all of it the step-0 compile+eval (366 s measured
# on mscluster78). 200k steps is deliberately tiny -- this measures the I/O
# path, not learning.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/checkpoint_write_test.sbatch
# Read: ckpt_write_test_%j.log -- look for "PASS" at the end, and for the
# restored shapes matching the network the env implies.
#SBATCH --job-name=ckpt-write-test
#SBATCH --output=ckpt_write_test_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=00:45:00
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
echo "home quota / free space:"
df -h ~ | tail -1
echo

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

CKPT="$(pwd)/checkpoints/ckpt_write_test"
rm -rf "$CKPT"
echo "=== 200k steps, checkpointing to $CKPT ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant_gym \
  --task run \
  --penalizer none \
  --num_timesteps 200_000 \
  --num_evals 2 \
  --checkpoint_logdir "$CKPT"
status=$?
echo
echo "=== train exit status: $status ==="
if [ "$status" -ne 0 ]; then
  echo "FAIL: training did not exit cleanly; checkpoint result is meaningless."
  exit 1
fi

echo
echo "=== what landed on disk ==="
ls -1 "$CKPT" 2>&1
du -sh "$CKPT" 2>&1
echo

echo "=== restore it back ==="
python - "$CKPT" <<'PY'
import sys
from pathlib import Path

import jax.numpy as jp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics

ckpt = Path(sys.argv[1])
steps = sorted(p for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit())
assert steps, f"FAIL: no step directories under {ckpt} -- nothing was written"
leaf = steps[-1]
print(f"restoring {leaf}")

loaded = ocp.PyTreeCheckpointer().restore(str(leaf.resolve()))

# BY KEYWORD, not positionally. orbax hands these back as dicts whose leaves
# sort alphabetically (count, mean, std, summed_variance) while the dataclass
# field order is (mean, std, count, summed_variance) -- a positional rebuild
# silently transposes them. Same trap puts the cost critic in the policy slot.
normalizer = running_statistics.RunningStatisticsState(**loaded[0])
policy = loaded[1]["policy"]
value = loaded[1]["value"]

print(f"  normalizer.mean shape : {jp.asarray(normalizer.mean).shape}")
first = policy["params"]["hidden_0"]["kernel"]
print(f"  policy hidden_0 kernel: {jp.asarray(first).shape}")
print(f"  value keys            : {sorted(value['params'].keys())}")

assert jp.isfinite(jp.asarray(first)).all(), "FAIL: policy weights are not finite"
obs_dim = jp.asarray(first).shape[0]
assert jp.asarray(normalizer.mean).shape == (obs_dim,), (
    f"FAIL: normalizer mean {jp.asarray(normalizer.mean).shape} does not match "
    f"the policy's input width {obs_dim} -- the restore is transposed"
)
print("PASS: checkpoint written to /home-mscluster and restored correctly.")
PY
rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "=== PASS -- safe to run ant_gym_run_baseline.sbatch with checkpointing ==="
else
  echo "=== FAIL (status $rc) -- do NOT submit a multi-hour run until this passes,"
  echo "    it would train for hours and save nothing. ==="
fi
rm -rf "$CKPT"
echo "=== done ==="

#!/bin/bash
# Runs ON a Vast instance (2026-10-19): environment-stepping throughput of
# MorphoSafety vs the original mjx-safety-gym 0.1.6 vs Safety-Gymnasium 1.0.0,
# on the same tasks. Results: logs/bench/results.jsonl (one JSON per config),
# machine description: logs/bench/machine.txt. See scripts/bench/*.py.
#
# Must run ALONE on the box (GPU and all CPU cores are being measured).
set -uo pipefail
cd /root/MorphoSafety || exit 1
source /opt/conda_profile.sh
OUT=/root/MorphoSafety/logs/bench/results.jsonl
mkdir -p logs/bench

{
  echo "date: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  echo "affinity cpus: $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')"
  free -g | head -2
} > logs/bench/machine.txt 2>&1
cat logs/bench/machine.txt

echo "=== environments for the other two systems ==="
if ! conda env list | grep -qE '^mjxsg\s'; then
  conda create -y -q -n mjxsg python=3.11 >/dev/null
  conda run -n mjxsg pip install -q --no-cache-dir "mjx-safety-gym==0.1.6" "jax[cuda12]==0.6.0"
fi
if ! conda env list | grep -qE '^sg\s'; then
  # python 3.10: safety-gymnasium pins pygame 2.1.0, which has no 3.11 wheel
  conda create -y -q -n sg python=3.10 >/dev/null
  conda run -n sg pip install -q --no-cache-dir "safety-gymnasium==1.0.0"
fi
conda run -n mjxsg python -c "import mjx_safety_gym, jax; print('mjx-safety-gym', mjx_safety_gym.__version__, mjx_safety_gym.__file__, jax.devices())"
conda run -n sg python -c "import safety_gymnasium, mujoco; print('safety-gymnasium ok, mujoco', mujoco.__version__)"

PT=(256 1024 4096 16384)   # point batch sizes
AN=(256 1024 4096 8192)    # ant / minefield batch sizes

echo "=== MorphoSafety ==="
conda activate morpho
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
for impl in warp jax; do
  python scripts/bench/bench_morphosafety.py --task pointgoal2 --impl $impl --envs "${PT[@]}" --out $OUT
  python scripts/bench/bench_morphosafety.py --task antgoal2 --impl $impl --envs "${AN[@]}" --out $OUT
  python scripts/bench/bench_morphosafety.py --task minefield --impl $impl --envs "${AN[@]}" --out $OUT
done
# the MJX-JAX speed setting this repo trained with 2026-09-27..10-03
python scripts/bench/bench_morphosafety.py --task minefield --impl jax --solver 4 8 --envs "${AN[@]}" --out $OUT
conda deactivate

# Each env's own interpreter, not `conda run` (which parses the script's flags as its own).
BASE="$(conda info --base)"
echo "=== mjx-safety-gym 0.1.6 (isolated: its package is also called mjx_safety_gym) ==="
(cd /tmp && "$BASE/envs/mjxsg/bin/python" -I /root/MorphoSafety/scripts/bench/bench_mjxsg.py \
   --envs "${PT[@]}" --out $OUT)

echo "=== Safety-Gymnasium 1.0.0 (CPU) ==="
for task in pointgoal2 antgoal2; do
  for phys in matched native; do
    for mode in single vector; do
      # BENCH_CPUS: the vCPUs RENTED (24 on this offer). The container can see
      # all 192 host cores, but those are shared with other tenants.
      (cd /tmp && "$BASE/envs/sg/bin/python" /root/MorphoSafety/scripts/bench/bench_sg.py \
         --task $task --physics $phys --mode $mode --num_envs "${BENCH_CPUS:-24}" --out $OUT)
    done
  done
done
echo "=== done: $(wc -l < $OUT) results in $OUT ==="

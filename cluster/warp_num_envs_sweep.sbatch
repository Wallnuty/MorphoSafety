#!/bin/bash
# MuJoCo Warp: DOES TRAINING GET FASTER WITH MORE ENVS PER GPU? (2026-10-04)
#
# At 1024 envs Warp trains at about the same speed on a 3090 (118k single ant,
# 133k co-design, job 63512) as on the laptop RTX 4050 (~113k), so at that size
# it is not limited by GPU compute -- per-step overhead (kernel launches, the
# PPO update) is the suspect. This sweeps the env count to find where a 3090
# saturates, for the single ant and for co-design, and logs GPU utilisation
# alongside training steps/s as direct evidence of which side is the limit.
#
# THROUGHPUT ONLY. num_minibatches scales with num_envs (num_envs / 32, the
# trainer's batch_size), so every training step still consumes one 10-step
# unroll per env and the number of gradient updates per env step is unchanged.
# What does change is how many envs each update's data comes from, so the
# LEARNING outcome at a bigger env count still needs a real run before it
# becomes a default. Co-design keeps 16 bodies, i.e. num_envs/16 lanes each.
#
#   cd ~/MorphoSafety && sbatch cluster/warp_num_envs_sweep.sbatch
#   ENVS="1024 4096" STEPS=20000000 sbatch cluster/warp_num_envs_sweep.sbatch   # smaller
#   bash cluster/vast/vast.sh warp-sweep      # same thing on a rented 3090
# Also runs as a plain bash script (the #SBATCH lines are then comments); PY
# and OUT override the cluster's interpreter and output directory.
#
# Read the SUMMARY at the end of warp_envs_<jobid>.log; per-run logs and GPU
# samples are in warp_envs_<jobid>/.
#
#SBATCH --job-name=warp-envs
#SBATCH --output=warp_envs_%j.log
#SBATCH --partition=bigbatch
# Same exclude list as every other job here (wedged GPUs incl. 44; 75 = user's call).
#SBATCH --exclude=mscluster44,mscluster45,mscluster75,mscluster83
#SBATCH --time=02:00:00
set -u

ENVS="${ENVS:-1024 2048 4096 8192}"
STEPS="${STEPS:-40000000}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${OUT:-${SLURM_SUBMIT_DIR:-$PWD}/warp_envs_${SLURM_JOB_ID:-local}}"
PY="${PY:-$HOME/miniconda3/envs/mjx-warp/bin/python}"
mkdir -p "$OUT"

echo "=== node ==="
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "repo: $REPO ($(git -C "$REPO" log -1 --format=%h 2>/dev/null))  envs: $ENVS  steps/run: $STEPS"
echo

echo "=== preflight ==="
(cd "$REPO" && "$PY" - <<'PY') || { echo "=== preflight FAILED, nothing spent ==="; exit 1; }
import jax, warp as wp
from mjx_safety_gym.algorithms import train_ppo as T
assert any(d.platform == "gpu" for d in jax.devices()), "ABORT: JAX sees no GPU"
wp.init()
assert wp.get_cuda_device_count() > 0, f"ABORT: Warp sees no CUDA device (driver {wp.get_cuda_driver_version()})"
assert "impl" in {a.dest for a in T.build_argparser()._actions}, "ABORT: no --impl (pre-Warp checkout?)"
print(f"PASS: JAX {jax.devices()}, Warp {wp.__version__} on {wp.get_device('cuda:0')}")
PY
echo

# JAX takes 0.6 of the card; Warp allocates its own buffers in the same process.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
COMMON=(--robot ant --task minefield --impl warp --num_timesteps "$STEPS" --num_evals 2
        --num_eval_envs 32 --num_eval_episodes 1 --seed 0)
CODESIGN=(--hazard_lidar --safety_budget 2 --design_optimization --num_morphologies 16
          --design_components 8 --design_updates_per_eval 2
          --steps_before_design_update 0 --steps_after_design_update 0)

run() {  # name num_envs [extra trainer args...]
  local name=$1 n=$2; shift 2
  echo "=== $name ($n envs, $((n / 32)) minibatches) $(date +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 \
    > "$OUT/$name.gpu" 2>/dev/null &
  local smi=$!
  (cd "$REPO" && timeout 3000 "$PY" -u -m mjx_safety_gym.algorithms.train_ppo "${COMMON[@]}" \
      --num_envs "$n" --num_minibatches $((n / 32)) \
      --checkpoint_logdir "$OUT/ckpt_$name" "$@") > "$OUT/$name.log" 2>&1
  local status=$?
  kill "$smi" 2>/dev/null
  echo "    exit $status  $(grep -o 'training/sps=[0-9.]*' "$OUT/$name.log" | tail -1)"
}

for n in $ENVS; do
  run "single_$n" "$n"
  run "codesign_$n" "$n" "${CODESIGN[@]}"
done

echo
echo "=== SUMMARY: MuJoCo Warp training steps/s vs envs per GPU, $STEPS steps each, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
OUT="$OUT" ENVS="$ENVS" "$PY" - <<'PY'
import os, re, pathlib, statistics as st
out = pathlib.Path(os.environ["OUT"])
envs = [int(e) for e in os.environ["ENVS"].split()]
def sps(name):
    p = out / f"{name}.log"
    m = re.findall(r"training/sps=([0-9.]+)", p.read_text()) if p.is_file() else []
    return float(m[-1]) if m else float("nan")
def gpu(name):
    """Median GPU utilisation and peak memory over the second half of the run
    (the first half includes compilation, when the GPU idles)."""
    p = out / f"{name}.gpu"
    rows = []
    for line in (p.read_text().splitlines() if p.is_file() else []):
        try:
            u, m = (float(x) for x in line.split(","))
            rows.append((u, m))
        except ValueError:
            pass
    if not rows:
        return float("nan"), float("nan")
    late = rows[len(rows) // 2:]
    return st.median(u for u, _ in late), max(m for _, m in rows)
base = {w: sps(f"{w}_{envs[0]}") for w in ("single", "codesign")}
print(f"{'envs':>6} | {'single sps':>11} {'x':>5} {'GPU%':>5} {'MiB':>6} | {'co-design sps':>13} {'x':>5} {'GPU%':>5} {'MiB':>6}")
for n in envs:
    cols = []
    for w in ("single", "codesign"):
        v = sps(f"{w}_{n}")
        u, mem = gpu(f"{w}_{n}")
        cols.append(f"{v:11,.0f} {v / base[w]:5.2f} {u:5.0f} {mem:6.0f}" if w == "single"
                    else f"{v:13,.0f} {v / base[w]:5.2f} {u:5.0f} {mem:6.0f}")
    print(f"{n:6d} | {cols[0]} | {cols[1]}")
print(f"x = relative to {envs[0]} envs. GPU% = median utilisation over the second half of each run.")
PY
rm -rf "$OUT"/ckpt_*   # throwaway checkpoints; only the logs matter

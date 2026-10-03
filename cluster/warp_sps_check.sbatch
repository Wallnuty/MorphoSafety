#!/bin/bash
# MuJoCo Warp THROUGHPUT CHECK on a cluster GPU (2026-10-04).
#
# Repeats the laptop probe's speed matrix (notes/warp_migration.md) on a 3090,
# same configs, same seed, in ONE job on ONE node, so the ratios are clean:
#
#   A  current stack, single ant   MuJoCo 3.3.2  MJX-JAX   solver 4/8     ~/MorphoSafety, env mjx-safety-gym
#   D  Warp,          single ant   MuJoCo 3.14   MJX-Warp  solver 100/50  this checkout,  env mjx-warp
#   F  current stack, co-design    (16 bodies, 8 Gaussians, as the 2B runs)
#   H  Warp,          co-design
#
# Laptop (RTX 4050) training steps/s for reference: single ant at 512 envs
# (trainer default) A 13.8k -> D 73.3k (5.3x); co-design at 1024 envs F 19.0k ->
# H 100.0k (5.3x); single ant on Warp at 1024 envs (the 300M quality run)
# ~95-109k. Here EVERY run uses 1024 envs / 32 minibatches, as the real runs do.
# The 3090 ratio is NOT assumed to be the same -- this job is what measures it.
#
# BEFORE SUBMITTING (once):
#   1. On the laptop, push the branch:  git push -u origin warp-probe
#   2. On the cluster, check it out NEXT TO the main checkout and build the env:
#        cd ~/MorphoSafety && git fetch origin
#        git worktree add -b warp-probe ~/MorphoSafety-warp origin/warp-probe
#        bash ~/MorphoSafety-warp/cluster/warp/setup_env.sh
#   3. Submit from the warp checkout:
#        cd ~/MorphoSafety-warp && sbatch cluster/warp_sps_check.sbatch
#
# Read the SUMMARY block at the end of warp_sps_<jobid>.log. Per-run logs and
# throwaway checkpoints go to warp_sps_<jobid>/ next to it.
#
#SBATCH --job-name=warp-sps
#SBATCH --output=warp_sps_%j.log
#SBATCH --partition=bigbatch
# Same exclude list as every other job here (wedged GPUs; 75 = user's call).
#SBATCH --exclude=mscluster45,mscluster75,mscluster83
#SBATCH --time=01:30:00
set -u

MAIN_DIR="${MAIN_DIR:-$HOME/MorphoSafety}"
WARP_DIR="${WARP_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STEPS="${STEPS:-20000000}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/warp_sps_${SLURM_JOB_ID:-local}"
PY_OLD=$HOME/miniconda3/envs/mjx-safety-gym/bin/python
PY_WARP=$HOME/miniconda3/envs/mjx-warp/bin/python
mkdir -p "$OUT"

echo "=== node ==="
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "main checkout: $MAIN_DIR ($(git -C "$MAIN_DIR" log -1 --format=%h 2>/dev/null))"
echo "warp checkout: $WARP_DIR ($(git -C "$WARP_DIR" log -1 --format=%h 2>/dev/null), branch $(git -C "$WARP_DIR" branch --show-current 2>/dev/null))"
echo

# PREFLIGHT: both envs see the GPU, Warp can drive it, and each checkout is
# the one its env expects. A stale or missing piece fails in seconds here.
echo "=== preflight ==="
"$PY_OLD" - <<'PY' || { echo "=== preflight FAILED (old env), nothing spent ==="; exit 1; }
import jax, mujoco
assert any(d.platform == "gpu" for d in jax.devices()), "ABORT: mjx-safety-gym JAX sees no GPU"
print(f"PASS old env: mujoco {mujoco.__version__}, {jax.devices()}")
PY
"$PY_WARP" - <<'PY' || { echo "=== preflight FAILED (warp env), nothing spent ==="; exit 1; }
import jax, mujoco, warp as wp
assert any(d.platform == "gpu" for d in jax.devices()), "ABORT: mjx-warp JAX sees no GPU"
wp.init()
assert wp.get_cuda_device_count() > 0, (
    f"ABORT: Warp sees no CUDA device (driver {wp.get_cuda_driver_version()}); "
    "warp-lang 1.17 may need a newer driver than this node has")
import mjx_safety_gym.backend  # only on the warp-probe branch
print(f"PASS warp env: mujoco {mujoco.__version__}, warp {wp.__version__}, "
      f"CUDA driver {wp.get_cuda_driver_version()}, device {wp.get_device('cuda:0')}")
PY
echo

# Both JAX and Warp reserve VRAM in one process; half each is plenty on 24 GB.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
COMMON=(--robot ant --task minefield --num_timesteps "$STEPS" --num_evals 2
        --num_eval_envs 32 --num_eval_episodes 1 --seed 0 --num_envs 1024 --num_minibatches 32)
CODESIGN=(--hazard_lidar --safety_budget 2 --design_optimization --num_morphologies 16
          --design_components 8 --design_updates_per_eval 2
          --steps_before_design_update 0 --steps_after_design_update 0)
FULL_SOLVER=(--solver_iterations 100 --solver_ls_iterations 50)

run() {  # name python checkout [extra trainer args...]
  local name=$1 py=$2 dir=$3; shift 3
  echo "=== $name $(date +%T)"
  (cd "$dir" && timeout 2400 "$py" -u -m mjx_safety_gym.algorithms.train_ppo "${COMMON[@]}" \
      --checkpoint_logdir "$OUT/ckpt_$name" "$@") > "$OUT/$name.log" 2>&1
  echo "    exit $?  $(grep -o 'training/sps=[0-9.]*' "$OUT/$name.log" | tail -1)"
}

run A_current_single "$PY_OLD" "$MAIN_DIR"
MJX_IMPL=warp MJX_NWORLD=1024 run D_warp_single "$PY_WARP" "$WARP_DIR" "${FULL_SOLVER[@]}"
run F_current_codesign "$PY_OLD" "$MAIN_DIR" "${CODESIGN[@]}"
MJX_IMPL=warp MJX_NWORLD=1024 run H_warp_codesign "$PY_WARP" "$WARP_DIR" "${CODESIGN[@]}" "${FULL_SOLVER[@]}"

echo
echo "=== SUMMARY: training steps/s, $STEPS steps, 1024 envs, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
OUT="$OUT" "$PY_OLD" - <<'PY'
import os, re, pathlib
out = pathlib.Path(os.environ["OUT"])
def sps(n):
    p = out / f"{n}.log"
    m = re.findall(r"training/sps=([0-9.]+)", p.read_text()) if p.is_file() else []
    return float(m[-1]) if m else float("nan")
def warns(n):
    p = out / f"{n}.log"
    return p.read_text().count("limit reached") if p.is_file() else -1
rows = [("single ant", "A_current_single", "D_warp_single"),
        ("co-design", "F_current_codesign", "H_warp_codesign")]
print(f"{'workload':12s} {'current (JAX 4/8)':>18s} {'Warp (full)':>12s} {'speedup':>8s} {'overflow warnings (warp)':>25s}")
for label, a, b in rows:
    x, y = sps(a), sps(b)
    print(f"{label:12s} {x:18,.0f} {y:12,.0f} {y / x:7.2f}x {warns(b):25d}")
print("laptop RTX 4050 for reference: single (512 envs) 13.8k -> 73.3k (5.3x), co-design (1024) 19.0k -> 100.0k (5.3x)")
PY

#!/bin/bash
#SBATCH --job-name=ms-rest
#SBATCH --output=ms_rest_%A_%a.log
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster45,mscluster75,mscluster83
#SBATCH --time=20:00:00
#SBATCH --array=0-1
# The rest of the 2026-10-19 experiments, on TWO bigbatch nodes: one job array,
# each task an independent queue on its own RTX 3090. Random minefield, budget 2.
#
#   task 0  feasibility-first co-design s0, s1, s2 (resumed), speed-ranked s0, s1;
#           each followed by its evolved body retrained from scratch
#   task 1  unconstrained co-design s0, s2 (resumed), s1; speed-ranked s2;
#           each followed by its body retrained WITH the constraint; nominal s0-s2
#
# Every training run is followed by scripts/eval_design.py (3 seeds x 512
# episodes) writing logs/eval/<run>.json. Estimated ~10.5 h per task at the
# ~310k sps measured on a Vast 3090 (cluster 3090s at 4096 envs: unmeasured;
# read training/sps off the first eval line).
#
# BEFORE YOU SUBMIT
#  1. The cluster checkout must have the 2026-10-19 code: feasibility-first
#     ranking, --body_from, and the launcher fixes. Push from the laptop, then
#     pull on the cluster.
#  2. Copy the five partial runs' last checkpoints from the laptop (only these
#     step dirs are needed, ~10 MB each):
#       for r in cd_ff_s0/001303511040 cd_ff_s1/001153105920 cd_ff_s2/001002700800 \
#                cd_unc_s0/001253376000 cd_unc_s2/001203240960; do
#         rsync -az --relative checkpoints/vast/./$r <cluster>:MorphoSafety/checkpoints/resume_src/
#       done
#  3. From the repo root on the cluster:  sbatch cluster/rest_experiments.sbatch
#
# RESUBMITTING continues where each task stopped: jobs already DONE with status
# 0 are skipped (logs/cluster_rest/task<N>/status.txt). A job that died midway
# leaves a partial checkpoints/<run>; delete it (or resume it by hand) first.
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-warp
TASK="${SLURM_ARRAY_TASK_ID:-0}"
# One compile cache per task: two tasks writing the same entries at once is the
# race the cluster notes warn about.
export JAX_CACHE_DIR="$HOME/.cache/jax_rest_task$TASK"
SRC="${RESUME_ROOT:-$PWD/checkpoints/resume_src}"
LOGS="$PWD/logs/cluster_rest/task$TASK"
# Both tasks start in the same second and create the same parent dirs. The
# cluster's Rust coreutils `mkdir -p` fails ("Already exists") if the other task
# creates a parent mid-call (GNU mkdir does not), which killed task 1 of job
# 72805 before it ran anything. Retry until the dirs exist.
for d in "$LOGS" logs/eval; do
  for _ in 1 2 3 4 5; do mkdir -p "$d" 2>/dev/null; [ -d "$d" ] && break; sleep 1; done
  [ -d "$d" ] || { echo "ABORT: could not create $d"; exit 1; }
done
STATUS="$LOGS/status.txt"
echo "=== task $TASK on $(hostname), $(date) ==="

# --- job builders ----------------------------------------------------------
# co-design, fresh: NAME SEED RANKING PENALIZER
cd_new() {
  echo "NAME=$1 HAZARD_PLACEMENT=random PENALIZER=$4 bash cluster/ant_morphology_codesign.sbatch --seed $2 --design_ranking $3"
}
# co-design, resumed: NAME SEED RANKING PENALIZER STEP_DIR NUM_EVALS
# The remaining budget runs at the same 3 x ~16.7M steps per eval as the
# original 2B/41-eval run; the design axis continues from design_state.npz.
cd_resume() {
  local step remaining
  step=$(basename "$5"); remaining=$((2000000000 - 10#$step))
  echo "RESUME=$5 STEPS=$remaining DESIGN_TMAX=2000000000 NUM_EVALS=$6 NAME=$1 HAZARD_PLACEMENT=random PENALIZER=$4 bash cluster/ant_morphology_codesign.sbatch --seed $2 --design_ranking $3"
}
# single body, 300M at budget 2: NAME SEED [CO-DESIGN RUN whose evolved body to train]
single() {
  local extra="--hazard_placement random --seed $2"
  [ -n "${3:-}" ] && extra="$extra --body_from $PWD/checkpoints/$3"
  echo "SAFE_NAME=$1 SAFE_BUDGET=2 SAFE_STEPS=300000000 SAFE_NUM_EVALS=11 SAFE_EXTRA=\"$extra\" bash cluster/ant_minefield_safe_b50.sbatch"
}
evaluate() {
  echo "python scripts/eval_design.py --run checkpoints/$1 --budget 2 --hazard_placement random --seeds 0 1 2 --json logs/eval/$1.json"
}

# --- the two queues (each co-design run before the retrain that needs it) --
JOBS=()
add() { JOBS+=("$1|$2"); }
if [ "$TASK" = 0 ]; then
  add cd_ff_s0     "$(cd_resume cd_ff_s0 0 feasibility_first ppo_lagrangian $SRC/cd_ff_s0/001303511040 15)"
  add eval_cd_ff_s0 "$(evaluate cd_ff_s0)"
  add fx_cd_ff_s0  "$(single fx_cd_ff_s0 0 cd_ff_s0)"
  add eval_fx_cd_ff_s0 "$(evaluate fx_cd_ff_s0)"
  add cd_ff_s1     "$(cd_resume cd_ff_s1 1 feasibility_first ppo_lagrangian $SRC/cd_ff_s1/001153105920 18)"
  add eval_cd_ff_s1 "$(evaluate cd_ff_s1)"
  add fx_cd_ff_s1  "$(single fx_cd_ff_s1 1 cd_ff_s1)"
  add eval_fx_cd_ff_s1 "$(evaluate fx_cd_ff_s1)"
  add cd_ff_s2     "$(cd_resume cd_ff_s2 2 feasibility_first ppo_lagrangian $SRC/cd_ff_s2/001002700800 21)"
  add eval_cd_ff_s2 "$(evaluate cd_ff_s2)"
  add fx_cd_ff_s2  "$(single fx_cd_ff_s2 2 cd_ff_s2)"
  add eval_fx_cd_ff_s2 "$(evaluate fx_cd_ff_s2)"
  for s in 0 1; do
    add cd_speed_s$s "$(cd_new cd_speed_s$s $s objective ppo_lagrangian)"
    add eval_cd_speed_s$s "$(evaluate cd_speed_s$s)"
    add fx_cd_speed_s$s "$(single fx_cd_speed_s$s $s cd_speed_s$s)"
    add eval_fx_cd_speed_s$s "$(evaluate fx_cd_speed_s$s)"
  done
else
  add cd_unc_s0    "$(cd_resume cd_unc_s0 0 objective none $SRC/cd_unc_s0/001253376000 16)"
  add eval_cd_unc_s0 "$(evaluate cd_unc_s0)"
  add fx_cd_unc_s0 "$(single fx_cd_unc_s0 0 cd_unc_s0)"
  add eval_fx_cd_unc_s0 "$(evaluate fx_cd_unc_s0)"
  add cd_unc_s2    "$(cd_resume cd_unc_s2 2 objective none $SRC/cd_unc_s2/001203240960 17)"
  add eval_cd_unc_s2 "$(evaluate cd_unc_s2)"
  add fx_cd_unc_s2 "$(single fx_cd_unc_s2 2 cd_unc_s2)"
  add eval_fx_cd_unc_s2 "$(evaluate fx_cd_unc_s2)"
  add cd_unc_s1    "$(cd_new cd_unc_s1 1 objective none)"
  add eval_cd_unc_s1 "$(evaluate cd_unc_s1)"
  add fx_cd_unc_s1 "$(single fx_cd_unc_s1 1 cd_unc_s1)"
  add eval_fx_cd_unc_s1 "$(evaluate fx_cd_unc_s1)"
  add cd_speed_s2  "$(cd_new cd_speed_s2 2 objective ppo_lagrangian)"
  add eval_cd_speed_s2 "$(evaluate cd_speed_s2)"
  add fx_cd_speed_s2 "$(single fx_cd_speed_s2 2 cd_speed_s2)"
  add eval_fx_cd_speed_s2 "$(evaluate fx_cd_speed_s2)"
  for s in 0 1 2; do
    add nom_s$s "$(single nom_s$s $s)"
    add eval_nom_s$s "$(evaluate nom_s$s)"
  done
fi

# Resume sources must be here before anything is spent.
for j in "${JOBS[@]}"; do
  src=$(grep -o 'RESUME=[^ ]*' <<<"${j#*|}" | cut -d= -f2)
  [ -z "$src" ] || [ -f "$src/design_state.npz" ] || {
    echo "ABORT: resume source $src (with design_state.npz) is missing -- see step 2 in the header"; exit 1; }
done

# --- run ---------------------------------------------------------------------
for j in "${JOBS[@]}"; do
  name="${j%%|*}"; cmd="${j#*|}"
  if grep -q "^DONE $name 0 " "$STATUS" 2>/dev/null; then echo "skip $name (done)"; continue; fi
  echo "START $name $(date -u +%FT%TZ)" | tee -a "$STATUS"
  echo "  $cmd"
  bash -c "$cmd" > "$LOGS/$name.log" 2>&1
  st=$?
  # The launchers end in an echo, so their exit code is always 0; their own
  # "=== exit status: N ===" line is the real result.
  if grep -q '^=== exit status: ' "$LOGS/$name.log"; then
    grep -q '^=== exit status: 0 ===' "$LOGS/$name.log" || st=1
  fi
  echo "DONE $name $st $(date -u +%FT%TZ)" | tee -a "$STATUS"
  [ "$st" = 0 ] || { echo "STOPPED at $name -- see $LOGS/$name.log"; exit "$st"; }
done
echo "QUEUE_FINISHED $(date -u +%FT%TZ)" | tee -a "$STATUS"

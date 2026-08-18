#!/bin/bash
# Runs ON the Vast instance. Finds the throughput-maximising PPO config for
# THIS card, separating the knobs that are free from the ones that only look
# free.
#
# ============================ THE TRAP =====================================
# `training/sps` counts ENVIRONMENT STEPS per second, so it can be raised by
# doing less learning per environment step. Read it alone and the sweep will
# recommend the config that learns slowest per sample. From brax's own
# accounting (ppo/train.py):
#
#     env_step_per_training_step = batch_size * unroll_length
#                                  * num_minibatches * action_repeat
#     gradient updates per training step = num_minibatches * num_updates_per_batch
#
# Divide the second by the first and num_minibatches CANCELS:
#
#     updates per env-step = num_updates_per_batch / (batch_size * unroll_length
#                                                     * action_repeat)
#
# So, at fixed num_updates_per_batch (2, not CLI-exposed) and action_repeat:
#
#   num_envs         FREE. Absent from both formulas. Pure parallelism; the
#                    only constraint is validate()'s
#                    `batch_size * num_minibatches % num_envs == 0`.
#   num_minibatches  FREE. Raises data per training step AND updates per
#                    training step in the same proportion. Minibatch size
#                    (= batch_size) is unchanged.
#   batch_size       NOT FREE. Doubling it halves updates per env-step.
#   unroll_length    NOT FREE. Doubling it halves updates per env-step, and
#                    also lengthens the GAE horizon.
#
# PHASE A therefore holds batch_size and unroll_length FIXED and scales
# num_minibatches with num_envs: every row has IDENTICAL learning density, so
# the fastest row is a free win. PHASE B moves the other two and is reported
# with `upd/Ms` so the trade is visible rather than hidden inside a bigger sps.
#
# ======================= WHY --num_evals 3 =================================
# `training/sps` here is PER EPOCH, not cumulative (ppo/train.py: it divides by
# `epoch_training_time`, the time for that epoch alone). The first epoch pays
# XLA compilation; the second does not. So with 3 evals the LAST sps line is
# already compile-free and needs none of the 56.4 s correction arithmetic the
# laptop sweep had to do. That is why this does not just use --num_evals 2.
#
# num_timesteps is sized PER CONFIG, not fixed: at num_envs 4096 one training
# step consumes 163,840 env steps, so a flat 1M budget would be ~3 training
# steps per epoch and the rate would be quantisation noise. Each epoch gets
# >=6 training steps.
set -uo pipefail
cd /root/MorphoSafety

source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

ROBOT="${ROBOT:-ant}"
TASK="${TASK:-minefield}"
OUT="${OUT:-/root/throughput.tsv}"
UNROLL_DEFAULT=10
ACTION_REPEAT=4          # ant default, _ROBOT_DEFAULTS
NUPB=2                   # num_updates_per_batch, hardcoded in ppo/train.py

# phase:num_envs:batch:minib:unroll:extra
# Phase A -- identical learning density in every row (batch 32, unroll 10).
# Phase B -- one knob off the Phase A baseline, learning density CHANGES.
# Phase C -- the config the next real run actually uses (8 morphologies, the
#            widened 256x4 policy), because a knee measured on a narrow
#            single-body net does not necessarily hold for it.
CONFIGS="${CONFIGS:-
A:512:32:16:10:
A:1024:32:32:10:
A:2048:32:64:10:
A:4096:32:128:10:
B:2048:64:32:10:
B:2048:16:128:10:
B:2048:32:64:20:
B:2048:32:64:5:
C:2048:32:64:10:--num_morphologies 8 --policy_hidden_layer_sizes 256 256 256 256
}"

echo "############################################################"
echo "# throughput sweep: $ROBOT on $TASK"
echo "############################################################"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import jax; d=jax.devices()[0]; print('jax', jax.__version__, d.platform, d.device_kind)"
echo

: > "$OUT"
printf 'phase\tenvs\tbatch\tminib\tunroll\tsps\tspu\tsteps\tsecs\texit\textra\n' >> "$OUT"

# SPLIT ON NEWLINES ONLY. `for cfg in $CONFIGS` word-splits on spaces, which
# shredded the phase C row (its EXTRA field holds
# "--num_morphologies 8 --policy_hidden_layer_sizes 256 256 256 256") into
# seven bogus configs that each exited 2. The EXTRA field is the last one and
# is allowed to contain spaces, so only a newline may end a config.
while IFS= read -r cfg; do
  [ -z "${cfg// /}" ] && continue
  IFS=: read -r PH N B M U EXTRA <<< "$cfg"
  SPU=$(( B * U * M * ACTION_REPEAT ))
  # >=6 training steps per epoch, 2 epochs, floor 1M.
  STEPS=$(( SPU * 12 )); [ "$STEPS" -lt 1000000 ] && STEPS=1000000
  echo "===== [$PH] envs=$N batch=$B minib=$M unroll=$U  ${EXTRA:-}"
  echo "      $SPU env-steps/training-step, running $STEPS steps"
  start=$(date +%s)
  # shellcheck disable=SC2086
  out=$(timeout 3000 python -u -m mjx_safety_gym.algorithms.train_ppo \
      --robot "$ROBOT" --task "$TASK" --penalizer none \
      --num_timesteps "$STEPS" --num_evals 3 \
      --num_envs "$N" --batch_size "$B" --num_minibatches "$M" \
      --unroll_length "$U" \
      --num_eval_envs 8 --num_eval_episodes 2 --no_checkpoint $EXTRA 2>&1)
  status=$?
  secs=$(( $(date +%s) - start ))
  # LAST sps line = second epoch = no compile in the denominator.
  sps=$(printf '%s' "$out" | grep -o 'training/sps=[0-9.]*' | tail -1 | cut -d= -f2)
  [ -n "$sps" ] || sps=0
  printf '%s\n' "$out" | grep -E "Error|Traceback|RESOURCE_EXHAUSTED|Killed|must be a multiple" | tail -3
  echo "      -> training/sps=$sps  exit=$status in ${secs}s"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$PH" "$N" "$B" "$M" "$U" "$sps" "$SPU" "$STEPS" "$secs" "$status" "${EXTRA:-none}" >> "$OUT"
  # 137 host-OOM, 124 timeout, 139 segfault -- a bigger config will not help.
  if [ "$status" = "137" ] || [ "$status" = "124" ] || [ "$status" = "139" ]; then
    echo "      !! failed; skipping the rest of this phase's larger sizes"
  fi
  echo
done <<< "$CONFIGS"

echo
echo "=============================== SUMMARY ==============================="
NUPB=$NUPB AR=$ACTION_REPEAT python3 - "$OUT" <<'PY'
import os, sys
nupb, ar = int(os.environ["NUPB"]), int(os.environ["AR"])
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
print(f"{'ph':>3}{'envs':>6}{'batch':>7}{'minib':>7}{'unroll':>7}"
      f"{'sps':>9}{'upd/s':>8}{'upd/Ms':>9}{'ok':>4}")
best = {}
for ph, n, b, m, u, sps, spu, steps, secs, st, extra in rows:
    # A config that died before argparse leaves empty numeric fields; float('')
    # would crash the whole summary and lose the rows that DID succeed.
    try:
        sps = float(sps); b, u = int(b), int(u)
    except ValueError:
        print(f"{ph:>3}{n:>6}{'':>7}{'':>7}{'':>7}{'--':>9}{'--':>8}{'--':>9}{'NO':>4}")
        continue
    # updates per env-step: num_minibatches cancels out entirely.
    upd_per_step = nupb / (b * u * ar)
    ok = "y" if st == "0" and sps > 0 else "NO"
    print(f"{ph:>3}{n:>6}{b:>7}{m:>7}{u:>7}{sps:>9.0f}"
          f"{sps*upd_per_step:>8.1f}{1e6*upd_per_step:>9.0f}{ok:>4}")
    if ok == "y":
        best.setdefault(ph, []).append((sps, n, b, m, u))
print()
if "A" in best:
    a = sorted(best["A"])[-1]
    base = min(best["A"])
    print(f"PHASE A winner: num_envs={a[1]} minib={a[3]} at {a[0]:.0f} sps "
          f"({a[0]/base[0]:.2f}x the smallest row).")
    print("  Learning density is IDENTICAL across phase A, so this is free.")
if "B" in best:
    print("PHASE B rows change updates-per-env-step -- compare upd/s, NOT sps.")
    print("  A row with higher sps but lower upd/s is slower to converge.")
if "C" in best:
    c = best["C"][0]
    same = [r for r in best.get("A", []) if r[1] == c[1]]
    if same:
        print(f"PHASE C (8 morphologies, 256x4 policy): {c[0]:.0f} sps vs "
              f"{same[0][0]:.0f} for the same size single-body "
              f"({100*c[0]/same[0][0]:.0f}%).")
PY
echo
echo "raw: $OUT"

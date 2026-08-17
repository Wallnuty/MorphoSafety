#!/bin/bash
# Runs ON the Vast instance. Short training runs at several batch sizes, to
# measure what this card actually does BEFORE renting it for a long job.
#
# WHY A SWEEP RATHER THAN ONE NUMBER. On the laptop, minefield's throughput was
# strongly batch-size dependent and had NOT saturated at the largest size that
# fit in 6 GB:
#
#     envs    run (sps)   minefield (sps)
#       64        702.9             665.4     <- both launch-bound, no gain
#      128        781.1            1239.6
#      256        870.0            2725.9     <- laptop training default
#      512      untested           4184.0     <- still climbing when VRAM ran out
#
# So the laptop never found the knee; it ran out of card first. A 24 GB GPU has
# 4x the VRAM, and these boxes have far more host RAM than the laptop's 7.6 GB
# (which is what OOM-killed num_envs=2048 during XLA COMPILATION -- a different
# resource from VRAM, and the one that bit first). Both ceilings move here, so
# the right size is an empirical question, not a carry-over.
#
# Quoting a laptop number for this card would be a guess. This measures it.
set -uo pipefail
cd /root/MorphoSafety

source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

ROBOT="${ROBOT:-ant}"
TASK="${TASK:-minefield}"
STEPS="${STEPS:-500000}"

# num_envs:batch_size:num_minibatches -- batch MUST scale past 512.
# validate() requires num_envs to divide batch_size * num_minibatches, and the
# default product is 32*16 = 512. Sweeping num_envs alone would make 1024 and
# 2048 exit before a single step, so the sweep would silently only ever test
# two sizes.
#
# THE TWO HALVES ARE NOT COMPARABLE, for the same reason as the laptop sweep:
# 256/512 share an identical learning config and differ only in parallelism,
# while 1024+ doubles the gradient batch and therefore HALVES the number of
# updates per env-step. Read them as two separate curves. Laptop reference
# (compile-corrected training/sps, ant on minefield): 3593 @256, 4747 @512,
# 4900 @1024, 4735 @2048 -- i.e. saturated by 512 on a 6 GiB RTX 4050.
CONFIGS="${CONFIGS:-256:32:16 512:32:16 1024:64:16 2048:128:16}"

echo "############################################################"
echo "# smoke: $ROBOT on $TASK, $STEPS steps at each of: $CONFIGS"
echo "############################################################"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

for cfg in $CONFIGS; do
  N="${cfg%%:*}"; rest="${cfg#*:}"; B="${rest%%:*}"; M="${rest##*:}"
  SPU=$(( B * M * 10 * 4 ))     # batch * minib * unroll_length * action_repeat
  echo "============ num_envs=$N batch=$B minib=$M ($SPU steps/update) ============"
  # --num_evals 2 for the same reason the laptop chain uses it: no intermediate
  # evals, so training/sps is reported once, at the end, uncontaminated by
  # eval time. --no_checkpoint because a smoke run's weights are worthless.
  start=$(date +%s)
  timeout 3000 python -u -m mjx_safety_gym.algorithms.train_ppo \
      --robot "$ROBOT" --task "$TASK" --penalizer none \
      --num_timesteps "$STEPS" --num_evals 2 --num_envs "$N" \
      --batch_size "$B" --num_minibatches "$M" \
      --num_eval_envs 8 --num_eval_episodes 2 --no_checkpoint 2>&1 \
    | grep -E "training/sps|step=|Error|error|RESOURCE_EXHAUSTED|Killed|Traceback" \
    | tail -n 6
  status=${PIPESTATUS[0]}
  echo "--- num_envs=$N exit=$status in $(( $(date +%s) - start ))s"
  # 137 = OOM-killed host-side, 124 = timeout. Either means this size and every
  # larger one is out; stop rather than burn rental time proving it twice.
  if [ "$status" = "137" ] || [ "$status" = "124" ] || [ "$status" = "139" ]; then
    echo "=== stopping sweep: num_envs=$N failed (exit $status) ==="
    break
  fi
  echo
done

echo
echo "=== SMOKE DONE ==="
echo "Take the largest num_envs with the best training/sps, then:"
echo "  bash cluster/vast/vast.sh train --robot $ROBOT --task $TASK \\"
echo "       --penalizer none --num_envs <N> --num_timesteps 50000000 \\"
echo "       --num_evals 20 --checkpoint_logdir /root/MorphoSafety/checkpoints/vast_run"

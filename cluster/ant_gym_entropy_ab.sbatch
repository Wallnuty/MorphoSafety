#!/bin/bash
# ============================================================================
# SUPERSEDED 2026-08-12 -- DO NOT SUBMIT. The hypothesis this job tests was
# measured and REFUTED on the laptop before this job ever ran.
#
# Two matched 1.5M runs, identical but for entropy_cost:
#
#     entropy_cost   avg_episode_length      final entropy
#                    step 0 -> step 1.5M
#     1e-4            99.75 -> 135.50          -1.0 nats
#     1e-3           109.25 -> 110.00          -1.1 nats
#
# 10x the entropy bonus produced the SAME collapsed entropy and no improvement
# in survival. Whatever drives sigma down to ~0.21 is not an insufficient
# entropy term, so arm 1 buys nothing and this array would spend ~15 GPU-hours
# to learn that a second time.
#
# Submit cluster/ant_gym_run_baseline.sbatch instead -- it is arm 0 of this
# job, and arm 0 is the only arm still worth running.
#
# Kept rather than deleted so the refutation is discoverable next to the
# hypothesis. The reasoning below is preserved but is now HISTORY.
# ============================================================================
# ant_gym on RunForward, 15M env steps, entropy_cost 1e-4 vs 1e-3.
#
# SUPERSEDES SUBMITTING ant_gym_run_baseline.sbatch ON ITS OWN. Arm 0 here is
# that baseline (same robot, task, penalizer, step count, resolved defaults) --
# it just also runs the arm that tests the leading hypothesis for why reward
# has not moved. Do not run both; you would be paying twice for arm 0.
#
# WHY entropy_cost IS THE VARIABLE
#
# `normalize_advantage=True` in ppo/train.py makes --forward_reward_weight
# scale-invariant for the policy gradient, so there is no point tuning reward
# magnitude. entropy_cost is different. `entropy_loss = -entropy_cost *
# entropy`, so the -0.0001 logged at entropy_cost=1e-4 implies entropy ~= 1.0
# nats. A freshly initialised 8-dim Gaussian policy has ~11.4 nats (sigma ~1.0
# per dim); 1.0 nats is sigma ~0.27. The policy had shrunk its exploration
# noise ~4x by 200k steps -- very early, for a task whose entire difficulty is
# discovering a gait from scratch. (The log rounds to 4 dp so entropy is
# somewhere in 0.5-1.5 nats; the conclusion holds at either end.)
#
# 1e-4 is ss2r's value, inherited from a config tuned for the POINT robot,
# whose "gait" is a single actuator. There is no reason it transfers to an
# 8-actuator quadruped, and premature entropy collapse is a classic cause of
# exactly the "no measurable improvement" result seen so far.
#
# WHY THIS BELONGS ON THE CLUSTER AND NOT THE LAPTOP
#
# The laptop cannot answer it. 15M env steps at env_step_per_training_step =
# batch_size * unroll_length * num_minibatches * action_repeat = 32*10*16*4 =
# 20,480 is 732 PPO iterations; the laptop's survivable runs (--num_evals 2,
# ~1.5M steps) reach ~74, a tenth of that, and land within noise of an
# untrained policy. The 3090 does the full 15M in ~6.5h and job 38961 proved
# the laptop's segfault class does not occur there.
#
# SIZING. 15M / ~750 training sps ~= 5.6h. Eval unrolls episode_length /
# action_repeat = 625 decisions, 2.5x the old ant's 250, so ~550-900 s per eval
# at the full 32x10 width; 8 evals ~= 1.2-2h. Call it ~7.6h. --time 16h is
# ~2.1x margin and bigbatch's cap is 72h. training/sps 668-798 is measured
# (job 38961) but on the OLD ant -- ant_gym has identical topology (same
# ngeom/nbody/nq/nv/nu and contact counts, only geometry scales) so it should
# carry over. Read training/sps off the first eval and adjust if it disagrees.
#
# HOW TO READ THE RESULT
#
#   * eval/episode_reward IS METRES TRAVELLED IN +x per episode. This is the
#     whole question. Untrained is ~1.1 (measured, laptop, 8x2 eval). If BOTH
#     arms are still ~1 at 15M, the task/morphology change was not sufficient
#     and the next lever is reward shaping or the morphology signal from CRAX
#     -- not more compute, and not another hyperparameter.
#   * training/entropy_loss: divide by -entropy_cost to get nats. Watch whether
#     1e-3 actually holds entropy up. If 1e-3 collapses to ~1 nat as well, the
#     entropy hypothesis is dead and 1e-2 is not worth trying either.
#   * If 1e-3 wins on reward, it becomes the default before ANY morphology
#     search runs -- comparing morphologies under a policy that never explored
#     would rank them on noise.
#   * FINAL eval/episode_cost of the winning arm sets --safety_budget. Every
#     cost figure recorded before 2026-08-10 is void (different task, robot and
#     episode length), including the 150 default and the 728.
#   * Cost split: watch whether cost is dominated by the corridor boundary
#     rather than by hazards. If a TRAINED policy is still boundary-dominated,
#     narrow --corridor_half_width or shorten --corridor_length to raise
#     obstacle density rather than reweighting the cost.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_gym_entropy_ab.sbatch
# Watch: squeue --me     Read: ant_gym_ent_<jobid>_0.log and _1.log
#
# IF ONE DIES PARTWAY: checkpoints are written every eval, so resume with
#   --restore_checkpoint_path <repo>/checkpoints/ant_gym_ent_<value>/<step_dir>
# (a resumed run restarts its own step counter at 0 and will overwrite
# same-named checkpoint dirs -- back up the good one first).
#SBATCH --job-name=antgym-ent
#SBATCH --output=ant_gym_ent_%A_%a.log
#SBATCH --partition=bigbatch
#SBATCH --time=16:00:00
#SBATCH --array=0-1
set -u
ENTROPIES=(1e-4 1e-3)
E=${ENTROPIES[$SLURM_ARRAY_TASK_ID]}

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

echo "=== ARM $SLURM_ARRAY_TASK_ID: entropy_cost=$E, 15M env steps = 3.75M decisions ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# python -u is NOT optional. A laptop run reached 573k steps (proven by
# checkpoints on disk) and left a COMPLETELY EMPTY log, because Python
# block-buffers stdout when redirected and a segfault never flushes it.
#
# action_repeat / episode_length / discounting / corridor dimensions are
# deliberately NOT passed: they resolve from _ROBOT_DEFAULTS and the robot's
# arena_scale, and the job log prints the resolved values on its first line.
# Pinning them here would freeze this run to whatever was correct on the day
# the script was written.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant_gym \
  --task run \
  --penalizer none \
  --entropy_cost "$E" \
  --num_timesteps 15_000_000 \
  --num_evals 8 \
  --checkpoint_logdir "$(pwd)/checkpoints/ant_gym_ent_$E"
status=$?
echo
echo "=== exit status: $status  (entropy_cost=$E) ==="
echo
echo "--- checkpoints written ---"
ls -1 "checkpoints/ant_gym_ent_$E/" 2>&1 | tail -5
du -sh "checkpoints/ant_gym_ent_$E/" 2>&1
echo
if [ "$status" -eq 0 ]; then
  echo "PASS (entropy_cost=$E). Compare the two arms' eval/episode_reward"
  echo "curves side by side -- that is the experiment. Then read"
  echo "training/entropy_loss / -$E to check entropy in nats actually differed"
  echo "between arms; if it did not, this A/B tested nothing."
else
  echo "FAILED (status $status, entropy_cost=$E). If 139, this is the segfault"
  echo "class seen on the laptop -- but job 38961 ran the ant at full default"
  echo "size on this partition cleanly, so treat a crash HERE as new and check"
  echo "the last eval line for how far it got before resuming from the newest"
  echo "checkpoint rather than restarting (see header)."
fi
echo "=== done ==="

#!/bin/bash
# !! SUPERSEDED 2026-08-10 -- DO NOT SUBMIT WITHOUT READING THIS !!
# This job targets `--robot ant` on the GoToGoal task, which was measured this
# night to be UNLEARNABLE for the ant: over 16 seeds, a scripted gait that
# walks several metres earns +0.0781 +- 0.6715 return, i.e. statistically zero,
# because goals sit in a uniformly random direction and locomotion alone is
# worth nothing. The ant also never travels far enough to touch an obstacle --
# measured hazard cost was 0.0 in every mode -- so any cost number this job
# produces describes a near-stationary robot and is the wrong number to
# calibrate --safety_budget from.
#
# The replacements are `--task run` (envs/run_forward.py, reward linear in
# +x displacement) and `--robot ant_gym` (envs/xmls/ant_gym.xml, the
# Gym-proportioned ant: ~20x the travel per episode, and the first config to
# register non-zero hazard cost at all).
#
# Everything below is kept because the sizing arithmetic, the python -u
# rationale and the checkpoint-path validation are all still correct -- only
# the task/robot choice is stale. Re-point --robot/--task before submitting,
# and note that action_repeat is back to 4, so the step accounting in any
# header below that assumes 10 is wrong by 2.5x.
#
# Stage 3: the converged UNCONSTRAINED ant baseline.
#
# REWRITTEN 2026-08-09 after discovering the ant had never actually been
# learning. The previous version of this script targeted 15M steps under
# --action_repeat 4; both the step accounting and the premise have changed.
#
# WHY THE OLD VERSION WOULD HAVE WASTED ~7 HOURS. The trained ant policy was
# measured moving 0.011 m per episode against 0.313 m for uniform-random
# actions -- it had converged to standing perfectly still. That is rational
# under this reward, not a bug: reward is a signed distance delta, so undirected
# motion has strictly negative expected return (distance-to-a-point is convex),
# while freezing scores exactly 0. Reaching a goal was also close to
# geometrically impossible: goals spawn ~0.8 m away with a 0.3 m capture radius
# (~0.5 m of travel needed) while a random policy netted only 0.32 m in the old
# 10-second episode, so the +1 goal bonus never fired and nothing ever taught
# the policy that goals exist. An "unconstrained cost baseline" measured on that
# policy would have described a stationary robot standing near hazards -- a
# plausible-looking but wrong number to calibrate --safety_budget from.
#
# The env now applies per-robot defaults (_ROBOT_DEFAULTS in train_ppo.py):
# ant gets action_repeat=10, episode_length=2500, discounting=0.97 -- 25 s of
# simulated time (vs 10 s) and a 3.3 s value horizon (vs 0.4 s, shorter than the
# measured 0.3 s best gait period). Point is untouched and bit-identical.
#
# !! STEP ACCOUNTING CHANGED -- READ BEFORE COMPARING TO OLD RUNS !!
# --num_timesteps counts PHYSICS steps INCLUDING action_repeat (see
# env_step_per_training_step in ppo/train.py). At the ant's new
# action_repeat=10, a given budget buys 2.5x FEWER gradient steps than it did
# at 4. The old plan's "15M steps" was 3.75M decisions; 15M today would be only
# 1.5M. This script requests 37.5M specifically so the DECISION count matches
# the originally planned 3.75M -- that is the quantity that determines whether
# the policy learns, and matching it is what makes this comparable to the point
# unconstrained baseline.
#
# WHAT THIS JOB ANSWERS, in priority order:
#   1. DID THE FREEZE BREAK? Watch eval/episode_reward. It starts slightly
#      NEGATIVE (-0.08 measured at step 0) because a random walk drifts away
#      from the goal on average. The signal to look for is it crossing zero and
#      climbing. If it is still pinned at ~0 after the first third of the run,
#      config tuning has been exhausted and the next lever is a locomotion
#      shaping term (a forward-progress reward, which is what makes
#      safety-gymnasium's own SafetyAntVelocity work). Kill the job at that
#      point rather than paying for the remaining hours -- that is why
#      --num_evals is 15 rather than a handful.
#   2. WHAT SHOULD --safety_budget BE? The current default of 150 is now
#      definitively wrong: measured untrained cost under the new episode length
#      is 728 (up from ~137, of which ~2.5x is the longer episode and the rest
#      is genuinely more hazard contact from more movement). Set the budget from
#      this run's FINAL eval/episode_cost, not from 150 and not from any cost
#      number recorded before 2026-08-09 -- all of those are frozen-ant numbers
#      at a different episode length.
#   3. DOES THE ORBAX CHECKPOINT-WRITE PATH WORK ON /home-mscluster? Still
#      unvalidated -- the stage-2 smoke test ran with --no_checkpoint.
#
# RUNTIME PROJECTION. Cluster measured 668-798 training/sps at the old config;
# the new config raised env-step throughput ~15% on the laptop (288.9 -> 332.2)
# because the fixed ~3 ms per-decision overhead now amortizes over 10 physics
# steps instead of 4. Call it ~850 sps here:
#   37.5M / ~850 sps ~= 12.3h training
#   eval is ~2.2x its old cost per epoch (same 250-decision unroll, 2.5x the
#   physics): ~480-800s x 15 evals ~= 2.7h
#   => ~15h total. --time is 24h for ~1.6x margin; bigbatch's cap is 72h.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_unconstrained_baseline.sbatch
# Watch: squeue --me     Read: ant_unconstrained_%j.log
#
# IF IT DIES PARTWAY: checkpoints are written every eval, so resume with
#   --restore_checkpoint_path <repo>/checkpoints/ant_unconstrained/<step_dir>
# (note the resumed run restarts its own step counter at 0 and will overwrite
# same-named checkpoint dirs -- back up the good one first).
#SBATCH --job-name=ant-unconstrained
#SBATCH --output=ant_unconstrained_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=24:00:00
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
df -h ~ | tail -1
echo
echo "=== unconstrained ant baseline: 37.5M env steps = 3.75M decisions ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# python -u is NOT optional. A laptop run reached 573k steps (proven by
# checkpoints on disk) and left a COMPLETELY EMPTY log, because Python
# block-buffers stdout when redirected to a file and a segfault never flushes
# it. Unbuffered output is the difference between diagnosing a failure and
# guessing at it -- and here it is also what lets you bail early on question 1.
#
# action_repeat / episode_length / discounting are deliberately NOT passed:
# they now resolve from _ROBOT_DEFAULTS for --robot ant, and the job log prints
# the resolved values on the first line ("[ant] robot defaults applied: ...").
# Passing them explicitly would silently pin this run to whatever was correct
# on the day it was written.
#
# Separate checkpoint dir (not the default checkpoints/ant) so this baseline
# cannot clobber, or be clobbered by, other ant runs -- mirrors the existing
# point / point_unconstrained convention. main.py won't auto-discover it, so
# view it with:  python main.py --robot ant --checkpoint checkpoints/ant_unconstrained
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant \
  --penalizer none \
  --num_timesteps 37_500_000 \
  --num_evals 15 \
  --checkpoint_logdir checkpoints/ant_unconstrained
status=$?
echo
echo "=== exit status: $status ==="
echo
echo "--- checkpoints written (validates the orbax write path on /home-mscluster) ---"
ls -1 checkpoints/ant_unconstrained/ 2>&1 | tail -5
du -sh checkpoints/ant_unconstrained/ 2>&1
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. What to read off the eval lines above, in priority order:"
  echo "  * eval/episode_reward TRAJECTORY -- the headline result. Starts ~-0.08."
  echo "    Crossing zero and climbing => the ant learned to move toward goals and"
  echo "    the freeze is broken. Still ~0 or drifting more negative at the end =>"
  echo "    the config fix was not sufficient; stop tuning episode/horizon settings"
  echo "    and add a locomotion shaping term instead."
  echo "  * FINAL eval/episode_cost -- set --safety_budget from THIS number."
  echo "    Untrained under this config was 728. Expect it to move as the policy"
  echo "    learns (point's unconstrained cost ROSE to 353 once trained, because a"
  echo "    goal-seeking agent takes more risk). Ignore the old 150 default and"
  echo "    every cost figure recorded before 2026-08-09."
  echo "  * eval/episode_cost is only meaningful IF reward moved. A frozen ant"
  echo "    near hazards produces a cost number that looks fine and means nothing."
  echo "  * training/sps -- schedules the real morphology runs, and checks the"
  echo "    ~850 sps projection this script's --time was sized from."
else
  echo "FAILED (status $status). If 139, this is the segfault class seen on the"
  echo "laptop appearing here too -- unexpected, since stage 2 passed. Check the"
  echo "last eval line above for how far it got, and resume from the newest"
  echo "checkpoint rather than restarting (see header)."
fi
echo "=== done ==="

#!/bin/bash
# The unconstrained baseline for the CURRENT task and robot: ant_gym on
# RunForward. Supersedes cluster/ant_unconstrained_baseline.sbatch, which
# targets a task the ant provably cannot learn.
#
# WHAT CHANGED AND WHY (all measured 2026-08-10, see the project plan)
#
# GoToGoal is unlearnable for the ant. Over 16 seeds, episode return against a
# do-nothing policy: zero action -0.0000 +- 0.0000, random +0.0441 +- 0.1351,
# and a SCRIPTED GAIT THAT WALKS SEVERAL METRES +0.0781 +- 0.6715 -- i.e.
# statistically zero. Goals sit in a uniformly random direction, so locomotion
# by itself earns nothing and gait and steering have to be discovered
# simultaneously. Nobody ships a working ant navigation task either:
# safety-gymnasium publishes no SafetyAntGoal results, and CRAX registers
# safe_goal_point but no ant navigation env at all.
#
# RunForward (--task run) rewards x-displacement, which is LINEAR, so zero-mean
# motion has expected reward 0 instead of negative and exploration is free.
# Verified: episode return EQUALS net +x displacement to 0.00e+00, a +x gait
# scores +0.4179, the same gait reversed -0.1987, freezing exactly 0.0000.
#
# ant_gym (--robot ant_gym) is ant.xml at 4x length scale without the
# density="50000" ankle hack -- which reproduces the Gym/Brax ant's mass to
# three decimals (0.9109 vs 0.9110 kg), because our ant IS that ant shrunk 4x.
# torque/kg goes 3.5 -> 164.7. In RunForward over 12 s the old ant nets -0.05 m
# with hazard cost 0.0 IN EVERY MODE -- it never reaches an obstacle, so the
# safety signal cannot fire; ant_gym nets +1.00 m with hazard cost 3.1.
#
# SIZING. env_step_per_training_step = batch_size * unroll_length *
# num_minibatches * action_repeat = 32*10*16*4 = 20,480, so 15M env steps is
# 3.75M decisions -- the same decision count the original plan was built
# around, and comparable to the point unconstrained baseline. action_repeat is
# 4, NOT 10: GoToGoal.step already applies n_substeps=2, so the real control
# period is action_repeat*2*timestep, and a sweep showed the achievable
# locomotion ceiling falls 2.7x going from 4 to 10 while random-action travel
# saturates by 4.
#
# RUNTIME. ant_gym has IDENTICAL topology to ant (same ngeom/nbody/nq/nv/nu and
# the same contact counts -- only geometry scales), so the cluster's measured
# 668-798 training/sps for ant carries over; episode_length does not change sps,
# only how often episodes reset. Call it ~750:
#   15M / ~750 sps ~= 5.6h training
#   eval unrolls episode_length/action_repeat = 625 steps, 2.5x the old 250, so
#   ~550-900s per eval at full 32x10 width; 15 evals ~= 2.9h
#   => ~8.5h. --time 16h is ~1.9x margin; bigbatch's cap is 72h.
# These are extrapolations from the ANT's cluster numbers, not measurements of
# ant_gym -- read training/sps off the first eval and adjust if it disagrees.
#
# WHAT THIS ANSWERS
#   1. DOES REWARD CLIMB? eval/episode_reward is metres travelled per episode.
#      A laptop smoke test measured +0.5585 at step 0 (untrained). If it is
#      still ~0.5 after a few million decisions, the task/robot change was not
#      sufficient and the next lever is reward shaping, not more compute.
#   2. WHAT SHOULD --safety_budget BE? Cost is now boundary + hazard + vase on
#      a different episode length with a different robot, so EVERY cost figure
#      recorded before 2026-08-10 is void, including the 728 and the 150
#      default. Set it from this run's final eval/episode_cost.
#   3. HOW IS COST SPLIT? Watch whether cost is dominated by the corridor
#      boundary rather than by hazards. Untrained rollouts were boundary-heavy,
#      which is expected for a flailing policy but would make the safety
#      question vacuous if it persists once trained -- if so, narrow
#      --corridor_half_width or shorten --corridor_length to raise obstacle
#      density rather than reweighting the cost.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_gym_run_baseline.sbatch
# Watch: squeue --me     Read: ant_gym_run_%j.log
#
# IF IT DIES PARTWAY: checkpoints are written every eval, so resume with
#   --restore_checkpoint_path <repo>/checkpoints/ant_gym_run/<step_dir>
# (the resumed run restarts its own step counter at 0 and will overwrite
# same-named checkpoint dirs -- back up the good one first).
#SBATCH --job-name=antgym-run
#SBATCH --output=ant_gym_run_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=16:00:00
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
echo "=== unconstrained ant_gym on RunForward: 15M env steps = 3.75M decisions ==="
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
  --num_timesteps 15_000_000 \
  --num_evals 15 \
  --checkpoint_logdir checkpoints/ant_gym_run
status=$?
echo
echo "=== exit status: $status ==="
echo
echo "--- checkpoints written ---"
ls -1 checkpoints/ant_gym_run/ 2>&1 | tail -5
du -sh checkpoints/ant_gym_run/ 2>&1
echo
if [ "$status" -eq 0 ]; then
  echo "PASS. Read the eval lines above in this order:"
  echo "  * eval/episode_reward IS METRES TRAVELLED per episode. Untrained is"
  echo "    ~0.56. Climbing => the ant learned to walk and the whole"
  echo "    task/morphology change worked. Flat at ~0.5 => stop tuning config"
  echo "    and add reward shaping; more compute will not fix it."
  echo "  * FINAL eval/episode_cost -- set --safety_budget from THIS number."
  echo "    Ignore 150 and every cost figure recorded before 2026-08-10."
  echo "  * eval/episode_cost is only meaningful IF reward moved."
  echo "  * training/sps -- checks the ~750 extrapolation this --time was"
  echo "    sized from, and schedules the real morphology runs."
else
  echo "FAILED (status $status). If 139, this is the segfault class seen on the"
  echo "laptop. Check the last eval line for how far it got and resume from the"
  echo "newest checkpoint rather than restarting (see header)."
fi
echo "=== done ==="

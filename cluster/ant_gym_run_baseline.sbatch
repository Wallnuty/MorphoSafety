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
# WHAT THIS ANSWERS -- REWRITTEN 2026-08-12, READ THIS BEFORE THE LOG
#
#   1. READ eval/avg_episode_length FIRST. NOT reward. The ant now terminates
#      when it flips (RunForward.is_flipped), and it was measured inverted for
#      94% of every episode before that existed. So episode length IS the
#      uprightness learning curve: an untrained ant flips at ~100 steps against
#      the 2500 cap. On the laptop, 1.5M steps moved it 99.75 -> 135.50, with a
#      +-10 noise floor at 16 eval episodes. If this run does not push it
#      substantially past ~200, the ant still cannot stand up and nothing
#      downstream is meaningful.
#
#   2. eval/episode_cost WILL READ 0.0000 EARLY AND THAT IS NOT A BUG. Episodes
#      end before the ant travels far enough to reach an obstacle, so the
#      safety signal is vacuous until episodes lengthen. **--safety_budget
#      CANNOT be derived from a run whose avg_episode_length is still ~100.**
#      Only read cost once episode length is climbing.
#
#   3. eval/episode_reward is metres travelled in +x, but it is confounded with
#      episode length now -- a longer-surviving policy has more time to travel.
#      Read it only alongside avg_episode_length, never alone.
#
#   4. training/entropy_loss / -entropy_cost = entropy in nats. Expect it to go
#      NEGATIVE (~-1.0, sigma ~0.21 against 1.0 at init). Do NOT reach for
#      --entropy_cost to fix that: a matched 1.5M A/B measured 1e-3 giving the
#      SAME collapsed entropy (-1.1 nats) and no survival benefit. That
#      hypothesis is refuted; see cluster/ant_gym_entropy_ab.sbatch.
#
#   5. training/sps -- checks the ~750 extrapolation this --time was sized
#      from. The laptop measures 472 on this config.
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
# PREFLIGHT: abort if JAX cannot see a GPU.
#
# This is not paranoia. Job 40227 landed on mscluster65, whose GPU0 was in a
# bad state ("Unable to determine the device handle for GPU0: 0000:17:00.0:
# Unknown Error", then "cuInit(0) failed: CUDA_ERROR_UNKNOWN"). JAX does NOT
# fail on that -- it prints a WARNING and silently falls back to CPU. The job
# then ran the first sweep arm on CPU for eight hours and was killed at the
# time limit having produced zero data points, and all 9,484 nvidia-smi
# samples it collected read "No devices were found".
#
# pyproject.toml documents this fallback, and cluster/gpu_smoke_test.sbatch
# already asserts against it -- but the assertion lived only in the smoke test,
# i.e. only in the job that could afford to fail. Thirty seconds here saves a
# whole time allocation. Assert on device PLATFORM, not on jax.devices()
# merely returning without raising, or a CPU fallback reads as a pass.
echo "=== GPU preflight ==="
python -c "
import jax
devices = jax.devices()
print('JAX devices:', devices)
gpus = [d for d in devices if d.platform == 'gpu']
assert gpus, (
    'ABORT: jax.devices() returned no GPU -- silently fell back to CPU. '
    'This node cannot run the job. Check nvidia-smi output above: if it '
    'reports a device handle error, the NODE is broken, not the code -- '
    'resubmit (SLURM will usually pick a different node) and consider '
    'reporting the node to support@wits-mss.supportsystem.com.'
)
print(f'PASS: {len(gpus)} GPU device(s): {gpus}')
" || { echo "=== GPU preflight FAILED, aborting before spending the allocation ==="; exit 1; }
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

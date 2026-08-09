#!/bin/bash
# Stage 3: the converged UNCONSTRAINED ant baseline.
#
# This single job answers three separate open questions:
#
#   1. WHAT SHOULD --safety_budget BE FOR THE ANT? Currently 150, which is an
#      extrapolation from one untrained reading (108.94 laptop / 102.27
#      cluster), never a measurement of a trained policy. The 500k smoke test
#      showed cost 102 -> 86 -> 101 with reward still ~0, i.e. the ant had not
#      yet learned to pursue goals. That matters because point's unconstrained
#      cost ROSE to 353 once trained (a goal-seeking agent takes far more
#      risk), so ant cost is expected to climb too. If converged cost lands
#      well ABOVE 150 the current budget binds and the safety/performance
#      tradeoff is real; if it lands at ~100 the budget is slack and the
#      constraint is vacuous -- which would gut the research question. Either
#      way this is the number to set the budget from.
#   2. DOES THE ORBAX CHECKPOINT-WRITE PATH WORK ON /home-mscluster? The stage-2
#      smoke test ran with --no_checkpoint, so this is still unvalidated. Better
#      to find out here than ten hours into a morphology run.
#   3. IS THERE A TRAINED ANT WORTH LOOKING AT? No ant policy has ever trained
#      to convergence in this project. 15M steps produces one.
#
# Sizing rationale: 15M steps and num_evals=15 mirror the point unconstrained
# baseline exactly, so the two are directly comparable (that comparison is the
# whole basis for expecting ant cost to rise). Full default eval width
# (num_eval_envs=32 x num_eval_episodes=10) is kept because the 3090 handles it
# -- that is what makes these numbers citable rather than the noisy 8x2 the
# laptop is stuck with. num_envs stays at the proven 256: 512 is permitted by
# validate() (num_envs must divide batch_size*num_minibatches=512) but is
# untested here and larger num_envs super-scales COMPILE time, which is the
# risky part.
#
# Runtime projection from stage-2's measured 668-798 training/sps:
#   15M / ~798 sps ~= 5.2h training, + 15 evals x ~300s ~= 1.25h  =>  ~6.5h.
# --time is set to 12h for ~2x margin without requesting so much that the job
# becomes hard to backfill. bigbatch's cap is 72h.
#
# Submit from the repo root on the cluster:
#   sbatch cluster/ant_unconstrained_baseline.sbatch
# Watch: squeue --me     Read: ant_unconstrained_%j.log
#
# IF IT DIES PARTWAY: checkpoints are written every ~1M steps, so resume with
#   --restore_checkpoint_path <repo>/checkpoints/ant_unconstrained/<step_dir>
# (note the resumed run restarts its own step counter at 0 and will overwrite
# same-named checkpoint dirs -- back up the good one first).
#SBATCH --job-name=ant-unconstrained
#SBATCH --output=ant_unconstrained_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=12:00:00
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
echo "=== unconstrained ant baseline: 15M steps, full-width eval, checkpointing ON ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# python -u is NOT optional. A laptop run reached 573k steps (proven by
# checkpoints on disk) and left a COMPLETELY EMPTY log, because Python
# block-buffers stdout when redirected to a file and a segfault never flushes
# it. Unbuffered output is the difference between diagnosing a failure and
# guessing at it.
#
# Separate checkpoint dir (not the default checkpoints/ant) so this baseline
# cannot clobber, or be clobbered by, other ant runs -- mirrors the existing
# point / point_unconstrained convention. main.py won't auto-discover it, so
# view it with:  python main.py --robot ant --checkpoint checkpoints/ant_unconstrained
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant \
  --penalizer none \
  --num_timesteps 15_000_000 \
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
  echo "PASS. What to read off the eval lines above:"
  echo "  * FINAL eval/episode_cost -- this is the number to set --safety_budget"
  echo "    from. Compare against the current default of 150 and against point's"
  echo "    unconstrained 353. Well above 150 => the budget binds, tradeoff is"
  echo "    real. Around ~100 => budget is slack and needs LOWERING, or the"
  echo "    constraint does nothing."
  echo "  * eval/episode_reward -- confirms the ant actually learned to reach"
  echo "    goals. If it is still ~0, cost is not yet a converged number and the"
  echo "    budget question stays open regardless of what cost says."
  echo "  * training/sps -- schedules the real morphology runs."
else
  echo "FAILED (status $status). If 139, this is the segfault class seen on the"
  echo "laptop appearing here too -- unexpected, since stage 2 passed at this"
  echo "config. Check the last eval line above for how far it got, and resume"
  echo "from the newest checkpoint rather than restarting (see header)."
fi
echo "=== done ==="

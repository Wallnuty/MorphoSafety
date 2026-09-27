#!/bin/bash
# Runs ON the Vast instance. Schaff et al. (ICRA 2019) co-design: the design
# DISTRIBUTION is trained jointly with the policy. This is the Vast twin of
# cluster/ant_morphology_codesign.sbatch -- same flags, same preflights, same
# reading guide. Read that file's header for the method and the sizing
# argument; only what DIFFERS on a rented box is documented here.
#
#   bash cluster/vast/vast.sh codesign          # launches this in tmux
#
# ===================== WHY THIS EXISTS SEPARATELY =========================
#
# `vast.sh train` would run the training command fine, but it runs NO
# preflight. On mscluster that cost an eight-hour allocation once (job 40227:
# a wedged GPU, jax fell back to CPU with a warning, zero data points). A
# rented box bills by the hour from `up`, so the same failure costs money
# rather than queue position. The two asserts below take ~40 s and are the
# cheapest insurance available.
#
# ========================= DIFFERENCES FROM THE SBATCH ====================
#
# 0. 2026-09-25: FRESH RUN ON THE FINAL SAFE-RL RECIPE BY DEFAULT (PPO-
#    Lagrangian b25, 7x7 foot grid, 5 cm linear cost, flip 50, finish line).
#    The Saute-era checkpoints below cannot resume into it (obs 55 -> 250), so
#    RESUME is now empty by default; points 1-3 apply only when you set it.
#
# 1. RESUMING (RESUME=<remote step dir>) continued, on 2026-08,
#    checkpoints/cluster/ant_codesign_lidar_300M/000060293120 -- 60.29M steps
#    from mscluster job 46267, whose design search was measured to be moving
#    far too slowly to commit (see DESIGN_LR below).
#
# 2. THE CHECKPOINT MUST BE PUSHED SEPARATELY. `vast.sh sync` excludes
#    checkpoints/ by design (347 MB locally, none of which the remote needs).
#    `vast.sh push-ckpt <local-dir>` puts one where RESUME expects it. This
#    script REFUSES to start without it rather than silently beginning a fresh
#    search dressed as a continuation.
#
# 3. (resume only) --num_evals 9, not 11. Design iterations are
#    (num_evals-1) * design_updates_per_eval, and the quantity that must not
#    change is EPISODES PER LANE PER ITERATION:
#
#        per_lane = num_timesteps / (num_evals-1) / updates / num_envs / ep_len
#
#      300M / 10 / 8 / 1024 / 2500 = 1.46     <- the original run
#      240M /  8 / 8 / 1024 / 2500 = 1.46     <- this one, unchanged
#
#    Below 1.0 a lane cannot finish a full-length episode, so only early
#    terminations (flips, arrivals) get scored and the stable-but-slow bodies
#    drop silently out of the gradient. 9 keeps the estimator identical to the
#    run being continued, which is what makes the two halves comparable.
#
# ============================ DESIGN_LR = 0.03 ============================
#
# THE HEADLINE CHANGE, and the reason this is worth 12 more hours.
#
# The lr is annealed by (1 - t/tmax) and Adam normalises by sqrt(v-hat), so a
# step moves each gene by ~lr * lr_frac REGARDLESS of gradient magnitude. Total
# mean travel is therefore lr * sum(lr_frac) over the iterations, and that sum
# was ~39.3 across the original 80 iterations:
#
#     lr 1e-3  ->  0.039 of travel in a 2.0-wide gene space   = 2%
#
# Two percent. The distribution could not commit to anything because it was
# never going to move. Confirmed on the checkpoint itself: after 60M steps the
# live components' sigma reads 0.5763 against an init of 0.577 -- unchanged to
# four figures -- and the means sit essentially where U(-0.8, 0.8) put them.
#
# Over the REMAINING 240M, sum(lr_frac) is ~25.6 (it starts at 0.799, not 1.0,
# because the design axis resumes at 60.29M of tmax=300M). So:
#
#     lr 0.03  ->  0.77 of travel                             = 38%
#
# THE COST OF RAISING IT, stated so it is not a surprise. REINFORCE is high
# variance and Adam's step size is gradient-independent, so pure noise random-
# walks the means by ~lr*sqrt(n) = 0.03*8 = 0.24. Signal has to beat that. At
# 1e-3 the noise floor was 0.008 -- and so was everything else. A distribution
# that cannot move is not "safely" converged, it is inert.
#
# ============================== SIZING ====================================
#
# 240M at the 3090's measured TRAINED rate (7127 sps at 1024 envs, from the
# 50M Saute runs -- NOT the 7755 in the throughput sweep, which is untrained
# and therefore optimistic), less ~17% for morphology batching:
#
#   240M / 5900          ~= 11.3 h training
#   9 evals x ~360 s     ~=  0.9 h
#   host model rebuilds  ~=  0.1 h
#   ---------------------------------
#                        ~= 12.3 h  ~= $1.70 at $0.14/hr
#
# Expect the LOW end: co-design resamples bodies every iteration, so the policy
# is repeatedly put onto bodies it has not settled on and may never reach the
# clean-gait regime that earns the trained rate in the first place.
set -uo pipefail
cd /root/MorphoSafety || exit 1

source /opt/conda_profile.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate morpho

echo "=== box ==="
hostname; date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h / | tail -1
echo

# ---- overridable at launch, e.g. DESIGN_LR=0.05 bash cluster/vast/vast.sh codesign
STEPS="${STEPS:-300000000}"
DESIGN_TMAX="${DESIGN_TMAX:-300000000}"
DESIGN_LR="${DESIGN_LR:-0.03}"
# 2026-09-17: score designs on time AND cost. DESIGN_OBJECTIVE=safe_time with
# DESIGN_COST_WEIGHT (decisions per unit of episode cost; 1.0 makes a careless
# nominal-ant crossing's ~58 linear cost units worth ~half its arrival time)
# is the morphology-safety question. Default keeps the time-only objective.
DESIGN_OBJECTIVE="${DESIGN_OBJECTIVE:-speed}"
DESIGN_COST_WEIGHT="${DESIGN_COST_WEIGHT:-0}"
FLIP_COST="${FLIP_COST:-50}"
PENALIZER="${PENALIZER:-ppo_lagrangian}"   # none = the Schaff baseline
BUDGET="${BUDGET:-25}"
MULT_LR="${MULT_LR:-1.5e-5}"
MULT_MAX="${MULT_MAX:-3.0}"
MULT_INIT="${MULT_INIT:-0.01}"
GRID="${GRID:-7}"
# 11 evals over 300M = 80 design iterations at 1.46 episodes/lane (see 3.);
# for a resume of the 2026-08 run use NUM_EVALS=9 STEPS=240000000.
NUM_EVALS="${NUM_EVALS:-11}"
RESUME="${RESUME:-}"
NAME="${NAME:-ant_codesign_lagrangian_b${BUDGET}_grid${GRID}_r$(date -u +%m%d%H%M)}"
CKPT="/root/MorphoSafety/checkpoints/$NAME"

# PREFLIGHT 1: a GPU that JAX can actually SEE -- assert on device.platform,
# not on `import jax` merely not raising. A missing or mismatched CUDA plugin
# makes jax fall back to CPU silently.
echo "=== GPU preflight ==="
python -c "
import jax
d = jax.devices(); print('JAX devices:', d)
gpus = [x for x in d if x.platform == 'gpu']
assert gpus, 'ABORT: no GPU -- jax fell back to CPU silently. Destroy and re-rent.'
print('PASS:', gpus)
" || { echo "=== GPU preflight FAILED, no compute spent ==="; exit 1; }
echo

# PREFLIGHT 2: the synced code is new enough, and the observation is the width
# this run is FOR.
#
# 250 = 47 (proprio + goal + lidar) + 196 (4 feet x 7x7 grid) + 7 genes; no
# budget scalar (that was Saute's). A silent 54 would mean the foot grid is
# off; a silent 243 would mean morphology conditioning never reaches the
# observation, training a policy that cannot see which body it is driving --
# the whole method gone, with a normal-looking log. rsync makes a stale checkout unlikely here, unlike the git-clone path
# on mscluster, but the assert is free.
echo "=== code / observation preflight ==="
python -c "
from mjx_safety_gym import design, morphology
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T

for name in ('save', 'load', 'state_dict', 'load_state_dict'):
    assert hasattr(design.DesignLoop, name), (
        f'ABORT: design.DesignLoop has no {name}() -- this tree predates '
        f'design-state checkpointing (2026-08-24). Re-sync.')
assert 'design_tmax' in {a.dest for a in T.build_argparser()._actions}, (
    'ABORT: no --design_tmax; a resume would anneal the design lr against the '
    'wrong horizon. Re-sync.')

p = T.build_argparser().parse_args(['--robot', 'ant', '--task', 'minefield'])
T.apply_robot_defaults(p)
assert (p.penalizer, p.safety_budget, p.foot_hazard_grid, p.flip_cost,
        p.corridor_walls, p.terminate_out_of_bounds, p.exit_cost,
        p.solver_iterations, p.solver_ls_iterations) == (
    'ppo_lagrangian', 25.0, 7, 50.0, False, True, 50.0, 4, 8), (
    f'ABORT: trainer defaults are not the 2026-09-25 recipe: {p.penalizer} '
    f'b{p.safety_budget} grid{p.foot_hazard_grid} flip{p.flip_cost}. Re-sync.')
kw = dict(T.robot_env_kwargs('ant'))
kw['morphology_conditioning'] = True
env = Minefield(**kw)
assert env._ground_contact_eps == 0.05 and env._hazard_cost_shape == 'linear', (
    'ABORT: cost is not linear at the 5 cm gate. Re-sync.')
w = env.observation_size
assert w == 243 + morphology.NUM_GENES, f'expected 250 (grid 7 + genes), got {w}'
o = env.mj_model.opt
assert (o.iterations, o.ls_iterations) == (4, 8), f'ABORT: solver caps {o.iterations}/{o.ls_iterations}, expected 4/8'
print(f'PASS: obs {w} (grid 7 + genes), 5 cm linear cost, design checkpointing present')
import jax, numpy as np, jax.numpy as jp
a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
ref = a.astype(np.float64) @ a.astype(np.float64)
err = float(np.abs(np.asarray(jax.jit(jp.matmul)(a, a), np.float64) - ref).max() / np.abs(ref).max())
prec = jax.config.jax_default_matmul_precision
assert prec == 'highest' and err < 1e-5, (
    f'ABORT: matmuls are not float32 here (precision={prec}, rel err {err:.1e}; '
    f'TF32 is ~3e-4). Unset JAX_DEFAULT_MATMUL_PRECISION, or the checkout predates '
    f'mjx_safety_gym/numerics.py (2026-09-26).')
print(f'PASS: float32 matmuls on {jax.default_backend()} (rel err {err:.1e})')
" || { echo "=== code preflight FAILED, no compute spent ==="; exit 1; }
echo

# PREFLIGHT 3: the thing being resumed FROM actually arrived.
#
# Without design_state.npz the policy resumes onto a FRESH distribution -- the
# search silently restarts from its init while the log reads like a
# continuation. train.py warns, but a warning 12 hours into a paid run is not
# a control. Refuse instead.
if [ -n "$RESUME" ]; then
echo "=== resume preflight ==="
[ -d "$RESUME" ] || {
  echo "ABORT: RESUME=$RESUME is not a directory."
  echo "Push it first, from the laptop:"
  echo "  bash cluster/vast/vast.sh push-ckpt checkpoints/cluster/ant_codesign_lidar_300M/000060293120"
  exit 1; }
[ -f "$RESUME/design_state.npz" ] || {
  echo "ABORT: no design_state.npz in $RESUME. Resuming would restart the"
  echo "design search from its init while the policy continued. Refusing."
  exit 1; }
python -c "
import numpy as np, sys
z = np.load('$RESUME/design_state.npz', allow_pickle=True)
lm = z['gmm_log_mixprobs']; live = int((lm > -1e5).sum())
print(f'PASS: design axis resumes at {int(z[\"last_step\"]):,}, '
      f'{live} of {lm.size} components live, last chop '
      f'{int(z[\"last_chop\"]):,}')
sig = np.exp(z['gmm_log_stds'][lm > -1e5]).mean()
print(f'      mean sigma of live components: {sig:.4f}  (init 0.577)')
" || exit 1
[ "$CKPT" = "$RESUME" ] && { echo "ABORT: output dir == resume dir"; exit 1; }
  set -- --restore_checkpoint_path "$RESUME"
else
  echo "=== fresh run (no RESUME) ==="
  set --
fi
echo

# 0.75 of 24 GB -- the same fraction the mscluster jobs use, left alone so a
# VRAM problem here is directly comparable to one there.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

echo "=========================================================="
echo " co-design $([ -n "$RESUME" ] && echo "RESUME from $RESUME" || echo FRESH)"
echo "   steps       $STEPS  (design_tmax $DESIGN_TMAX)"
echo "   design_lr   $DESIGN_LR   objective $DESIGN_OBJECTIVE"
echo "   safe RL     $PENALIZER b$BUDGET  lr=$MULT_LR cap=$MULT_MAX init=$MULT_INIT"
echo "   env         grid $GRID  flip $FLIP_COST  linear 5 cm  finish line  float32 matmuls"
echo "   num_evals   $NUM_EVALS"
echo "   out         $CKPT"
echo "=========================================================="
echo

# python -u is NOT optional: Python block-buffers stdout when redirected and a
# segfault never flushes it. A laptop run once reached 573k steps -- proven by
# checkpoints on disk -- and left a completely empty log.
#
# action_repeat / episode_length / discounting / healthy_reward /
# terminate_on_flip / terminate_on_goal are deliberately NOT passed: they
# resolve from _ROBOT_DEFAULTS and print on the first line of the log. Pinning
# them here would freeze this run to whatever was correct the day it was
# written.
time python -u -m mjx_safety_gym.algorithms.train_ppo \
  --robot ant --task minefield \
  --penalizer "$PENALIZER" --safety_budget "$BUDGET" --adaptive_budget_horizon \
  --lagrangian_multiplier_lr "$MULT_LR" --lagrangian_multiplier_max "$MULT_MAX" \
  --initial_lagrange_multiplier "$MULT_INIT" \
  --hazard_size 0.16 --hazard_lidar --no-foot_obstacle_obs \
  --foot_hazard_grid "$GRID" --hazard_cost_shape linear --finish_line \
  --matmul_precision highest --solver_iterations 4 --solver_ls_iterations 8 \
  --no-corridor_walls --terminate_out_of_bounds --exit_cost 50 --boundary_cost_weight 0 \
  --design_optimization \
  --design_objective "$DESIGN_OBJECTIVE" --design_cost_weight "$DESIGN_COST_WEIGHT" \
  --flip_cost "$FLIP_COST" \
  --num_morphologies 16 \
  --design_components 8 \
  --design_updates_per_eval 8 \
  --chop_freq 60000000 \
  --steps_after_design_update 30000000 \
  --num_envs 1024 --num_minibatches 32 \
  --design_lr "$DESIGN_LR" \
  --num_timesteps "$STEPS" \
  --design_tmax "$DESIGN_TMAX" \
  --num_evals "$NUM_EVALS" \
  --checkpoint_logdir "$CKPT" "$@"
status=$?

echo
echo "=== exit status: $status ==="
ls -1 "$CKPT" 2>&1 | tail -5
du -sh "$CKPT" 2>&1
echo "--- design state written? (a further resume needs this non-empty) ---"
ls -1 "$CKPT"/*/design_state.npz 2>&1 | tail -3
echo
echo "PULL BEFORE YOU DESTROY:  bash cluster/vast/vast.sh pull"

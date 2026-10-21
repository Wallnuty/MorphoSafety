# Experiments (2026-10-19)

All on the random minefield, budget 2, 3 seeds each:

| Condition | Question it answers |
|---|---|
| Nominal ant, PPO-Lagrangian | the fixed-body baseline |
| Co-design, speed ranking (Schaff) | does co-design beat the fixed body? |
| Co-design, feasibility-first ranking | does the design search need to know about the constraint? |
| Each evolved body retrained from scratch | is it the body, or the extra training? |
| Unconstrained co-design | does safety change which body evolves? |
| Unconstrained body retrained with the constraint | is evolving under the constraint better than adding safety afterwards? |
| Simulator benchmark | MorphoSafety vs mjx-safety-gym vs Safety-Gymnasium, same tasks and physics |

So far only partial runs exist, pulled to `checkpoints/vast/`:
- the first halves of three feasibility-first co-design runs (`cd_ff_s0`, `cd_ff_s1`, `cd_ff_s2`)
- the first halves of two unconstrained co-design runs (`cd_unc_s0`, `cd_unc_s2`)

The rest never ran because the Vast credit ran out.

All commands below run from `~/MorphologyResearch/MorphoSafety`. Let `PY=~/miniconda3/envs/mjx-warp/bin/python`.

## Watch the environments (MuJoCo viewer)

`main.py` replays a trained policy in a viewer window. Without `--checkpoint` it replays the newest checkpoint for the robot, or random actions if none exist. `JAX_PLATFORMS=cpu` keeps the replay in real time; on the GPU the viewer runs in slow motion.

Add any of these to a command:
- `--deterministic`: show the mean action instead of a sample.
- `--duration 60`: how long to run, in seconds.
- `--seed N`: a different layout or episode.

**LaserField** (sparse movie lasers):
```
JAX_PLATFORMS=cpu $PY main.py --robot ant --task laserfield --beam_obs segments \
  --checkpoint checkpoints/laserfield_v2_b1_300M
```

**LaserWeb** (dense ant-scale web):
```
JAX_PLATFORMS=cpu $PY main.py --robot ant --task laserweb --web_style ant --web_obs objects \
  --num_beams 16 --checkpoint checkpoints/antweb_v10C_n16
```

**Random minefield, nominal ant.** No single-body random-field run exists yet. This policy was trained on the lattice and is shown on random fields:
```
JAX_PLATFORMS=cpu $PY main.py --robot ant --task minefield --hazard_placement random \
  --checkpoint checkpoints/vast/ant_minefield_final_eps2_b0_250M
```

**Random minefield, evolved body** (the current mode of a co-design run; change `cd_ff_s1` to see another run):
```
JAX_PLATFORMS=cpu $PY main.py --robot ant --task minefield --hazard_placement random \
  --design_from checkpoints/vast/cd_ff_s1 --checkpoint checkpoints/vast/cd_ff_s1
```

## Evaluate (numbers, many episodes)

`scripts/eval_design.py` reports return, speed, cost, hazard steps, hazard-free %, flipped %, out-of-bounds % and under-budget %. A co-design run is scored on its evolved body, a single-body run on its own body. Add `--json <file>` to save the results.

```
$PY scripts/eval_design.py --task laserfield --beam_obs segments \
  --run checkpoints/laserfield_v2_b1_300M --budget 1 --seeds 0 1 2
$PY scripts/eval_design.py --task laserweb --web_style ant --web_obs objects \
  --run checkpoints/antweb_v10C_n16 --budget 3 --seeds 0 1 2
$PY scripts/eval_design.py --task minefield --hazard_placement random \
  --run checkpoints/vast/cd_ff_s1 --budget 2 --seeds 0 1 2
```

Without `JAX_PLATFORMS=cpu` it uses the GPU, which is much faster here; with it, it runs on the CPU.

## Train

Defaults supply the rest: 4096 envs, MuJoCo Warp, flip and exit cost 50, the 7×7 foot grid.

LaserField:
```
$PY -u -m mjx_safety_gym.algorithms.train_ppo --robot ant --task laserfield --beam_obs segments \
  --safety_budget 1 --matmul_precision highest --num_timesteps 300000000 --num_evals 31 \
  --checkpoint_logdir $PWD/checkpoints/laserfield_b1_300M
```

LaserWeb:
```
$PY -u -m mjx_safety_gym.algorithms.train_ppo --robot ant --task laserweb --web_style ant --web_obs objects \
  --safety_budget 3 --lagrangian_multiplier_max 1.5 --matmul_precision highest \
  --num_timesteps 300000000 --num_evals 31 --checkpoint_logdir $PWD/checkpoints/laserweb_ant_b3_300M
```

Random minefield, nominal ant (add `--body_from checkpoints/<co-design run>` to train an evolved body instead):
```
$PY -u -m mjx_safety_gym.algorithms.train_ppo --robot ant --task minefield --hazard_placement random \
  --safety_budget 2 --matmul_precision highest --num_timesteps 300000000 --num_evals 11 \
  --checkpoint_logdir $PWD/checkpoints/minefield_random_b2_300M
```

Random minefield, co-design. Use `--design_ranking objective` for speed ranking, and `--seed N` for other seeds. It takes about 2 h on a 3090.
```
$PY -u -m mjx_safety_gym.algorithms.train_ppo --robot ant --task minefield --hazard_placement random \
  --safety_budget 2 --design_optimization --num_morphologies 64 --design_components 8 --design_lr 0.08 \
  --design_updates_per_eval 3 --num_evals 41 --num_timesteps 2000000000 \
  --chop_freq 200000000 --steps_before_design_update 200000000 --steps_after_design_update 300000000 \
  --design_ranking feasibility_first --num_eval_envs 64 --num_eval_episodes 5 \
  --matmul_precision highest --checkpoint_logdir $PWD/checkpoints/cd_ff_random_b2
```

## Paper

`paper/gecco_draft/main.tex` has Algorithm 1, short plain-language pseudocode of the method. The Method text explains the feasibility-first ranking, and the Setup lists the conditions above.

## Comparisons

- the nominal ant with PPO-Lagrangian (does the body matter at all?)
- the evolved body retrained from scratch (better body, or just more training?)
- unconstrained co-design (does safety change which body evolves?)
- speed-only vs feasibility-first ranking (does the search need constraint handling?)

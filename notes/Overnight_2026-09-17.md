# Overnight 2026-09-17 — graded cost, flip cost, foot grid; Vast runs

User's instructions before bed: hazards detect up to 5 cm; implement all four
source-backed changes (graded cost IN the cost, no ring outside the disc,
flip cost, foot-resolution obs); verify; run on Vast (credit $5.73 at start);
keep working until manually stopped.

## Code (all uncommitted — user commits)
- `ground_contact_eps` default 0.001 -> 0.05 x arena_scale.
- `--hazard_cost_shape {linear,quadratic,binary}` default linear: cost per
  hazard = penetration depth (Safety Gym's rule); quadratic = CRAX Pathway.
  Binary count always logged as `eval/episode_hazard_steps`.
- `--hazard_shaping_radius` default = the disc (was 0.25 in the sbatch).
- `--flip_cost` (RunForward): charged once on the upright->flipped transition.
- `--foot_hazard_grid N` (+ `_spacing`, 0.08): per-foot NxN map of the linear
  cost that foot would pay landing there, torso-yaw frame, centred on the TOE.
  N=5 -> obs 47 -> 147. Checkpoint-width candidates extended.
- sbatch + new `cluster/vast/remote_safe.sh` / `vast.sh safe` (SAFE_* vars,
  SAFE_SESSION, SAFE_AFTER queueing).
- Verified on CPU: widths 47/147/63/163; linear/quadratic/binary values on a
  settled ant; grid centre cell == foot's own depth; 5 cm gate (4 cm grounded,
  6 cm not); flip transition indicator; end-to-end tiny training run with all
  flags (eval line carries episode_hazard_steps).

## Vast instances (destroy with `vastai destroy instance <id> -y`)
- a: 51256737 (destroyed 00:17Z, 350 KB/s network) -> replaced by   b: 51256740   c: 51256747

## Runs (fill in)

## Cost scale at 5 cm (Saute500 walker, 32 episodes, 10.22 m, same trajectories)
    binary @1mm (2026-09-14 scale)     32.7    3.20 /m
    hazard_steps binary @5cm          118.6   11.61 /m
    cost LINEAR @5cm (new default)     57.9    5.67 /m   <- budgets in these units
    cost quadratic @5cm                37.7    3.69 /m
Budget 25 = ~45% of a careless crossing. flip_cost 50 = ~one careless crossing.
Multiplier lr 1.5e-5 (|constraint| ~0.7x the original 2 cm scale's).

## Plan (each box runs two 50M runs back to back, ~2.3 h each)
    box a: A1 none 47 flip50            -> A2 Lagrangian b25 warm from A1
    box b: B1 none grid5 (147) flip50   -> B2 Lagrangian b25 grid5 warm from B1
    box c: C1 Lagrangian b25 flip50 from scratch (47) -> C2 decided from results
All arms: linear cost, 5 cm, flip_cost 50, adaptive budget, 1024/32, 256x4.
    box c: C1 -> C2 Lagrangian b25 grid5 flip50 FROM SCRATCH (147): with A2/B2
           (warm) this makes a 2x2 of {grid, no grid} x {warm, scratch}.
Launched: B1 23:50Z, C1 23:51Z, B2 queued behind b1 00:04Z, C2 queued behind c1.
- a (new): 51259163
- a (2nd replacement, 51259163 never booted): 51260116
A1 launched 00:38Z, A2 queued behind a1

## Also built tonight: cost-aware co-design objective
`--design_objective safe_time --design_cost_weight W`: design fitness = time
fitness + W x episode cost (EpisodeStatsWrapper now latches `ep_cost_last`).
Exposed as DESIGN_OBJECTIVE / DESIGN_COST_WEIGHT / FLIP_COST in
cluster/ant_morphology_codesign.sbatch and cluster/vast/remote_codesign.sh.
Verified: unit arithmetic; CPU end-to-end (2 morphologies); laptop GPU 3M-step
run at 512 envs x 8 morphologies exit 0, design/cost_mean logged. This is the
next experiment once a single body shows placement: which body makes the safe
crossing cheap.

## Progress log (UTC)
    00:37  B1 10M: reward 19.4, cost 88.7, hz_steps 176 | C1 10M: 20.8, 116, 254, lambda 0.58
    00:57  B1 20M: 21.0, 66.8, 146, ep_len 497        | C1 20M: 21.4, 97.1, 217, lambda 1.60
    02:21  B1 DONE (none+grid): reward 21.2, displ 10.56 m, cost 51.1, 4.8/m, hz_steps 116, ep_len 401
           -> careless walker; the grid alone changes nothing (nothing pushes it to use it).
    02:23  C1 DONE (Lagrangian b25 scratch, 47): reward 21.2, displ 10.55 m, cost 78.6, 7.5/m,
           hz_steps 179, ep_len 642, lambda 4.08 and still rising linearly.
           -> same pathology as 2026-09-12: pressure made the gait SLOWER and WORSE per metre
              (7.5/m vs 4.8/m unconstrained), cost never approached B=25. No collapse yet at 50M.
    02:23  B2 started (Lagrangian b25 + grid, warm from B1; network + Adam restored, multiplier fresh).
           C2 started (Lagrangian b25 + grid from scratch).
    03:20  A1 DONE (control, 47): reward 20.9, displ 10.4 m, cost 63.1, 6.0/m, hz_steps 138, ep_len 498.
           The new-scale control. B1 (grid) was 51.1 / 4.8/m at the same reward -- 19% lower, single seed.
    03:21  A2 started (Lagrangian b25 warm from A1).
    03:51  B2 30M (Lag+grid warm): cost 38.5 (from 51.1), hz_steps 91, reward 21.1, lambda 1.11 -- falling, traverse intact.
           C2 25M (Lag+grid scratch): cost 53.2, hz_steps 120, reward 21.0, lambda 1.48 -- vs C1 (no grid) 87.9 at 25M.
           A2 10M (Lag warm, no grid): cost 62.2, flat so far, lambda 0.50.
    04:53  B2 DONE (Lag b25 + grid, warm from B1): cost 33.0 (start 51.1, -35%), hz_steps 81 (-30%),
           reward 21.5 (unchanged, full traverse), ep_len 531, lambda 1.49 -- NOT winding up.
           ~3.1/m vs control 6.0/m (A1) and 4.8/m (B1). FIRST constrained run in the project
           where cost fell substantially with the traverse intact.
           A2 30M (same, NO grid): cost 61.3 -- flat since its start (61.2); lambda 1.68.
           -> the foot grid is what makes the constraint act on placement.
    04:56  B3 launched: B2 continued +50M (multiplier restored), to see if it reaches B=25.
    05:03  C2 DONE (Lag b25 + grid, scratch): cost 44.8, hz_steps 106, reward 21.1, lambda 2.49, ep_len 510.
           vs C1 (no grid, scratch) 78.6: -43%. Below BOTH unconstrained controls (63.1 / 51.1).
    05:06  box c logs + C1/C2 final checkpoints pulled to checkpoints/vast/; box c destroyed.
           MISTAKE: C2's checkpoint was NOT pulled (wrong name in the pull: the run is
           ..._adaptive_grid5_flip50, I asked for ..._flip50_grid5) and box c was destroyed
           before I noticed. C2 survives as its log only (logs/vast/2026-09-17/c/). C1's
           checkpoint was pulled. Rerun C2 on the cluster if its weights are wanted.
    05:54  A2 DONE (Lag b25 warm from A1, NO grid): cost 52.8 (start 61.2, -14%), hz_steps 112,
           reward 20.6, lambda 2.80. Box a destroyed after pulling A1/A2 checkpoints + logs.

## RESULT TABLE (50M each; cost = linear penetration @5cm + flip 50; budget 25; 1024/32; 256x4)

    arm  obs   penalizer     start      reward  displ   cost   cost/m  hz_steps  lambda@50M
    A1   47    none          scratch    20.9    10.4    63.1    6.1     138        --
    B1   147   none          scratch    21.2    10.6    51.1    4.8     116        --
    C1   47    Lagrangian    scratch    21.2    10.6    78.6    7.5     179       4.08 rising
    C2   147   Lagrangian    scratch    21.1    10.5    44.8    4.3     106       2.49
    A2   47    Lagrangian    warm(A1)   20.6    10.2    52.8    5.2     112       2.80
    B2   147   Lagrangian    warm(B1)   21.5    10.7    33.0    3.1      81       1.49
    B3   147   Lagrangian    B2 +50M    (running; 24.99 at 65M -- AT the budget, reward 21.2)

    Every arm crosses the full corridor (reward 20.6-21.5). Read cost/m.

## What it says
1. With the graded cost, the 5 cm gate, the flip cost and the FOOT GRID, PPO-Lagrangian
   lowers hazard cost with the traverse intact -- B2: -35% from its own start, ~half the
   control's cost per metre, and its continuation B3 reached the budget (25) at 65M with
   reward unchanged. First time in the project a constraint was satisfied by placement
   rather than by doing less of the task.
2. Without the grid the same recipe does what it always did: C1 is WORSE than unconstrained
   (7.5/m vs 6.1/m, lambda still climbing linearly); A2 moved only 14%. The observation was
   the missing piece, not the optimiser -- consistent with Hwang et al.'s foot-map ablation.
3. From scratch with the grid (C2) also beats both unconstrained controls (4.3/m), so the
   warm start helps but is not required once the policy can see its feet against the map.
4. The unconstrained grid control B1 (4.8/m) is 19% below A1 (6.1/m) -- single seed, and
   the project has measured ~25% seed variance, so do not read that as an effect yet.
5. Caveats: one seed per arm; eval cost includes flip_cost x flip rate (hz_steps is the
   clean placement count: B2 81 vs B1 116, -30%); the 5 cm gate charges feet swinging low.

## Next (in order)
- Read B3's final table (box b, session b3; destroy box b afterwards:
  `vastai destroy instance 51256740 -y`, then `rm cluster/vast/.instance.b`).
- Second seed of B2/A2 on the cluster (commands in the sbatch header) to pin the grid effect.
- Then the research question proper: co-design with `--design_objective safe_time`
  (built and smoke-tested tonight; DESIGN_OBJECTIVE/DESIGN_COST_WEIGHT/FLIP_COST in the
  codesign scripts) with the grid obs -- which body makes the safe crossing cheap.
- View B2: python main.py --robot ant --task minefield \
    --checkpoint checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm --seed 3
  (width 147 reconciles to foot_hazard_grid=5 automatically).
    07:20  B3 DONE (B2 continued to 100M): cost 23.6 (budget 25) -- UNDER BUDGET from 65M on,
           hz_steps 64 (from 116 at B1), reward 21.3, displ 10.6 m, lambda 1.33 and FALLING.
           2.2 cost/m vs the control's 6.1. Checkpoint pulled:
           checkpoints/vast/ant_minefield_ppo_lagrangian_b25_grid5_flip50_warm_100M/000050380800
    07:22  box b destroyed. No live instances. Credit left $1.89 (started $5.73).

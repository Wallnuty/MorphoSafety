# Moving the simulator from MJX-JAX to MuJoCo Warp

Methods log, 2026-10-03 / 04. It records what changed, what was measured, how,
and what is still unverified, so the switch can be described and defended in a
paper. Every number below comes from a run whose script and log are listed in
[Reproduction](#reproduction). All of it was done on the `warp-probe` branch
(checkout `~/MorphologyResearch/MorphoSafety-warp`) in a separate conda env
(`mjx-warp`); the existing env and checkout were not modified.

## Summary

* **Speed.** On the laptop GPU (RTX 4050, 6 GB), training throughput went from
  13.8k to 73.3k env steps/s for the single ant (512 envs) and from 19.0k to
  100.0k for co-design (1024 envs, 16 bodies): **5.3x** in both cases. The
  single ant at 1024 envs, in the 300M quality run, trains at 95-109k.
* **Where the speed comes from.** Not the MuJoCo upgrade (3.14 on the old JAX
  backend: 14.1k, unchanged), but the Warp backend: it only does work for
  contacts and constraints that are actually active, and its solver stops once
  every world has converged.
* **Solver cap no longer needed.** On MJX-JAX we capped the Newton solver at
  4 iterations / 8 line-search steps for speed (2026-09-27). On Warp the
  uncapped solver (MuJoCo's default 100 / 50, i.e. run to convergence) is the
  *fastest* configuration, so the new setup simulates fully converged contact
  physics.
* **Same physics.** With the solver converged, MuJoCo 3.3.2 (old) and 3.14
  (new) agree to float32 rounding after one step from identical states, and the
  Warp and JAX backends agree to ~1e-7, including with a different body in every
  environment (the co-design setting).
* **One behavioural change to know about.** Policies trained on the old capped
  solver are sensitive to how the unfinished solve is computed: the same budget-0
  policy is 4.5% slower (1.141 vs 1.195 m/s, 12 SE) on 3.14 even on the JAX
  backend, with cost, flips and exits unchanged. Results from the old and new
  setups must therefore not be mixed in one comparison; the reference runs need
  repeating on the new setup.
* **Learning quality holds.** Repeating the old 300M final baseline on the new
  setup (same flags, same seed) gives the same or slightly better result: at
  241-261M, 1.52 vs 1.45 m/s, cost 26.3 vs 27.4 (budget 25), 0.3% vs 2.6%
  flips. It trained in 46 min on the laptop; the old run needed 1.7 h on a 3090.
* **Cluster.** On an RTX 3090 (mscluster): 38.2k -> 118.3k single ant (3.1x),
  40.4k -> 133.3k co-design (3.3x), no overflow warnings. A 2B co-design run
  would take ~4 h of training instead of ~14 h.
* **Adopted 2026-10-04.** Merged into `main` as the default (see
  [Adoption](#adoption-2026-10-04)); the vase cost now uses contact sensors and
  matches the old one step for step. The old stack stays reproducible from the
  `pre-warp` tag.
* **More envs per GPU help on a 3090:** 4096 envs train 1.73-1.77x faster
  than 1024 (233k steps/s co-design), 8192 ~1.95x; co-design then needs
  longer design iterations (section 3b).
* **Still owed:** repeating the reference runs on the new setup before mixing
  old and new results, and checking that learning holds at a larger env count.

## Why we looked

The question was whether evolving robot *topology* is limited by MJX. Part of
the answer was that MJX's cost scales with *possible* contacts rather than
active ones, because JAX needs static array shapes (MuJoCo Playground paper,
limitations section: "computation time related to contacts does not scale like
the number of active contacts in the scene, but like the number of possible
contacts"; it names Warp and Taichi as the way around it). We had already paid
for this: rarely touched corridor walls cost 27% of throughput (2026-09-26)
because they added 34 contact slots (188 -> 324 constraint rows per env). The
MJX package now ships a Warp backend (`impl='warp'`), so we measured it.

## Setup

| | old (all results up to 2026-10-03) | new (this probe) |
|---|---|---|
| MuJoCo / MJX | 3.3.2 / 3.3.2 | 3.14.0 / 3.14.0, + mujoco-warp 3.14.0, warp-lang 1.17.0 |
| backend | MJX-JAX | MJX-Warp (`impl='warp'`) |
| solver | Newton, 4 iterations / 8 line search | Newton, 100 / 50 (MuJoCo default; converges and exits early) |
| JAX | 0.10.2 | 0.10.2 (unchanged) |
| brax / playground | forks Andrew-Luo1/brax@aeda2a42 (0.12.3) / mujoco_playground_new@179d9801 (0.0.4) | brax 0.14.2 / playground 0.2.0 (PyPI) |
| flax / optax / orbax | 0.10.6 / 0.2.8 / (old) | 0.12.8 / 0.2.8 / 0.12.6 |

Unchanged throughout: physics timestep 0.01 s, RK4 integrator, pyramidal friction
cones, solver tolerance 1e-8, line-search tolerance 0.01, float32 matmuls; one
env step = 2 physics steps (0.02 s), one policy decision = 4 env steps (0.08 s).
Hardware for every number here: NVIDIA RTX 4050 Laptop GPU (6 GB), driver
610.88, WSL2. The exact package list of the new env is
`cluster/warp/requirements-warp.txt`.

## Experiments and results

### 1. Is MuJoCo 3.14 the same physics as 3.3.2? (JAX backend, CPU)

Same ant model, same states, same actions, both versions:

* **One step from identical states.** 60 states recorded along a random-action
  rollout in 3.3.2; from each, one step in each version with a cold warm-start.
  Script `scripts/warp_probe/physics_one_step.py`.

  | solver | median max abs. difference in qvel | worst | worst qpos |
  |---|---|---|---|
  | converged (100 / 50) | 1.5e-7 | 9.5e-7 | 1.2e-7 |
  | capped (4 / 8, our old setting) | 1.2e-7 | 1.4e-3 | 1.5e-5 |

  Converged, the two versions agree to float32 rounding: the physics model is
  identical. Capped, a few contact-heavy states differ by ~1e-3 in velocity,
  because an unfinished solve depends on implementation details that changed
  between versions.
* **Rollout divergence.** 50 random-action steps from the same reset
  (`scripts/warp_probe/physics_rollout.py`): differences 7e-9 (qpos) after one
  step, 3e-8 after five, 2e-3 (qpos) / 0.44 (qvel) after 50. The growth is the
  usual amplification of rounding differences by contact-rich legged dynamics.
* **Policy level.** The budget-0, 2 cm single ant
  (`checkpoints/vast/ant_minefield_final_eps2_b0_250M`, step 250,880,000),
  evaluated by the trainer's own evaluator, 512 episodes (64 envs x 8), seed 0,
  old solver cap in both:

  | MuJoCo | return | speed (m/s) | cost | hazard_steps | flipped | out of bounds |
  |---|---|---|---|---|---|---|
  | 3.3.2 | 11.05 | 1.195 | 1.49 | 3.56 | 0.78% | 0.98% |
  | 3.14 | 11.08 | 1.141 | 1.55 | 5.15 | 0.78% | 0.59% |
  | difference / SE | +0.6 | **-12.2** | +0.4 | +3.4 | 0.0 | -0.7 |

  The speed loss is real; cost, return, flips and exits are unchanged. A policy
  trained on the capped solver has adapted to that solver's specific errors, so
  changing them (here by a version upgrade) costs it some speed.

### 2. Are the Warp and JAX backends the same physics? (GPU)

* 16 environments, reset plus 50 zero-action steps: identical torso heights
  (0.159 m) and rewards on both backends.
* **Different body per environment** (what co-design needs): two environments,
  nominal ant and an extreme body (genes 1, 1, 1, 1, 0.5, 0.3, 0.1), batched
  through our morphology code, 60 random-action steps
  (`scripts/warp_probe/per_lane_bodies.py`). Warp vs JAX, per environment: max
  difference 3e-8 / 4e-9 over the first 10 steps, 1.7e-7 / 4.8e-5 over 60. The
  two bodies differ from each other by 0.062 on both backends, so each Warp
  world simulates its own body, matching JAX.

### 3. Throughput

**Training** (the real trainer, 6M env steps, seed 0, default recipe; single ant
at the trainer default of 512 envs, co-design at 1024 envs with 16 bodies and
8 Gaussians; logs `notes/warp_probe_logs/bench_*.log`):

| run | MuJoCo | backend | solver | single ant | co-design |
|---|---|---|---|---|---|
| A / F | 3.3.2 | JAX | 4 / 8 | 13.8k | 19.0k |
| B | 3.14 | JAX | 4 / 8 | 14.1k | - |
| C / G | 3.14 | Warp | 4 / 8 | 37.4k* | 43.1k* |
| D / H | 3.14 | Warp | 100 / 50 | **73.3k** | **100.0k** |

\* The capped Warp runs printed an overflow warning every time the solver hit
its cap: 4.76M and 3.87M warning lines respectively. Printing that much costs
time, so these two numbers are probably understated and should not be used to
argue about the cap on Warp. The headline comparison (A/F vs D/H) is
unaffected: neither side printed any warnings.

**Cluster RTX 3090** (`cluster/warp_sps_check.sbatch`, job 63512 on mscluster49,
driver 595.71.05; all four runs in one job on one node, 20M steps each, 1024
envs / 32 minibatches for both workloads; log
`notes/warp_probe_logs/cluster_warp_sps_63512.log`):

| workload | MJX-JAX 3.3.2, solver 4/8 | MJX-Warp 3.14, solver 100/50 | speed-up | Warp overflow warnings |
|---|---|---|---|---|
| single ant | 38.2k | 118.3k | 3.10x | 0 |
| co-design (16 bodies, 8 Gaussians) | 40.4k | 133.3k | 3.30x | 0 |

The ratio is smaller than on the laptop because the old stack gains much more
from the bigger GPU (single ant 13.8k at 512 envs on the 4050 vs 38.2k here)
while Warp barely does (~113k on the 4050 vs 118k here, both at 1024 envs). At
1024 envs Warp is therefore not limited by GPU compute; per-step overhead
(kernel launches, the PPO update) is the likely limit. Hypothesis, untested:
more envs per GPU would widen the gap on a 3090. Wall clock per 20M-step run
(including JAX compile, first-use Warp kernel compilation, two evaluations):
single ant 12.6 min -> 4.8 min. For a 2B co-design run, 133k steps/s means
~4.2 h of training instead of the ~13.9 h the last runs took.

**Pure physics** (no learning; 1024 envs, random actions, 100-step jitted scan
after warm-up; `scripts/warp_probe/step_throughput.py`), env steps/s:

| backend | solver 4 / 8 | solver 100 / 50 |
|---|---|---|
| JAX | 21.7k | 4.4k |
| Warp | 47.2k | 140.1k |

On MJX-JAX the line search is a fixed-length scan, so every allowed line-search
step runs every iteration (found 2026-09-27, when the line search turned out to
dominate step time), and under `vmap` the solver loop runs until the slowest of
the 1024 environments converges. That is why the cap was worth 5x there. The
MuJoCo Warp docs state that its solver exits early once all worlds have
converged; whatever the exact mechanism, on Warp the full solver measured as
both the accurate and the fast choice.

**Buffer use** (`scripts/warp_probe/buffer_usage.py`, full solver, 1024 envs,
300 random-action steps): at most 1,310 contacts across all 1024 worlds
(capacity 65,536) and 20 constraint rows per world (capacity 256), against the
~188 rows per env that MJX-JAX processes whatever is touching. That gap is the
mechanism behind the speed-up.

### 3b. Envs per GPU (RTX 3090, Vast)

`cluster/warp_num_envs_sweep.sbatch` run on a rented 3090 (`vast.sh
warp-sweep`; the cluster was full), Warp with the full solver, 40M steps per
run, `num_minibatches = num_envs / 32` so every training step still uses one
unroll per env; log `notes/warp_probe_logs/vast_warp_envs_sweep.log`:

| envs | single ant (steps/s) | vs 1024 | co-design (steps/s) | vs 1024 |
|---|---|---|---|---|
| 1024 | 124.1k | 1.00 | 134.9k | 1.00 |
| 2048 | 183.5k | 1.48 | 191.8k | 1.42 |
| 4096 | 219.6k | 1.77 | 233.4k | 1.73 |
| 8192 | 243.9k | 1.97 | 262.5k | 1.95 |

* At 1024 envs this host matches mscluster (co-design 134.9k vs 133.3k).
* Diminishing returns: 4096 -> 8192 adds only ~11%.
* nvidia-smi reported 100% GPU utilisation at every size. That metric is the
  share of time some kernel is running, not how much of the GPU it uses: the
  GPU was never idle at 1024 envs, but its kernels were too small to fill it.
  This corrects the earlier guess of host-side overhead. The memory column
  (~14.9 GB) is JAX's preallocation (0.6 x 24 GB), not what the runs needed.
* No overflow warnings at any size (contacts sized 64 per env).
* **Co-design caveat.** A design is scored only from episodes that finish
  within its iteration: episodes per lane = steps per iteration / num_envs /
  2500. The 2B schedule's 3.85M-step iterations give 1.50 at 1024 envs but
  0.75 at 2048 and 0.38 at 4096, so more envs require proportionally longer
  design iterations (fewer design updates) or more total steps. The 8192 run
  here already printed the trainer's < 1.0 warning and one unscored design.
* Throughput only: a bigger env count changes how many envs each PPO update's
  data comes from, so the learning outcome needs its own check before it
  becomes a default.

### 4. Vases

Not used by minefield, but the `run` task has them.

* The current vase cost (`collision.geoms_colliding`) reads `data.contact`,
  which Warp does not expose. It **fails loudly** on Warp
  (`'Data' object has no attribute 'contact'`), it does not silently return 0.
* MuJoCo's **contact sensors** are the documented replacement. A test scene (a
  capsule falling onto a free box, sensor `data="found"`) gives the same reading
  at every step on JAX, Warp and CPU MuJoCo
  (`scripts/warp_probe/vases_and_contact_sensor.py`). Porting the vase cost
  means one sensor per vase (robot subtree vs vase) and reading `sensordata`;
  the same code then runs on both backends.
* Throughput with vases is not measured. Vases were removed from minefield
  partly for speed (2026-08-15); since Warp only pays for active contacts they
  may be much cheaper there, but that needs measuring.

### 5. Full-length quality run

The old final baseline (`ant_minefield_final_b25_300M`, Vast 3090, MJX-JAX 3.3.2,
solver 4/8, stopped at 261M) repeated with identical flags and seed but on Warp
with the full solver, on the laptop: 5 cm gate, budget 25, multiplier lr
1.5e-5 / cap 3 / init 0.01, grid 7, torso lidar, linear cost, flip and exit cost
50, no walls, finish line, policy 256x4, 1024 envs / 32 minibatches, 300M steps,
31 evaluations of 320 episodes. Log `notes/warp_probe_logs/warp_ant_minefield_final_b25_300M.log`
(filtered copy), checkpoints `checkpoints/warp_ant_minefield_final_b25_300M`
(laptop only). Both rows below average the three evaluations at 241, 251 and
261M, the last three the old run reached:

| run | return | speed (m/s) | cost | hazard_steps | flipped | out of bounds | multiplier |
|---|---|---|---|---|---|---|---|
| old: MJX-JAX, solver 4/8, RTX 3090 | 10.89 | 1.447 | 27.4 | 70.8 | 2.6% | 1.9% | 0.47 |
| new: MJX-Warp, solver 100/50, RTX 4050 | 11.06 | 1.520 | 26.3 | 76.6 | 0.3% | 2.4% | 0.30 |
| new, final three evaluations (281-301M) | 11.08 | 1.532 | 26.2 | 77.8 | 0.5% | 0.6% | 0.33 |

* **Learning is not degraded.** At equal steps the new run is ~5% faster, sits
  at its budget of 25, and flips less. With one seed per arm these differences
  are within what seeds alone can produce, so the claim is "matches", not
  "improves".
* **Wall time.** 300M steps took 2,747 s of training (46 min; median 112.7k
  steps/s) on the laptop, against 6,102 s for 261M (1.7 h; median 43.2k) for the
  old run on a 3090.
* The run printed no solver or contact overflow warnings.

## Code changes during the probe (branch `warp-probe`)

| file | change | why |
|---|---|---|
| `mjx_safety_gym/backend.py` (new) | `put_model` / `make_data` wrappers; backend chosen by `MJX_IMPL` (unset = JAX); for Warp, `make_data` gets the host `MjModel` and buffer sizes (`MJX_NACONMAX`, default 64 x `MJX_NWORLD`; `MJX_NJMAX`, default 256) | Warp's `make_data` only accepts a `mujoco.MjModel` and needs its contact (all worlds) and constraint (per world) capacities up front |
| `envs/go_to_goal.py`, `envs/run_forward.py`, `morphology.py` | call the wrappers instead of `mjx.put_model` / `mjx.make_data` | as above |
| `algorithms/ppo/train.py` | env reset and design reset via `jax.pmap` instead of `jax.jit(jax.vmap(...))` | Warp keeps its contact buffers shared across worlds (one array, no per-env axis). Only `pmap` gives those a device axis, which the pmapped training epoch then removes; brax 0.14 resets the same way |
| `algorithms/ppo/train.py` | checkpoint config gets an empty `network_factory_kwargs` | brax 0.14's checkpoint writer requires the key |
| `algorithms/rl/utils.py` | old checkpoints' observation normaliser converted by field name | brax 0.14 stores the sample count as a 64-bit integer split into two 32-bit halves and adds `std_eps`; old checkpoints no longer line up leaf for leaf |
| `morphology.py` | `_pin` helper: pinned static fields go where MJX 3.14 keeps them (`geom_rbound_hfield` now lives in `Model._impl`; Warp has none) | co-design batching of per-env bodies |

Not changed: the environments, rewards, costs, observation, PPO and the
Lagrangian. `data.contact` is not available on Warp; minefield does not use it.
JAX and Warp both reserve GPU memory in one process, so runs set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.

## Adoption (2026-10-04)

What changed on `main` when the probe was merged, beyond the probe's own
changes above.

**Defaults.**
* `train_ppo` trains on MuJoCo Warp unless told otherwise: `--impl warp`
  (default) or `--impl jax`. Warp's buffers are sized from the run:
  `--warp_contacts_per_env` (64; the shared buffer holds this x
  max(num_envs, num_eval_envs)) and `--warp_njmax` (256 rows per env).
  `--impl warp` without a GPU stops with an error instead of falling back.
* Everything else (main.py, replay, render and eval scripts) uses MJX-JAX by
  default, so it runs on CPU too; `main.py --impl` overrides. With the solver
  converged the two backends are the same physics (sections 1-2).
* The ant has **no solver cap**: MuJoCo's own 100 / 50 applies. The 4 / 8 cap
  of 2026-09-27..10-03 was an MJX-JAX speed fix; on `--impl jax` it is still
  the fast option and can be passed explicitly.
* `mjx_safety_gym/backend.py` is the single place models and data are made
  (`configure()`, `put_model()`, `make_data()`); the trainer configures it
  before any env exists.

**Vase cost on contact sensors** (`world.add_vase_contact_sensors`). Warp has
no `data.contact`, so the vase collision cost now reads one contact sensor per
(vase, robot geom) pair, data `found` and `dist` with `reduce="mindist"`; a
vase is hit when any of its pairs is found with distance < 0, the old rule.
Two things learned on the way:
* MJX-JAX does not implement subtree matching for contact sensors, hence one
  sensor per robot geom rather than one per vase.
* `found` alone counts contacts at distance >= 0 as well and over-counted vase
  hits by ~9%; requiring distance < 0 makes it exact.

Check (`scripts/warp_probe/vase_cost_equivalence.py`, run task, 64 envs, vases
moved into a ring around each ant, 300 random-action steps, JAX backend where
both costs exist): **0 differing env-steps** in 19,200, with 3,440-3,491
contact steps per run (GPU runs are not bit-reproducible, so totals vary
between runs; old and new agree within each run). Warp vs JAX from the
identical start state: the same 320 of 8,320 pairs touching, 0 disagreements.
Caveat: on deep box-capsule overlaps the two backends' contact distances differ
by up to 3.7 mm, so vase-heavy dynamics are not bit-identical across backends.

**Packages and envs.**
* `pyproject.toml` pins the new stack (MuJoCo / MJX 3.14.0 + `[warp]`,
  mujoco-warp 3.14.0, warp-lang 1.17.0, brax 0.14.2, playground 0.2.0, flax
  0.12.8, orbax-checkpoint 0.12.6), replacing the Andrew-Luo1 brax / playground
  forks; version 0.2.0; `[render]` extra for `scripts/export_usd.py`.
* Env `mjx-warp` (exact freeze: `cluster/warp/requirements-warp.txt`, built by
  `cluster/warp/setup_env.sh`) runs `main`. Env `mjx-safety-gym` stays the old
  stack and runs a checkout of the **`pre-warp` tag** (the last commit before
  the merge). Vast builds its `morpho` env from `pyproject.toml`, so it follows
  automatically; its provisioning now also checks Warp can drive the GPU.

**Launchers.** All sbatch/laptop scripts activate `mjx-warp`. The co-design
launchers (cluster and Vast) pass `--impl warp`, assert the full 100 / 50 solver
and that Warp sees the GPU, give JAX 0.6 of the GPU memory (Warp allocates its
own buffers in the same process; with JAX at 0.9 on the 6 GB laptop it is
tight), and allow 12 h instead of 36 h. The single-ant launcher has
`SAFE_IMPL` (default warp). Wrappers for pre-warp runs say so in their header;
the final-baseline wrapper pins `SAFE_IMPL=jax` with its old 4 / 8 cap, the
closest match on the new stack. `cluster/warp_sps_check.sbatch` now compares a
`pre-warp` checkout against `main`; `cluster/warp_num_envs_sweep.sbatch` is new.

**End-to-end checks of the merged code** (laptop RTX 4050, default flags
unless stated, short runs): single ant 3M steps -> backend warp, 66k steps/s;
co-design 6M (1024 envs, 16 bodies) -> warp, 102k, design iterations run;
single ant at 2048 envs / 64 minibatches (the sweep's scaling) -> 95k; all
with zero solver-limit warnings. The co-design launcher's preflight passes;
`scripts/export_usd.py` (CPU, JAX backend) replays the 300M Warp checkpoint.
An OLD checkpoint (budget-0 2 cm single ant, trained on MJX-JAX 3.3.2 with the
4 / 8 cap) restores and runs on Warp with the full solver; 512 episodes:

| setup | return | speed (m/s) | cost | hazard_steps | flipped | out of bounds |
|---|---|---|---|---|---|---|
| its own stack (3.3.2, JAX, 4/8), section 1 | 11.05 | 1.195 | 1.49 | 3.56 | 0.8% | 1.0% |
| new default (3.14, Warp, 100/50) | 11.04 | 1.135 | 1.96 | 5.48 | 0.4% | 1.6% |

As in section 1, a policy tuned to the capped solve loses ~5% speed on the
converged one; new results should come from policies trained on the new setup.

**Reproducing a pre-Warp result.** `git worktree add ../MorphoSafety-prewarp
pre-warp`, install it into `mjx-safety-gym` (`pip install --no-deps -e`), and
run from there: same code, same MuJoCo 3.3.2, same solver cap as at the time.

## Threats to validity

* **Single seed, short runs.** Throughput was measured over 6M steps, one seed.
  Speed rises 8-28% as a policy learns (fewer flailing contacts), so short runs
  understate absolute numbers. All configurations were measured the same way,
  so the ratios are fair. The learning outcomes of these 6M-step runs are not
  evidence either way.
* **One GPU.** Measured on a laptop RTX 4050. The 3090 ratio is measured by
  `cluster/warp_sps_check.sbatch`, not assumed.
* **Solver change.** The new setup changes two things at once relative to old
  results: the backend and the converged solver. The physics checks above
  separate them (the models agree when converged); the 300M run tests the
  combination end to end.
* **Old results stay valid on their own terms.** They were produced and
  compared within the old setup. Mixing old and new numbers in one table is not
  valid without re-running the old references.

## Suggested methods text

> Simulation uses MuJoCo 3.14 through MJX with the MuJoCo Warp backend on
> NVIDIA GPUs, timestep 0.01 s with the RK4 integrator. Contact is solved with
> the Newton solver at MuJoCo's default limits (100 iterations, 50 line-search
> steps, tolerance 1e-8); the batched solver exits once all environments have
> converged, so contact forces are converged rather than truncated. We verified
> that this backend reproduces the JAX implementation of MJX to within float32
> rounding, including when every parallel environment simulates a different
> morphology.

## Reproduction

All probe scripts are in `scripts/warp_probe/`; run them from the warp-probe
checkout with `~/miniconda3/envs/mjx-warp/bin/python`, choosing the backend
with `MJX_IMPL=jax|warp` (and `MJX_NWORLD` for buffer sizing). The 3.3.2 side
of the physics comparisons runs the same scripts from the main checkout with
the `mjx-safety-gym` env.

| measurement | script | log |
|---|---|---|
| one-step / rollout physics | `physics_one_step.py`, `physics_rollout.py` | (printed) |
| per-env bodies, Warp vs JAX | `per_lane_bodies.py` | (printed) |
| pure stepping throughput | `step_throughput.py <iterations> <ls_iterations>` | (printed) |
| buffer use | `buffer_usage.py` | (printed) |
| vases / contact sensors | `vases_and_contact_sensor.py` | (printed) |
| vase cost, sensors vs old contact read | `vase_cost_equivalence.py` | (printed) |
| state leaves without a device axis | `state_leaves.py` | (printed) |
| policy-level physics check | trainer, `--restore_checkpoint_path ... --num_timesteps 1 --num_evals 2 --num_eval_envs 64 --num_eval_episodes 8` | `notes/warp_probe_logs/evalA_old.log`, `evalB_new_jax.log` |
| training throughput A-H | trainer, flags in the logs' first lines | `notes/warp_probe_logs/bench_*.log` |

The saved logs are filtered copies: each starts with a line saying how many
lines of Warp noise (kernel-load messages, per-occurrence overflow warnings)
were removed. The cluster env is built with `cluster/warp/setup_env.sh` from
`cluster/warp/requirements-warp.txt`.

"""NSGA-II search over ant morphology, against a frozen conditioned policy.

DRAFT / UNVERIFIED: written while the ant training-throughput measurement
occupied the only GPU (see the project plan's "at most one GPU job at a
time" safeguard), so this has not been run yet. Verify against Verification
items 4-5 in the plan before trusting any output -- in particular, confirm
per-morphology returns actually differ (item 4) before running a real search,
or the fitness landscape here is measuring an untrained/undifferentiated
policy, not morphology.

Usage:
    python -m scripts.evolve --checkpoint checkpoints/ant --generations 20 \
        --population 32 --num_seeds 8

Requires a policy checkpoint trained WITH morphology conditioning
(--num_morphologies > 0 in train_ppo.py), since the network's observation
width must match base_obs_size + morphology.NUM_GENES.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.envs.go_to_goal import GoToGoal
from mjx_safety_gym.morphology import (
    GENE_NAMES,
    MASS_BAND,
    NUM_GENES,
    MorphologySpec,
    _BATCHED_FIELDS,
    batch_models,
    build_mj_model,
    mass_in_band,
    total_mass,
)

# Arena layouts are held fixed across every candidate and every generation
# (common random numbers) -- per-episode cost/reward variance is large (the
# project notes record episode cost ranging 0-961 for the point across
# layouts), so without this a GA chases layout noise, not morphology.
_DEFAULT_SEEDS = tuple(range(8))


def load_policy_fn(checkpoint: str):
    """Load a morphology-conditioned policy, matching main.py's convention.

    Rebuilds the network from known shapes (GoToGoal's own
    observation_size, which already includes NUM_GENES when
    morphology_conditioning=True) rather than a saved config, same reasoning
    as main.py.load_policy: the training code writes an empty ConfigDict, so
    nothing else can reconstruct the network.
    """
    ckpt = Path(checkpoint)
    if ckpt.is_dir():
        steps = sorted(p for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit())
        if steps:
            ckpt = steps[-1]
    ckpt = ckpt.resolve()
    loaded = ocp.PyTreeCheckpointer().restore(str(ckpt))
    normalizer = running_statistics.RunningStatisticsState(**loaded[0])
    policy_params, value_params = loaded[1]["policy"], loaded[1]["value"]

    base_env = GoToGoal(robot="ant", morphology_conditioning=True)
    network = ppo_networks.make_ppo_networks(
        base_env.observation_size,
        base_env.action_size,
        preprocess_observations_fn=lambda x, _: x,
    )
    policy = ppo_networks.make_inference_fn(network)(
        (normalizer, policy_params, value_params), deterministic=True
    )
    return jax.jit(lambda o, k: policy(o, k)[0]), base_env


def make_rollout_fn(
    env: GoToGoal, policy_fn, episode_length: int, num_seeds: int, seeds=None
):
    """Batched (K morphologies x S fixed seeds) rollout, K*S lanes total.

    `seeds` defaults to `_DEFAULT_SEEDS[:num_seeds]` (evolve.py's own common-
    random-numbers set, fixed across generations). fitness_noise.py overrides
    it with different seed lists at the same count to measure how much a
    num_seeds-sized estimate moves depending on WHICH seeds land in it --
    that's the thing worth calibrating; reusing the same fixed list would
    just measure zero variance against itself.

    Manually swaps `env._mjx_model`/`env._morphology_genes` together per vmap
    lane -- the same pattern `MorphologyDomainRandomizationWrapper`
    (mjx_safety_gym.algorithms.wrappers) uses for training, kept inline here
    rather than reusing that wrapper since evaluation doesn't need the full
    training wrapper stack (no episode/autoreset wrapper -- episodes just run
    for `episode_length` steps and get summed directly). Genes must be set
    the same way `_mjx_model` is (an attribute swap under vmap), NOT appended
    to `obs` after the fact -- appending externally is what caused a real
    shape-mismatch crash during training; see that wrapper's docstring.
    `env` must have been constructed with `morphology_conditioning=True` (see
    `load_policy_fn`) so `get_obs` appends `_morphology_genes` itself.
    """
    if seeds is None:
        seeds = _DEFAULT_SEEDS[:num_seeds]
    seed_keys = jp.stack([jax.random.PRNGKey(s) for s in seeds])

    def run_one(model, rng, genes):
        def reset(rng):
            env.unwrapped._mjx_model = model
            env.unwrapped._morphology_genes = genes
            return env.reset(rng)

        state = reset(rng)

        def step(carry, _):
            state, rng = carry
            env.unwrapped._mjx_model = model
            env.unwrapped._morphology_genes = genes
            rng, key = jax.random.split(rng)
            action = policy_fn(state.obs, key)
            nstate = env.step(state, action)
            return (nstate, rng), (
                nstate.reward,
                nstate.info["cost"],
                nstate.info["goal_reached"],
            )

        _, (rewards, costs, goals) = jax.lax.scan(
            step, (state, rng), (), episode_length
        )
        return jp.sum(rewards), jp.sum(costs), jp.sum(goals)

    def rollout(specs: list[MorphologySpec]):
        K = len(specs)
        S = num_seeds
        mj_models = [build_mj_model(s) for s in specs]
        masses = np.array([total_mass(m) for m in mj_models])
        batched, in_axes = batch_models(mj_models)
        batched = batched.tree_replace(
            {f: jp.repeat(getattr(batched, f), S, axis=0) for f in _BATCHED_FIELDS}
        )
        rngs = jp.tile(seed_keys, (K, 1))
        genes = jp.repeat(
            jp.asarray(np.stack([s.genes for s in specs]), dtype=jp.float32), S, axis=0
        )
        run_batched = jax.vmap(run_one, in_axes=(in_axes, 0, 0))
        rewards, costs, goals = run_batched(batched, rngs, genes)
        rewards = np.asarray(rewards).reshape(K, S)
        costs = np.asarray(costs).reshape(K, S)
        goals = np.asarray(goals).reshape(K, S)
        return rewards.mean(axis=1), costs.mean(axis=1), goals.mean(axis=1), masses

    return rollout


class MorphologyProblem(Problem):
    """2-objective: minimize -mean_return and mean_episode_cost.

    Mass is logged, not optimized (see the plan's "size confound" decision) --
    infeasible-mass candidates get penalized (pushed off the front) rather
    than rejected outright, so the GA can still learn the mass boundary
    instead of hitting a hard wall.
    """

    def __init__(self, rollout_fn, log_path: Path):
        super().__init__(n_var=NUM_GENES, n_obj=2, xl=0.0, xu=1.0)
        self.rollout_fn = rollout_fn
        self.log_path = log_path
        self.generation = 0

    def _evaluate(self, X, out, *args, **kwargs):
        specs = [MorphologySpec(genes=row) for row in X]
        returns, costs, goals, masses = self.rollout_fn(specs)
        infeasible = ~np.array([mass_in_band(build_mj_model(s)) for s in specs])
        penalty = np.where(infeasible, 1e3, 0.0)
        out["F"] = np.column_stack([-returns + penalty, costs + penalty])

        self.generation += 1
        with open(self.log_path, "a") as f:
            for i in range(len(specs)):
                row = {
                    "generation": self.generation,
                    "return": float(returns[i]),
                    "cost": float(costs[i]),
                    "goals": float(goals[i]),
                    "mass": float(masses[i]),
                    "infeasible_mass": bool(infeasible[i]),
                    **{name: float(v) for name, v in zip(GENE_NAMES, X[i])},
                }
                f.write(str(row) + "\n")
        print(
            f"gen {self.generation}: return {returns.mean():.2f}"
            f" (best {returns.max():.2f}), cost {costs.mean():.2f}"
            f" (best {costs.min():.2f}), goals {goals.mean():.2f},"
            f" mass {masses.mean():.1f} [{MASS_BAND[0]}-{MASS_BAND[1]}],"
            f" infeasible {infeasible.sum()}/{len(specs)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ant")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default="scratchpad/evolve_log.jsonl")
    args = parser.parse_args()

    policy_fn, env = load_policy_fn(args.checkpoint)
    rollout_fn = make_rollout_fn(env, policy_fn, args.episode_length, args.num_seeds)
    problem = MorphologyProblem(rollout_fn, Path(args.log))

    algorithm = NSGA2(
        pop_size=args.population,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(
        problem, algorithm, ("n_gen", args.generations), seed=args.seed, verbose=False
    )
    print("Pareto front (return, cost):")
    for f in result.F:
        print(f"  return={-f[0]:.2f} cost={f[1]:.2f}")


if __name__ == "__main__":
    main()

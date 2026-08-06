"""Fitness-noise calibration: how many arena seeds does evolve.py need?

DRAFT / UNVERIFIED, written alongside evolve.py while the GPU was occupied by
the throughput measurement -- see that file's header.

Evaluates one fixed morphology (nominal) repeatedly at increasing --num_seeds
and reports the standard error of the mean return/cost. Per the project plan,
episode cost for the point ranged 0-961 across layouts -- arena variance is
large, and evolve.py's GA will chase it unless enough seeds are averaged that
this standard error is small relative to the spread you expect between
distinct morphologies. Run this before trusting any NSGA-II output; if the
standard error is still large at a --num_seeds you can afford (compute scales
linearly with it), that's a sign the population size needs to shrink to buy
more seeds per candidate, not the other way around.

Usage:
    python -m scripts.fitness_noise --checkpoint checkpoints/ant \
        --seed_counts 4 8 16 32 --repeats 5
"""

import argparse

import numpy as np

from mjx_safety_gym.morphology import MorphologySpec
from scripts.evolve import load_policy_fn, make_rollout_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ant")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--seed_counts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent DIFFERENT seed sets evaluated per seed count -- "
        "the spread across these is the standard error that matters: it "
        "answers 'how much would my estimate have moved if evolve.py's fixed "
        "seed list had come out differently', not the (structurally zero) "
        "variance of re-evaluating the exact same fixed list twice.",
    )
    args = parser.parse_args()

    spec = MorphologySpec.nominal()
    print(f"{'seeds':>6}  {'return mean':>12}  {'return sem':>11}  "
          f"{'cost mean':>10}  {'cost sem':>9}")
    for num_seeds in args.seed_counts:
        policy_fn, env = load_policy_fn(args.checkpoint)
        returns, costs = [], []
        for rep in range(args.repeats):
            seeds = range(rep * num_seeds, (rep + 1) * num_seeds)
            rollout_fn = make_rollout_fn(
                env, policy_fn, args.episode_length, num_seeds, seeds=seeds
            )
            r, c, _, _ = rollout_fn([spec])
            returns.append(r[0])
            costs.append(c[0])
        returns, costs = np.array(returns), np.array(costs)
        r_sem = returns.std(ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else float("nan")
        c_sem = costs.std(ddof=1) / np.sqrt(len(costs)) if len(costs) > 1 else float("nan")
        print(f"{num_seeds:>6}  {returns.mean():>12.3f}  {r_sem:>11.3f}  "
              f"{costs.mean():>10.3f}  {c_sem:>9.3f}")


if __name__ == "__main__":
    main()

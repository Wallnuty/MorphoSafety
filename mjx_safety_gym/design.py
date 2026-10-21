"""A learnable distribution over morphologies, per Schaff et al. (ICRA 2019).

`Jointly Learning to Construct and Control Agents using Deep Reinforcement
Learning`, Schaff, Yunis, Chakrabarti & Walter -- arXiv 1801.01432. Reference
implementation at github.com/cbschaff/nlimb (MIT), mirrored locally at
~/MorphologyResearch/nlimb. That code is TensorFlow 1.11 on Roboschool and no
longer runs; this is a port of the method, not of the code.

WHAT THIS ADDS THAT THE REPO DID NOT HAVE. `morphology.randomization_fn`
samples bodies from a FIXED uniform distribution, once, at wrapper-construction
time. The policy is conditioned on the genes, which is half of Schaff's method;
the other half is that the DISTRIBUTION ITSELF is trained toward
higher-performing designs. This module is that half.

Why it matters here specifically: measured 2026-08-18 on the 30M conditioned
checkpoint, a decoupled pipeline (train on 8 fixed bodies, search elsewhere)
left 51% of candidate bodies outside the trained mass range and 29% of unseen
bodies unreliable. Training the distribution couples the two -- the policy is
always trained on whatever is currently being sampled, so there is no
train/test mismatch to manage.

THE UPDATE IS REINFORCE, and in the reference it is four lines
(`algorithm.py:RobotDistributionLoss`):

    loss = mean( neglogp(design) * episode_reward )

with `episode_reward` standardised across the designs sampled that iteration as
the baseline. Because the mixture weights are frozen (below), the gradient is
the plain diagonal-Gaussian score function of whichever component produced the
sample -- analytic, so no autodiff and no JAX is needed here. Everything in
this module is host-side numpy; it runs between training epochs, never under
jit.

DELIBERATE FIDELITY CHOICES, each mirroring the reference:

  * 8 components, means init U(-0.8, 0.8), std init 0.577
    (`model.py:RobotSampler`).
  * MIXTURE WEIGHTS ARE NEVER TRAINED. `mixprobs` is created with
    `trainable=False` upstream, so the mixture stays uniform over surviving
    components for the whole run. The distribution commits to one design by
    CHOPPING (below), not by learning weights. Get this wrong and the method
    silently becomes something else.
  * Gradients flow only through the SAMPLED component (their `GmmPd.sample`
    override returns the component index alongside the sample).
  * The log-prob is evaluated at the UNCLIPPED sample. Physics uses the value
    clipped into the design box, but `envs.py` keeps `unclipped_params` for the
    gradient -- otherwise a proposal outside the box gets zero gradient and the
    distribution can never be pulled back inward.
  * EACH DESIGN'S GRADIENT IS CLIPPED TO GLOBAL NORM 0.5 BEFORE AVERAGING
    (2026-10-19). Upstream gives every MPI worker one robot, takes
    `FlatGrad(clip_norm=args.clip_norm)` of that robot's loss
    (`algorithm.py:57`; `--grad_clip_norm 0.5`, `init.py:85`, `train.py:25`)
    and MpiAdam averages the clipped gradients over workers
    (`mpi_adam.py:30-32`). One design here is one worker.
  * Adam is MpiAdam's own arithmetic (`mpi_adam.py:34-38`): eps 1e-5 is added
    to the UNcorrected sqrt(v).
  * A chop scores every live component afresh: 100 designs drawn from it, one
    episode each under the current stochastic policy (`component_chopper.py`),
    and it resets BOTH optimizers -- the design's here, the policy's in
    `ppo/train.py` (`algorithm.py:105-109`).

Every co-design run before 2026-10-19 lacked the clip, the policy-optimizer
reset and the fresh chop rollouts, and scored each lane by its last episode
only; see `DesignLoop` for the flags that reproduce them.

DEVIATION FROM THE PAPER, the one this project exists for: the inner loop is
PPO-Lagrangian under a cost budget, not plain PPO, and each GMM component
keeps its OWN Lagrange multiplier (2026-09-28; `--lagrange_per_component`).
The lane -> component map comes from `sample()` and rides in
info["design_component"]; see ppo/losses.py. With one shared multiplier the
budget would only hold on average over the population, and the speed score
(which does not read cost) would favour a reckless body carried by safe ones.

DEVIATION FROM THE PAPER, recorded on purpose: designs live in [-1, 1] as they
do upstream, but map onto this repo's existing `MorphologySpec` genes
(`(p + 1) / 2`), whose scale range is SCALE_LO/HI = 0.6-1.4 rather than
Schaff's 0.5-1.5. Keeping our range means every number measured before today
stays comparable; the cost is that the design box is slightly narrower.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

# Reference defaults, `model.py:RobotSampler.__init__` and
# `algorithm.py:Algorithm.defaults`.
N_COMPONENTS = 8
MEAN_INIT_RANGE = 0.8  # means ~ U(-0.8, 0.8)
STD_INIT = 0.577
DESIGN_LO, DESIGN_HI = -1.0, 1.0

_DEAD = -1e6  # what upstream writes into logmixprobs to retire a component

_U64 = (1 << 64) - 1


def _u128_to_words(x: int) -> np.ndarray:
    """Split a 128-bit int into two uint64 words, low word first.

    numpy's PCG64 keeps its counter and increment as 128-bit PYTHON ints, which
    no array format stores natively -- hence the split. Round-tripped in
    `verify_design_checkpoint.py` by drawing from a restored generator and
    comparing against the original, which is the only check that actually
    proves the encoding.
    """
    x = int(x)
    return np.array([x & _U64, (x >> 64) & _U64], dtype=np.uint64)


def _words_to_u128(words) -> int:
    w = np.asarray(words, dtype=np.uint64)
    return int(w[0]) | (int(w[1]) << 64)


@dataclasses.dataclass
class Adam:
    """`MpiAdam(epsilon=1e-5, beta1=robot_momentum)`, arithmetic included.

    Separate from the policy optimizer on purpose: upstream keeps a distinct
    `mpi_adam_robot` and RESETS it after every chop, because the surviving
    components face a different objective landscape once their neighbours are
    gone and stale moments would drag them.

    Two instances (means, log-stds) behave exactly like upstream's single one
    over the flat vector: Adam is elementwise and both step and reset together.
    """

    shape: tuple[int, ...]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-5
    m: np.ndarray = dataclasses.field(init=False)
    v: np.ndarray = dataclasses.field(init=False)
    t: int = dataclasses.field(default=0, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.t = 0

    def step(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """Returns the update to SUBTRACT from the parameter.

        MpiAdam.update line for line (`mpi_adam.py:34-38`): the bias correction
        is folded into the step size and eps goes on the UNcorrected sqrt(v).
        The textbook form, mhat / (sqrt(vhat) + eps), which this used before
        2026-10-19, is the same with eps / sqrt(1 - beta2^t) in place of eps --
        3.2e-4 at the first step after a reset, 1e-5 in the limit.
        """
        self.t += 1
        a = lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        return a * self.m / (np.sqrt(self.v) + self.eps)


class GmmDesignDistribution:
    """Gaussian mixture over normalized design parameters in [-1, 1]^D.

    Uniform, FROZEN mixture weights over the surviving components; only the
    component means and log-stds are trained. See the module docstring for why
    that is not an oversight.
    """

    def __init__(
        self,
        n_params: int,
        n_components: int = N_COMPONENTS,
        std_init: float = STD_INIT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        beta2: float = 0.999,
        seed: int = 0,
        mean_init: np.ndarray | None = None,
        clip_norm: float | None = 0.5,
    ) -> None:
        self.n_params = int(n_params)
        self.n_components = int(n_components)
        self.rng = np.random.default_rng(seed)
        self.lr = float(lr)
        # Per-design gradient clip (upstream --grad_clip_norm 0.5); None or 0
        # turns it off, as every co-design run before 2026-10-19 had it.
        self.clip_norm = float(clip_norm) if clip_norm else None
        self.last_clipped = 0.0  # share of the last update's samples clipped

        if mean_init is not None:
            means = np.tile(np.asarray(mean_init, dtype=float), (self.n_components, 1))
        else:
            means = self.rng.uniform(
                -MEAN_INIT_RANGE, MEAN_INIT_RANGE, size=(self.n_components, self.n_params)
            )
        self.means = means
        self.log_stds = np.full((self.n_components, self.n_params), np.log(std_init))
        # Mirrors upstream's `logmixprobs`: 0.0 for a live component, -1e6 for a
        # chopped one. Never a trained variable.
        self.log_mixprobs = np.zeros(self.n_components)

        self._adam_mean = Adam(self.means.shape, beta1=momentum, beta2=beta2)
        self._adam_logstd = Adam(self.log_stds.shape, beta1=momentum, beta2=beta2)

    # -- component bookkeeping --------------------------------------------

    @property
    def alive(self) -> np.ndarray:
        return self.log_mixprobs == 0.0

    def components_left(self) -> int:
        return int(np.sum(self.alive))

    def _log_mixing_prob(self) -> float:
        """Uniform over survivors, so this is a constant offset in the loss.

        It has zero gradient with respect to every trainable parameter, but it
        is included because upstream's `neglogp` includes it and the reported
        loss value should match.
        """
        return -np.log(self.components_left())

    # -- sampling ----------------------------------------------------------

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw `n` designs. Returns (UNCLIPPED params (n, D), component (n,)).

        Unclipped by design -- see the module docstring. Clip at the point of
        use, not here.
        """
        live = np.flatnonzero(self.alive)
        comps = self.rng.choice(live, size=n)
        eps = self.rng.normal(size=(n, self.n_params))
        params = self.means[comps] + np.exp(self.log_stds[comps]) * eps
        return params, comps

    def sample_component(self, index: int, n: int) -> np.ndarray:
        """Draw `n` designs from ONE component. Used to score it before a chop."""
        eps = self.rng.normal(size=(n, self.n_params))
        return self.means[index] + np.exp(self.log_stds[index]) * eps

    def mode_component(self) -> int:
        """Index of the component whose mean is the mode (see `mode`)."""
        live = np.flatnonzero(self.alive)
        scores = [self.logp(self.means[i][None], np.array([i]))[0] for i in live]
        return int(live[int(np.argmax(scores))])

    def mode(self) -> np.ndarray:
        """The highest-density design, matching `GmmPd.mode`.

        Upstream compares `logp(mode_i) + log_mix_i` across components and takes
        the argmax. With uniform weights that reduces to the component with the
        smallest total log-std, i.e. the tightest one -- but it is written out
        so the behaviour still holds if weights ever become trainable.
        """
        return self.means[self.mode_component()].copy()

    # -- densities ---------------------------------------------------------

    def logp(self, params: np.ndarray, comps: np.ndarray) -> np.ndarray:
        """log p(design, component) for the component that produced each sample."""
        mu = self.means[comps]
        log_sd = self.log_stds[comps]
        z = (params - mu) / np.exp(log_sd)
        gauss = -0.5 * np.sum(z**2, axis=-1) - np.sum(log_sd, axis=-1) - 0.5 * (
            self.n_params * np.log(2.0 * np.pi)
        )
        return gauss + self._log_mixing_prob()

    def neglogp(self, params: np.ndarray, comps: np.ndarray) -> np.ndarray:
        return -self.logp(params, comps)

    # -- the update --------------------------------------------------------

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Upstream's `_norm_rewards`: standardise across this iteration's designs.

        This is the REINFORCE baseline. With no baseline every design gets
        pushed in the same direction whenever returns are all positive, which on
        this task they always are.
        """
        scores = np.asarray(scores, dtype=float)
        return (scores - scores.mean()) / (scores.std() + 1e-8)

    def grads(
        self, params: np.ndarray, comps: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analytic gradient of `mean(neglogp * score)` w.r.t. means and log_stds.

        For a diagonal Gaussian with z = (x - mu) / sigma:

            d(neglogp)/d(mu)      = -(x - mu) / sigma^2
            d(neglogp)/d(log sd)  = 1 - z^2

        accumulated ONLY into the component that drew each sample, and divided
        by the total sample count because the loss is a mean over samples (not
        per component).

        With `clip_norm`, each sample's gradient is first scaled as
        tf.clip_by_global_norm does, by clip_norm / max(norm, clip_norm), the
        norm taken over everything that sample touches (its component's means
        and log-stds; every other entry of upstream's flat gradient is zero).
        Upstream's per-worker loss is over a batch of ONE robot, so this is its
        FlatGrad followed by MpiAdam's average over workers.
        """
        params = np.asarray(params, dtype=float)
        scores = np.asarray(scores, dtype=float)
        n = len(comps)
        mu = self.means[comps]
        sd = np.exp(self.log_stds[comps])
        z = (params - mu) / sd

        d_mu = -(params - mu) / sd**2 * scores[:, None]
        d_logsd = (1.0 - z**2) * scores[:, None]
        if self.clip_norm:
            norm = np.sqrt(np.sum(d_mu**2, axis=1) + np.sum(d_logsd**2, axis=1))
            scale = self.clip_norm / np.maximum(norm, self.clip_norm)
            d_mu = d_mu * scale[:, None]
            d_logsd = d_logsd * scale[:, None]
            self.last_clipped = float(np.mean(norm > self.clip_norm))

        g_mu = np.zeros_like(self.means)
        g_logsd = np.zeros_like(self.log_stds)
        np.add.at(g_mu, comps, d_mu)
        np.add.at(g_logsd, comps, d_logsd)
        return g_mu / n, g_logsd / n

    def update(
        self,
        params: np.ndarray,
        comps: np.ndarray,
        scores: np.ndarray,
        lr_frac: float = 1.0,
    ) -> dict[str, float]:
        """One REINFORCE step. `lr_frac` is upstream's linear `1 - t/tmax` decay."""
        norm = self.normalize_scores(scores)
        g_mu, g_logsd = self.grads(params, comps, norm)
        lr = self.lr * lr_frac
        self.means -= self._adam_mean.step(g_mu, lr)
        self.log_stds -= self._adam_logstd.step(g_logsd, lr)
        # Dead components must not drift: they are still indexed by `means`, and
        # a resurrected-looking mean would be misleading in logs.
        dead = ~self.alive
        out = {
            "design/loss": float(np.mean(self.neglogp(params, comps) * norm)),
            "design/grad_norm": float(np.linalg.norm(g_mu)),
            "design/components": float(self.components_left()),
            "design/mean_std": float(np.exp(self.log_stds[self.alive]).mean()),
            "design/dead": float(dead.sum()),
        }
        if self.clip_norm:
            out["design/clipped_frac"] = self.last_clipped
        return out

    # -- annealing ---------------------------------------------------------

    def chop(self, component_scores: dict[int, float]) -> list[int]:
        """Kill the worst half of the surviving components. Returns those killed.

        Upstream (`component_chopper.py`) samples 100 robots from each surviving
        component, rolls each out under the current policy, and retires the
        worst half by writing -1e6 into their mixing probability. This is how
        the mixture collapses to a single design -- the weights are frozen, so
        nothing else makes it commit.

        Fixed relative to the reference: upstream argsorts an array whose dead
        entries are 0.0, so with negative returns a dead component can sort
        ahead of a live one and waste a slot in the kill loop (harmless, since
        `components_left` only counts live ones, but confusing). Only live
        components are considered here.
        """
        n = self.components_left()
        if n <= 1:
            return []
        live = [i for i in np.flatnonzero(self.alive) if i in component_scores]
        killed: list[int] = []
        for i in sorted(live, key=lambda k: component_scores[k]):
            if self.components_left() <= n // 2:
                break
            self.log_mixprobs[i] = _DEAD
            killed.append(int(i))
        if killed:
            # Upstream resets BOTH optimizers after a chop. The policy's is
            # reset in ppo/train.py when it sees design/chopped -- which it did
            # NOT do before 2026-10-19, although this comment already said so.
            self._adam_mean.reset()
            self._adam_logstd.reset()
        return killed

    # -- interop with morphology.py ---------------------------------------

    @staticmethod
    def to_genes(params: np.ndarray) -> np.ndarray:
        """Map designs in [-1, 1] onto MorphologySpec genes in [0, 1].

        Clipping happens HERE, at the point the design becomes a body, and not
        in `sample` -- the unclipped value is what the log-prob needs.
        """
        return (np.clip(params, DESIGN_LO, DESIGN_HI) + 1.0) / 2.0

    @staticmethod
    def from_genes(genes: np.ndarray) -> np.ndarray:
        return np.asarray(genes) * 2.0 - 1.0

    # -- checkpointing -----------------------------------------------------
    #
    # EVERY MUTABLE FIELD HAS TO BE IN HERE, and the ones that are easy to
    # forget are the ones that matter. A resumed run restores the policy from
    # the orbax checkpoint; if the distribution came back at its init, the run
    # would silently discard every design update it had already paid for and
    # NOTHING IN THE METRICS WOULD SAY SO -- `design/mode_*` would simply jump
    # and read as a large gradient step. The pieces:
    #
    #   means / log_stds    the trained parameters.
    #   log_mixprobs        which components have been chopped. Losing this
    #                       resurrects dead components, which is not a small
    #                       error: chopping is the ONLY mechanism by which this
    #                       mixture ever commits to a design (the weights are
    #                       frozen), so a resume that forgets it restarts the
    #                       commitment schedule from 8 components.
    #   Adam moments        `chop` resets these ON PURPOSE, so a resume that
    #                       also resets them is indistinguishable from an
    #                       unscheduled chop -- the surviving components take a
    #                       few oversized steps just as they would after one.
    #   rng                 the sampling stream. Re-seeding would redraw the
    #                       same eps sequence the run already used.
    def state_dict(self) -> dict:
        bg = self.rng.bit_generator.state
        if bg["bit_generator"] != "PCG64":
            raise RuntimeError(
                f"design RNG is {bg['bit_generator']}, not PCG64; the 128-bit "
                f"word encoding in _u128_to_words does not apply to it."
            )
        return {
            "means": self.means.copy(),
            "log_stds": self.log_stds.copy(),
            "log_mixprobs": self.log_mixprobs.copy(),
            "adam_mean_m": self._adam_mean.m.copy(),
            "adam_mean_v": self._adam_mean.v.copy(),
            "adam_mean_t": np.array(self._adam_mean.t, dtype=np.int64),
            "adam_logstd_m": self._adam_logstd.m.copy(),
            "adam_logstd_v": self._adam_logstd.v.copy(),
            "adam_logstd_t": np.array(self._adam_logstd.t, dtype=np.int64),
            "rng_state": _u128_to_words(bg["state"]["state"]),
            "rng_inc": _u128_to_words(bg["state"]["inc"]),
            # These two cache a half-consumed 32-bit draw. Dropping them
            # perturbs the stream by one word on resume -- harmless, but there
            # is no reason to accept a known-wrong round trip.
            "rng_has_uint32": np.array(bg["has_uint32"], dtype=np.int64),
            "rng_uinteger": np.array(bg["uinteger"], dtype=np.int64),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore in place. Shape mismatches raise; missing optional keys warn.

        SHAPE MISMATCH IS FATAL, deliberately: resuming with a different
        `--design_components` or a different gene count against an existing
        sidecar is a configuration error, and quietly reshaping it would
        produce a run whose distribution means something other than what its
        flags say. Missing Adam/RNG keys only warn, matching how
        `ppo/train.py` already treats a checkpoint whose optimizer state it
        cannot read -- a partially restored distribution beats a dead resume.
        """
        means = np.asarray(state["means"])
        log_stds = np.asarray(state["log_stds"])
        log_mixprobs = np.asarray(state["log_mixprobs"])
        want = (self.n_components, self.n_params)
        if means.shape != want or log_stds.shape != want:
            raise ValueError(
                f"design state is {means.shape}, but this run is configured for "
                f"{want} (--design_components {self.n_components}, "
                f"{self.n_params} genes). Refusing to reshape it."
            )
        if log_mixprobs.shape != (self.n_components,):
            raise ValueError(
                f"design log_mixprobs is {log_mixprobs.shape}, expected "
                f"{(self.n_components,)}."
            )
        self.means = means.astype(float).copy()
        self.log_stds = log_stds.astype(float).copy()
        self.log_mixprobs = log_mixprobs.astype(float).copy()

        try:
            self._adam_mean.m = np.asarray(state["adam_mean_m"]).astype(float).copy()
            self._adam_mean.v = np.asarray(state["adam_mean_v"]).astype(float).copy()
            self._adam_mean.t = int(np.asarray(state["adam_mean_t"]))
            self._adam_logstd.m = (
                np.asarray(state["adam_logstd_m"]).astype(float).copy()
            )
            self._adam_logstd.v = (
                np.asarray(state["adam_logstd_v"]).astype(float).copy()
            )
            self._adam_logstd.t = int(np.asarray(state["adam_logstd_t"]))
        except KeyError as exc:
            print(
                f"WARNING: design state has no Adam moments ({exc}); they "
                f"restart from zero, which behaves like an extra chop."
            )
        try:
            self.rng.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _words_to_u128(state["rng_state"]),
                    "inc": _words_to_u128(state["rng_inc"]),
                },
                "has_uint32": int(np.asarray(state["rng_has_uint32"])),
                "uinteger": int(np.asarray(state["rng_uinteger"])),
            }
        except KeyError as exc:
            print(
                f"WARNING: design state has no RNG state ({exc}); sampling "
                f"resumes from the seeded stream and will redraw eps values "
                f"this run has already used."
            )


def feasibility_first_order(
    score: np.ndarray,
    cost: np.ndarray,
    arrival: np.ndarray,
    budget: float,
    min_arrival: float,
    cost_slack: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deb's feasibility rules (Deb 2000), best first, with the two
    constraints in priority order. Returns (order, feasible, violation).

    A design is FEASIBLE when its mean episode cost is within the budget AND it
    arrives in at least `min_arrival` of its episodes. The order is:

      1. feasible designs, by `score` (the design objective, higher first);
      2. designs that arrive but exceed the budget, by cost excess (lower first);
      3. designs that do not arrive often enough, by arrival shortfall, then
         cost excess.

    Arrival comes before cost on purpose. Summing normalised violations (Deb's
    usual choice) caps a body that never leaves the start at a violation of 1
    while cost excess is unbounded, so on a task where no body meets the budget
    the motionless body would win -- "safe by not moving". Here no body that
    fails to arrive can outrank one that crosses. `cost_slack` widens the
    budget per design (the chop passes z x SE, so a component counts as over
    budget only when it is CLEARLY over). `violation` (logged only) is the
    normalised sum.
    """
    score = np.asarray(score, dtype=float)
    excess = np.asarray(cost, dtype=float) - budget - np.asarray(cost_slack, dtype=float)
    short = min_arrival - np.asarray(arrival, dtype=float)
    arrives = short <= 0.0
    feasible = (excess <= 0.0) & arrives
    tier = np.where(feasible, 0, np.where(arrives, 1, 2))
    within = np.where(tier == 0, -score, np.where(tier == 1, excess, short))
    tiebreak = np.where(tier == 2, excess, 0.0)
    order = np.lexsort((tiebreak, within, tier))
    violation = np.maximum(excess, 0.0) / max(budget, 1.0)
    if min_arrival > 0:
        violation = violation + np.maximum(short, 0.0) / min_arrival
    return order, feasible, violation


def rank_utilities(order: np.ndarray) -> np.ndarray:
    """Utility per item from a best-first order: n-1 for the best, 0 for the
    worst. normalize_scores then standardises them, which makes the REINFORCE
    weights the centred ranks of NES fitness shaping."""
    u = np.empty(len(order), dtype=float)
    u[order] = np.arange(len(order) - 1, -1, -1, dtype=float)
    return u


class DesignLoop:
    """Host-side driver: sample designs, build bodies, score them, update the GMM.

    Everything here runs between training epochs, in Python. It is the outer
    half of Schaff's algorithm; `mjx_safety_gym.algorithms.ppo.train` calls it
    and owns nothing about designs itself.

    THE THREE PHASES are upstream's (`algorithm.py:_before_step` /
    `_update_model`), driven by `steps_before_update` and `steps_after_update`:

        t < steps_before_update            policy-only burn-in; designs are
                                           still sampled and the policy learns
                                           to condition, but the distribution
                                           is frozen. Without this the design
                                           gradient is driven by a policy that
                                           cannot yet control anything.
        middle                             joint optimization.
        tmax - t < steps_after_update      design frozen at the distribution's
                                           MODE and sampled deterministically,
                                           so the policy fine-tunes on the one
                                           body that will be reported.

    Upstream's step order (`core.py:step`) is _before_step at t, rollout,
    t += steps, _update_model at the new t. So, with `start`/`end` the design
    axis at either end of an iteration, everything below is upstream's
    condition verbatim:

        sampled stochastically   tmax - start >= steps_after
        design update            end >= steps_before and tmax - end >= steps_after
        its lr                   robot_lr * max(0, 1 - start / tmax)
        chop, after the update   end // chop_freq != start // chop_freq
                                 and start > steps_before and end < tmax
                                 (the loop stops at t >= tmax first)

    CHOP SCORES (`chop_scores`). "rollout" (default since 2026-10-19) is
    upstream's ComponentChopper: `chop_designs` (100) designs drawn from EACH
    live component, one episode per lane from a fresh reset under the current
    stochastic policy, scored by the design objective, averaged per component.
    Every lane is used, so each design gets num_envs // (100 x live) lanes (5
    at 4096 envs and 8 components) where upstream ran one episode: the same
    estimate with less episode noise. "ema" is what every run before then did:
    an exponential moving average (0.9) of the scores the component's own
    samples earned in training. That was documented as pooling far more than
    100 samples, but per DESIGN its effective window is ~19 samples, spread
    over iterations in which both the component and the policy moved, and the
    components are compared on scores taken at different times. Upstream
    compares them under one policy at one moment.
    """

    def __init__(
        self,
        gmm: GmmDesignDistribution,
        spec_factory,
        batch_builder,
        wrapper,
        num_morphologies: int,
        num_envs: int,
        tmax: int,
        steps_before_update: int = 0,
        steps_after_update: int = 0,
        chop_freq: int | None = None,
        ema: float = 0.9,
        objective: str = "time",
        action_repeat: int = 1,
        max_decisions: int | None = None,
        return_ceiling: float | None = None,
        cost_weight: float = 0.0,
        score_episodes: str = "all",
        chop_scores: str = "rollout",
        chop_designs: int = 100,
        ranking: str = "objective",
        budget: float | None = None,
        min_arrival: float = 0.5,
        chop_cost_z: float = 2.0,
    ) -> None:
        # How designs are ranked for the update AND the chops. "objective"
        # (Schaff): by the design objective alone; cost never enters.
        # "feasibility_first": Deb's rules on (objective, cost, arrival) via
        # `feasibility_first_order`, with rank utilities in the update and a
        # cost allowance of chop_cost_z standard errors at the chops (a chop
        # is irreversible, so only a CLEARLY infeasible component loses on
        # feasibility).
        if ranking not in ("objective", "feasibility_first"):
            raise ValueError(f"ranking must be 'objective' or 'feasibility_first', got {ranking!r}")
        if ranking == "feasibility_first" and budget is None:
            raise ValueError("ranking='feasibility_first' needs the safety budget")
        self.ranking = ranking
        self.budget = budget
        self.min_arrival = float(min_arrival)
        self.chop_cost_z = float(chop_cost_z)
        if chop_scores not in ("rollout", "ema"):
            raise ValueError(f"chop_scores must be 'rollout' or 'ema', got {chop_scores!r}")
        self.chop_scores = chop_scores
        self.chop_designs = int(chop_designs)
        # "all" (default since 2026-10-19, as upstream): a design's score is the
        # mean over EVERY episode its lanes finished this iteration. "last"
        # reproduces earlier runs: one (the most recent) episode per lane.
        if score_episodes not in ("all", "last"):
            raise ValueError(f"score_episodes must be 'all' or 'last', got {score_episodes!r}")
        self.score_episodes = score_episodes
        self.gmm = gmm
        self._spec_factory = spec_factory  # genes array -> MorphologySpec
        self._batch_builder = batch_builder  # (specs, replicas) -> (base, fields, genes)
        self._wrapper = wrapper
        self.num_morphologies = int(num_morphologies)
        self.num_envs = int(num_envs)
        self.tmax = int(tmax)
        self.steps_before_update = int(steps_before_update)
        self.steps_after_update = int(steps_after_update)
        self.chop_freq = chop_freq
        self.ema = float(ema)
        if objective not in ("time", "return", "safe_time", "speed"):
            raise ValueError(f"unknown design objective {objective!r}")
        self.objective = objective
        self.action_repeat = int(action_repeat)
        # "safe_time": time fitness PLUS cost_weight * episode cost, both in
        # "decisions" (lower is better before negation). A body is then judged
        # on how fast AND how cleanly it crosses -- the co-design question this
        # project exists to ask, made askable once the cost is graded
        # (2026-09-17). cost_weight 1.0 makes a careless crossing's ~58 cost
        # units worth ~half an arrival time on the nominal ant.
        self.cost_weight = float(cost_weight)
        if objective == "safe_time" and self.cost_weight <= 0.0:
            raise ValueError("objective='safe_time' needs cost_weight > 0")
        if objective in ("time", "safe_time", "speed"):
            if not max_decisions or not return_ceiling:
                raise ValueError(
                    "objective='time' needs max_decisions and return_ceiling"
                )
            self.max_decisions = int(max_decisions)
            self.return_ceiling = float(return_ceiling)
        else:
            self.max_decisions = max_decisions
            self.return_ceiling = return_ceiling
        self._aux: dict[str, float] = {}

        self._pending = None  # (params, comps) awaiting a score
        self._comp_scores: dict[int, float] = {}
        self._last_chop = 0
        # Step of the last COMPLETED iteration, ON THE DESIGN AXIS. Tracked
        # separately from `history` because history is in-memory only: on a
        # resume it is empty, and reading the step off it would report 0 and
        # un-freeze a run that was in its final fine-tune phase.
        self._last_step = 0
        # THE TRAINER'S STEP COUNTER RESTARTS AT ZERO ON A RESUME.
        # `ppo/train.py` restores four things from the checkpoint -- normalizer,
        # network params, penalizer params, optimizer state -- and `env_steps`
        # is not among them, so `current_step` counts from 0 again. Every
        # schedule in this class is expressed in absolute steps (`lr_frac`, the
        # burn-in and fine-tune phases, and the chop interval), so taking the
        # trainer's number at face value after a resume would rewind all three.
        # The chop is the one that fails silently: `_last_chop` comes back as
        # an absolute step from before the crash, so `step - _last_chop` stays
        # negative until the resumed run has RE-CLIMBED past the pre-crash step
        # count. A run that crashed at 120M would not chop again until trainer
        # step 120M of the resume, and a resume shorter than that never chops
        # at all -- so the mixture never commits to a design, which is the
        # entire point of the method. This offset keeps a private, monotone
        # axis: zero on a fresh run (identical behaviour), and on a resume it
        # continues from where the checkpoint left off.
        self._step_offset = 0
        self.history: list[dict] = []

    # -- called from inside the jitted reset ------------------------------

    def install(self, fields, genes, components) -> None:
        """Assign the design onto the wrapper. Runs DURING TRACING.

        `fields`/`genes`/`components` are tracers at this point, which is the
        whole trick: assigning them here makes them arguments of the compiled
        reset instead of constants closed over by it, so a new population costs
        no recompile. `components` is each lane's GMM component, for the
        per-component Lagrange multipliers.
        """
        self._wrapper._fields = fields
        self._wrapper._genes = genes
        self._wrapper._components = components

    # -- per-iteration ----------------------------------------------------

    def _frozen(self, step: int) -> bool:
        """Is the design sampled at the mode (final fine-tune) from `step` on?

        Upstream's `not stochastic`, `tmax - t < steps_after_robot_update`,
        verbatim. Before 2026-10-19 this carried a `steps_after_update > 0`
        guard; it only ever differed past tmax, where upstream's loop has
        already stopped.
        """
        return (self.tmax - step) < self.steps_after_update

    def _updating(self, step: int) -> bool:
        """Upstream's design-update condition at the END of an iteration
        (`algorithm.py:116`): the burn-in is over and the fine-tune has not
        begun. Its last iteration can end past tmax, which this excludes as
        upstream does (`tmax - t >= steps_after` fails there even at 0)."""
        return (
            step >= self.steps_before_update
            and (self.tmax - step) >= self.steps_after_update
        )

    def _lanes(self, params: np.ndarray, comps: np.ndarray, n_devices: int, n_envs: int, reps: int):
        """Build `params` as bodies, `reps` contiguous lanes each, padded to
        `n_envs` lanes with copies of design 0. Device-shaped (fields, genes,
        components), plus how many leading lanes are real.

        Built 64 designs at a time: build_design_batch puts every model on the
        device before stacking its fields, and a chop draws up to 800.
        """
        import jax.numpy as jp

        n_real = len(params) * reps
        if n_real > n_envs or reps < 1:
            raise ValueError(f"{len(params)} designs x {reps} lanes do not fit {n_envs} lanes")
        chunks = []
        genes01 = self.gmm.to_genes(params)
        for i in range(0, len(params), 64):
            specs = [self._spec_factory(g) for g in genes01[i : i + 64]]
            _, f, g = self._batch_builder(specs, reps)
            chunks.append((f, g))
        fields = {k: jp.concatenate([f[k] for f, _ in chunks]) for k in chunks[0][0]}
        genes = jp.concatenate([g for _, g in chunks])
        lane_comps = np.repeat(comps, reps)
        if n_real < n_envs:
            pad = n_envs - n_real
            fields = {k: jp.concatenate([v, jp.repeat(v[:1], pad, axis=0)]) for k, v in fields.items()}
            genes = jp.concatenate([genes, jp.repeat(genes[:1], pad, axis=0)])
            lane_comps = np.concatenate([lane_comps, np.full(pad, lane_comps[0])])
        fields = {k: jp.reshape(v, (n_devices, -1) + v.shape[1:]) for k, v in fields.items()}
        genes = jp.reshape(genes, (n_devices, -1) + genes.shape[1:])
        # Lanes are BLOCKED (each design's `reps` lanes are contiguous), so the
        # lane -> component map is a repeat.
        components = jp.reshape(jp.asarray(lane_comps, dtype=jp.int32), (n_devices, -1))
        return fields, genes, components, n_real

    def sample(self, n_devices: int, n_envs: int):
        """Draw a population and compile it.

        Returns device-shaped (fields, genes, components), `components` being
        each lane's GMM component index.
        """
        step = self._last_step
        if self._frozen(step):
            params = np.tile(self.gmm.mode(), (self.num_morphologies, 1))
            # The component the mode belongs to, not 0: component 0 may have
            # been chopped, and its Lagrange multiplier would be stale.
            comps = np.full(self.num_morphologies, self.gmm.mode_component(), dtype=int)
        else:
            params, comps = self.gmm.sample(self.num_morphologies)
        self._pending = (params, comps)
        self._shape = (n_devices, n_envs)  # the chop rollouts reuse it
        fields, genes, components, _ = self._lanes(
            params, comps, n_devices, n_envs, n_envs // self.num_morphologies
        )
        return fields, genes, components

    def scores_from(self, env_state) -> tuple[np.ndarray, np.ndarray]:
        """Mean score per design (HIGHER IS BETTER), and episodes counted.

        Lanes are BLOCKED, not interleaved -- `build_design_batch` repeats each
        spec `replicas` times contiguously -- so a flat reshape recovers the
        per-design grouping.

        FOUR OBJECTIVES, and the default is NOT the upstream one. `"speed"`
        (default since 2026-09-18, the user's call) is net +x per decision with
        a non-arrival charged the full horizon; see `_lane_scores`. `"safe_time"`
        is `"time"` plus cost_weight x episode cost. The other two:

        `objective="time"` scores a design by how long it takes to
        reach the goal, negated so that higher is better. This is the same
        scalar `scripts/eval_morphology.time_fitness` computes, so a design's
        training score and its offline evaluation are the same quantity:

            arrived      -> the arrival decision
            did not      -> max_decisions * (1 + shortfall)

        with `shortfall = clip(1 - return/ceiling, 0, 1)`. Every arrival
        outranks every non-arrival, and non-arrivals are ordered by how far
        short they fell.

        IT DEGRADES GRACEFULLY EARLY IN TRAINING, which is the reason it is safe
        to switch on from step 0. Before any design arrives, every fitness is
        `max_decisions * (1 + shortfall)`, which is monotone in return -- so the
        ranking is EXACTLY the one `objective="return"` would give, i.e. "how far
        did it get". Only once bodies start arriving does the objective start
        discriminating on speed, which is precisely when return stops being able
        to. No burn-in is needed to keep the signal alive.

        `objective="return"` is Schaff's own score (`algorithm.py` reads
        `env.reward_buffer[-1]`) and is kept for the faithful baseline. IT IS
        NEARLY BLIND ON THIS TASK, which is why it is not the default. The
        reward telescopes to `dx + (d_start - d_end) + healthy`, so a full
        traverse scores the goal distance twice NO MATTER HOW LONG IT TOOK --
        measured on the 50M conditioned run, arrival times spanned 175-328
        decisions (1.87x) while returns spanned 22.49-22.51 (std 0.003). Worse,
        the only speed-dependent term left points the WRONG WAY: `healthy` is
        paid per inner step, so under terminate_on_goal a slower body that also
        arrives collects MORE of it (~0.24 vs ~0.13 for the ant, on a base of
        22). Under `objective="return"` REINFORCE can therefore separate
        "arrives" from "does not arrive" and essentially nothing else.

        With `objective="time"` arrival time IS the episode length, because
        terminate_on_goal ends the episode on contact with the goal --
        `train_ppo.validate()` refuses the combination without it, since
        otherwise every arriving body reads the truncation cap and the
        objective silently degenerates to a constant.
        """
        info = env_state.info
        counts = np.asarray(info["ep_count"]).reshape(self.num_morphologies, -1).sum(axis=1)
        if self.score_episodes == "all":
            scores, self._aux = self._episode_means(info, counts)
        else:
            lane, self._aux = self._lane_scores(info)
            scores = lane.reshape(self.num_morphologies, -1).mean(axis=1)
        return scores, counts

    def _cost_arrival(self, info, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per design: mean episode cost and share of episodes that arrived,
        over the same episodes its score uses (all finished ones, or each
        lane's last)."""
        shape = (self.num_morphologies, -1)
        if self.score_episodes == "all":
            n = np.maximum(counts, 1.0)
            tot = lambda k: np.asarray(info[k], dtype=float).reshape(shape).sum(axis=1)
            return tot("ep_cost_sum") / n, tot("ep_arrived_sum") / n
        cost = np.asarray(info["ep_cost_last"], dtype=float).reshape(shape).mean(axis=1)
        arr = (np.asarray(info["ep_arrived_last"]).reshape(shape) > 0).mean(axis=1)
        return cost, arr

    def _lane_scores(self, info) -> tuple[np.ndarray, dict[str, float]]:
        """Each lane's LAST finished episode, scored (higher is better), flat
        over lanes; plus the aux metrics in natural units.

        `score_episodes="last"` averages these per design. The chop rollouts
        use them too: there every lane has run exactly one episode.
        """
        ret = np.asarray(info["ep_return_last"], dtype=float).ravel()
        if self.objective == "return":
            return ret, {}
        arrived = np.asarray(info["ep_arrived_last"]).ravel() > 0
        decisions = np.asarray(info["ep_len_last"], dtype=float).ravel() / self.action_repeat
        arr = {
            "design/arrival_rate": float(arrived.mean()),
            "design/arrival_decisions": (
                float(decisions[arrived].mean()) if arrived.any() else float("nan")
            ),
        }
        if self.objective == "speed":
            # THE DEFAULT SINCE 2026-09-18 (user's call): distance travelled per
            # decision. Net +x displacement of the episode over its duration --
            # but a NON-arrival is charged the full horizon, not the time it
            # actually lasted, or "dash two metres and fall over" would score
            # like a crossing. With that charge it is monotone in both distance
            # and time and needs no arrival/miss special-casing: an arrival at
            # 100 decisions scores 10.5/100, a 3 m flip 3/625. Higher is better,
            # so it is returned un-negated. Requires info["ep_dx_last"], which
            # the corridor tasks provide (RunForward info["dx"]).
            if "ep_dx_last" not in info:
                raise KeyError("objective='speed' needs info['ep_dx_last'] (corridor tasks only)")
            dx = np.asarray(info["ep_dx_last"], dtype=float).ravel()
            charged = np.where(arrived, decisions, float(self.max_decisions))
            speed = dx / np.maximum(charged, 1.0)
            return speed, {"design/speed_mean": float(speed.mean()), **arr}
        shortfall = np.clip(1.0 - ret / self.return_ceiling, 0.0, 1.0)
        fitness = np.where(arrived, decisions, self.max_decisions * (1.0 + shortfall))
        aux: dict[str, float] = {}
        if self.objective == "safe_time":
            cost = np.asarray(info["ep_cost_last"], dtype=float).ravel()
            fitness = fitness + self.cost_weight * cost
            aux["design/cost_mean"] = float(cost.mean())
        # Reported in the natural units as well as the negated score, because
        # "design/score_mean = -212" is not a number anyone can sanity-check
        # against a rollout, and 212 decisions is.
        aux.update({"design/fitness_decisions": float(fitness.mean()), **arr})
        return -fitness, aux

    def _episode_means(
        self, info, counts: np.ndarray, groups: int | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Per design, the mean score over EVERY episode its lanes finished
        this iteration (upstream: np.mean(ep_rews), `envs.py:296`), from the
        per-lane sums EpisodeStatsWrapper keeps. Same per-episode quantity as
        `_lane_scores`; aux metrics are means over all finished episodes.
        `groups` (default: one per design) is how many blocks the lanes form."""
        shape = (groups or self.num_morphologies, -1)

        def total(key):
            if key not in info:
                raise KeyError(f"score_episodes='all' needs info[{key!r}] (EpisodeStatsWrapper)")
            return np.asarray(info[key], dtype=float).reshape(shape).sum(axis=1)

        n = np.maximum(counts, 1.0)
        n_all = max(float(counts.sum()), 1.0)
        n_arr = float(total("ep_arrived_sum").sum())
        arr = {
            "design/arrival_rate": n_arr / n_all,
            "design/arrival_decisions": (
                float(total("ep_arrived_len_sum").sum()) / n_arr / self.action_repeat
                if n_arr > 0 else float("nan")
            ),
        }
        if self.objective == "return":
            return total("ep_return_sum") / n, {}
        if self.objective == "speed":
            if "ep_dx_last" not in info:
                raise KeyError("objective='speed' needs info['ep_dx_last'] (corridor tasks only)")
            # Per-episode speeds are summed in inner steps; x action_repeat
            # makes them per decision.
            speed = total("ep_speed_sum") * self.action_repeat
            return speed / n, {"design/speed_mean": float(speed.sum()) / n_all, **arr}
        fitness = total("ep_fitness_sum")
        if not np.all(np.isfinite(fitness)):
            raise RuntimeError(
                "ep_fitness_sum is NaN: EpisodeStatsWrapper was built without "
                "return_ceiling, so the time fitness was never accumulated."
            )
        aux: dict[str, float] = {}
        if self.objective == "safe_time":
            cost = total("ep_cost_sum")
            fitness = fitness + self.cost_weight * cost
            aux["design/cost_mean"] = float(cost.sum()) / n_all
        aux.update({"design/fitness_decisions": float(fitness.sum()) / n_all, **arr})
        return -fitness / n, aux

    def _chop_due(self, start: int, end: int) -> bool:
        """Upstream's chop rule (`algorithm.py:102-106`), verbatim since
        2026-10-19: at `_before_step` with t = `end` and t_prev = `start`,
        chop when the iteration crossed a multiple of chop_freq and began after
        the burn-in. With Schaff's ant schedule (burn-in 200M, chop_freq 200M)
        that is 400M, 600M, 800M. `end < tmax` because upstream's loop stops at
        t >= tmax before another _before_step can run.

        INDEPENDENT OF THE DESIGN UPDATE, as upstream. Until 2026-10-19 a chop
        could only happen on an iteration that also updated the design, so
        none happened in the final fine-tune or on a skipped update; on every
        schedule actually run the chops fell on the same iterations.
        """
        if not self.chop_freq:
            return False
        f = int(self.chop_freq)
        return (
            end // f != start // f
            and start > self.steps_before_update
            and end < self.tmax
            and self.gmm.components_left() > 1
        )

    def _chop_component_scores(self, rollout_fn) -> tuple[dict[int, float], dict[str, float]]:
        """Upstream's ComponentChopper.chop scoring (`component_chopper.py:30-42`).

        `chop_designs` designs from EACH live component (`sample_gaussian(i)`),
        one episode per lane from a fresh reset under the current stochastic
        policy (`test_robot` -> `actor.act`, which samples), scored by this
        loop's objective and averaged per component. `rollout_fn(fields, genes,
        components)` runs the episodes and returns each lane's FIRST finished
        episode as flat EpisodeStatsWrapper latches; see ppo/train.py.
        """
        if rollout_fn is None:
            raise RuntimeError(
                "chop_scores='rollout' needs a rollout_fn from the trainer to "
                "score the components before a chop"
            )
        live = np.flatnonzero(self.gmm.alive)
        params = np.concatenate(
            [self.gmm.sample_component(int(i), self.chop_designs) for i in live]
        )
        comps = np.repeat(live, self.chop_designs)
        n_devices, n_envs = self._shape
        reps = n_envs // len(params)
        fields, genes, components, n_real = self._lanes(params, comps, n_devices, n_envs, reps)
        info = rollout_fn(fields, genes, components)
        lane, _ = self._lane_scores(info)
        # SELF-CHECK. Each lane ran exactly one episode, so its sums hold that
        # same episode: the all-episode path must give the same score as the
        # last-episode path, lane for lane, or the two have drifted apart.
        real = {k: v[:n_real] for k, v in info.items()}
        via_sums, _ = self._episode_means(real, real["ep_count"], groups=n_real)
        gap = float(np.max(np.abs(via_sums - lane[:n_real]) / (1.0 + np.abs(lane[:n_real]))))
        if not gap < 1e-4:
            raise RuntimeError(
                f"chop rollout: the all-episode and last-episode scores of the "
                f"same single episodes differ by {gap:.3g} (relative)"
            )
        design = lane[:n_real].reshape(len(params), reps).mean(axis=1)
        scores = {int(c): float(design[comps == c].mean()) for c in live}
        metrics = {
            "design/chop_designs": float(len(params)),
            "design/chop_lanes_per_design": float(reps),
            "design/chop_paths_gap": gap,
            **{f"design/chop_score_c{c}": v for c, v in scores.items()},
        }
        if self.ranking == "feasibility_first":
            # Per component, over its designs (each the mean of its lanes):
            # mean cost with its standard error, and the arrival share.
            cost_d = real["ep_cost_last"].reshape(len(params), reps).mean(axis=1)
            arr_d = (real["ep_arrived_last"] > 0).reshape(len(params), reps).mean(axis=1)
            sel = [comps == c for c in live]
            cost = np.array([cost_d[m].mean() for m in sel])
            se = np.array([cost_d[m].std() / np.sqrt(max(m.sum(), 1)) for m in sel])
            arr = np.array([arr_d[m].mean() for m in sel])
            obj = np.array([scores[int(c)] for c in live])
            order, feas, _ = feasibility_first_order(
                obj, cost, arr, self.budget, self.min_arrival, self.chop_cost_z * se
            )
            util = rank_utilities(order)
            for c, k, e, a, f in zip(live, cost, se, arr, feas):
                metrics[f"design/chop_cost_c{c}"] = float(k)
                metrics[f"design/chop_cost_se_c{c}"] = float(e)
                metrics[f"design/chop_arrival_c{c}"] = float(a)
                metrics[f"design/chop_feasible_c{c}"] = float(f)
            scores = {int(c): float(u) for c, u in zip(live, util)}
        return scores, metrics

    def finish_iteration(self, env_state, step: int, rollout_fn=None) -> dict:
        """Score the population, take one REINFORCE step, chop on schedule.

        `step` arrives on the TRAINER's axis, which restarts at zero after a
        resume; `_step_offset` maps it onto the design axis. See __init__.
        `rollout_fn` is needed only when a chop is due under
        chop_scores="rollout"; see `_chop_component_scores`.
        """
        start = self._last_step  # the design axis when this iteration began
        step = int(step) + self._step_offset
        params, comps = self._pending
        scores, counts = self.scores_from(env_state)
        metrics: dict[str, float] = {
            "design/score_mean": float(scores.mean()),
            "design/score_spread": float(scores.max() - scores.min()),
            "design/episodes_per_design": float(counts.mean()),
            "design/components": float(self.gmm.components_left()),
        }
        metrics.update(self._aux)

        # A DESIGN WITH NO COMPLETED EPISODE MUST NOT ENTER THE UPDATE. Its
        # `ep_return_last` is still the reset value (0.0), which after
        # standardisation is not "no information" but a confident-looking score
        # in the middle of the pack -- REINFORCE would then move the
        # distribution on a number that was never measured. Bodies that neither
        # flip nor arrive inside the iteration window hit this; the first smoke
        # run had 2 of 8. Masking is the honest fix, a long enough iteration is
        # the real one, which is why episodes_per_design is reported. (Upstream
        # never meets this on the ant: a 1000-step episode always ends inside
        # its 1024-step rollout.)
        valid = counts >= 1
        if not valid.all():
            metrics["design/UNSCORED_DESIGNS"] = float((~valid).sum())

        if self.chop_scores == "ema":
            for c, s in zip(comps[valid], scores[valid]):
                prev = self._comp_scores.get(int(c))
                self._comp_scores[int(c)] = (
                    float(s) if prev is None else self.ema * prev + (1 - self.ema) * float(s)
                )

        # Two valid designs is the minimum for a meaningful baseline: the
        # standardisation in `normalize_scores` is taken over the designs
        # sampled this iteration, and a lone sample standardises to 0.
        n_valid = int(valid.sum())
        active = self._updating(step) and n_valid >= 2
        if n_valid < 2:
            metrics["design/SKIPPED_UPDATE"] = 1.0
        utility = scores
        cost, arr = self._cost_arrival(env_state.info, counts)
        if valid.any():  # logged under every ranking, for the training curves
            metrics["design/design_cost_mean"] = float(cost[valid].mean())
            metrics["design/design_arrival_mean"] = float(arr[valid].mean())
            if self.budget is not None:
                metrics["design/under_budget_frac"] = float((cost[valid] <= self.budget).mean())
        if self.ranking == "feasibility_first":
            utility = np.full_like(scores, np.nan)
            if valid.any():
                order, feas, viol = feasibility_first_order(
                    scores[valid], cost[valid], arr[valid], self.budget, self.min_arrival
                )
                utility[valid] = rank_utilities(order)
                metrics["design/feasible_frac"] = float(feas.mean())
                metrics["design/violation_mean"] = float(viol.mean())
        if active:
            # Upstream computes _lr_frac in _before_step, i.e. at the START.
            lr_frac = max(0.0, 1.0 - start / self.tmax)
            metrics.update(
                self.gmm.update(params[valid], comps[valid], utility[valid], lr_frac)
            )
            metrics["design/lr_frac"] = lr_frac
        if self._chop_due(start, step):
            if self.chop_scores == "rollout":
                comp_scores, chop_metrics = self._chop_component_scores(rollout_fn)
                metrics.update(chop_metrics)
            else:
                comp_scores = dict(self._comp_scores)
            killed = self.gmm.chop(comp_scores)
            if killed:
                self._last_chop = step
                self._comp_scores = {
                    k: v for k, v in self._comp_scores.items() if k not in killed
                }
                metrics["design/chopped"] = float(len(killed))
                # Re-read AFTER the chop. `design/components` is captured
                # at the top of this method, before the chop that happens
                # below it, so a chop iteration reported the PRE-chop count
                # -- job 46267 logged `components=8` and `chopped=4` on the
                # same line, which reads like a contradiction. This is the
                # metric anyone uses to follow the 8->4->2->1 schedule, so
                # it should name the state the iteration ended in.
                metrics["design/components"] = float(self.gmm.components_left())
        metrics["design/updating"] = float(active)

        mode = self.gmm.mode()
        for i, v in enumerate(mode):
            metrics[f"design/mode_{i}"] = float(v)
        self._last_step = int(step)
        self.history.append(
            {"step": step, "scores": scores.tolist(), "mode": mode.tolist()}
        )
        return metrics

    # -- checkpointing -----------------------------------------------------

    SIDECAR = "design_state.npz"

    def state_dict(self) -> dict:
        """Everything a resume needs, flattened to arrays for `np.savez`.

        The GMM's own state plus the two things that live out here: the
        per-component score EMA that decides the next chop, and when the last
        chop happened. Lose the EMA and the next chop is taken on a window that
        starts at the resume, which is a different (much shorter) average than
        the schedule intends -- and chopping is irreversible.
        """
        idx = sorted(self._comp_scores)
        state = {f"gmm_{k}": v for k, v in self.gmm.state_dict().items()}
        state["comp_score_idx"] = np.array(idx, dtype=np.int64)
        state["comp_score_val"] = np.array(
            [self._comp_scores[i] for i in idx], dtype=float
        )
        state["last_chop"] = np.array(self._last_chop, dtype=np.int64)
        state["last_step"] = np.array(self._last_step, dtype=np.int64)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.gmm.load_state_dict(
            {k[len("gmm_") :]: v for k, v in state.items() if k.startswith("gmm_")}
        )
        idx = np.asarray(state["comp_score_idx"]).reshape(-1)
        val = np.asarray(state["comp_score_val"]).reshape(-1)
        self._comp_scores = {int(i): float(v) for i, v in zip(idx, val)}
        self._last_chop = int(np.asarray(state["last_chop"]))
        self._last_step = int(np.asarray(state["last_step"]))
        # Future trainer steps stack on top of where the checkpoint stopped.
        # Idempotent: saving mid-resume writes an already-absolute `last_step`,
        # so loading that again reproduces the same offset.
        self._step_offset = self._last_step

    def save(self, directory) -> str:
        """Write the sidecar into an existing checkpoint directory.

        A SIDECAR, not a fifth element of the orbax tuple. That tuple is
        unpacked POSITIONALLY by three separate loaders (`ppo/train.py`,
        `main.py`, `scripts/eval_checkpoint.py`), and this repo has already
        been bitten once by orbax returning a saved dataclass as a
        dict whose leaves sort alphabetically -- see `_load_checkpoint`, where
        that would have loaded the cost critic into the policy. Design state is
        host-side numpy that no policy loader wants; keeping it in its own file
        means adding it cannot perturb what those three already read.
        """
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} does not exist, so brax's checkpoint layout is "
                f"not what this expected ('<logdir>/<step:012d>'). The design "
                f"state would have been written somewhere the resume path "
                f"never looks."
            )
        path = directory / self.SIDECAR
        np.savez(path, **self.state_dict())
        return str(path)

    def load(self, directory) -> bool:
        """Restore from a checkpoint directory. False if there is no sidecar."""
        path = pathlib.Path(directory) / self.SIDECAR
        if not path.is_file():
            return False
        with np.load(path) as f:
            self.load_state_dict({k: f[k] for k in f.files})
        return True

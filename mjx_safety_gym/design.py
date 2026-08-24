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
    """Minimal Adam, matching `MpiAdam(epsilon=1e-5, beta1=robot_momentum)`.

    Separate from the policy optimizer on purpose: upstream keeps a distinct
    `mpi_adam_robot` and RESETS it after every chop, because the surviving
    components face a different objective landscape once their neighbours are
    gone and stale moments would drag them.
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
        """Returns the update to SUBTRACT from the parameter."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        mhat = self.m / (1 - self.beta1**self.t)
        vhat = self.v / (1 - self.beta2**self.t)
        return lr * mhat / (np.sqrt(vhat) + self.eps)


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
    ) -> None:
        self.n_params = int(n_params)
        self.n_components = int(n_components)
        self.rng = np.random.default_rng(seed)
        self.lr = float(lr)

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

    def mode(self) -> np.ndarray:
        """The highest-density design, matching `GmmPd.mode`.

        Upstream compares `logp(mode_i) + log_mix_i` across components and takes
        the argmax. With uniform weights that reduces to the component with the
        smallest total log-std, i.e. the tightest one -- but it is written out
        so the behaviour still holds if weights ever become trainable.
        """
        live = np.flatnonzero(self.alive)
        scores = [self.logp(self.means[i][None], np.array([i]))[0] for i in live]
        return self.means[live[int(np.argmax(scores))]].copy()

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
        """
        params = np.asarray(params, dtype=float)
        scores = np.asarray(scores, dtype=float)
        n = len(comps)
        mu = self.means[comps]
        sd = np.exp(self.log_stds[comps])
        z = (params - mu) / sd

        d_mu = -(params - mu) / sd**2 * scores[:, None]
        d_logsd = (1.0 - z**2) * scores[:, None]

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
        return {
            "design/loss": float(np.mean(self.neglogp(params, comps) * norm)),
            "design/grad_norm": float(np.linalg.norm(g_mu)),
            "design/components": float(self.components_left()),
            "design/mean_std": float(np.exp(self.log_stds[self.alive]).mean()),
            "design/dead": float(dead.sum()),
        }

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
            # Upstream resets BOTH optimizers after a chop; the policy's is
            # reset by the caller.
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

    DEVIATION, stated plainly: upstream scores a component for chopping by
    drawing 100 designs from it and rolling each out under the current policy
    (`component_chopper.py`). Here each component's score is an exponential
    moving average of the scores its own samples already earned during
    training (see `scores_from` -- time-to-goal by default, not return). The
    DECISION RULE is unchanged -- kill the worst half, 8 -> 4 -> 2 -> 1, reset
    Adam -- but the estimate costs no extra rollouts and is
    pooled over far more samples than 100. It is lagged, though: the average
    spans a window over which both the component and the policy moved, where
    upstream's is computed fresh at the moment of the chop.
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
    ) -> None:
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
        if objective not in ("time", "return"):
            raise ValueError(f"unknown design objective {objective!r}")
        self.objective = objective
        self.action_repeat = int(action_repeat)
        if objective == "time":
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

    def install(self, fields, genes) -> None:
        """Assign the design onto the wrapper. Runs DURING TRACING.

        `fields`/`genes` are tracers at this point, which is the whole trick:
        assigning them here makes them arguments of the compiled reset instead
        of constants closed over by it, so a new population costs no recompile.
        """
        self._wrapper._fields = fields
        self._wrapper._genes = genes

    # -- per-iteration ----------------------------------------------------

    def _frozen(self, step: int) -> bool:
        """Is the design frozen at the mode for the final fine-tune phase?

        The `> 0` guard is load-bearing: without it the default
        `steps_after_update = 0` makes this true the moment `step` passes
        `tmax`, which silently stops updating the distribution for the last
        iterations of every run. Caught in the first smoke test, where
        `design/updating` read 0 at the final eval.
        """
        return self.steps_after_update > 0 and (
            self.tmax - step
        ) < self.steps_after_update

    def sample(self, n_devices: int, n_envs: int):
        """Draw a population and compile it. Returns device-shaped (fields, genes)."""
        step = self._last_step
        if self._frozen(step):
            params = np.tile(self.gmm.mode(), (self.num_morphologies, 1))
            comps = np.zeros(self.num_morphologies, dtype=int)
        else:
            params, comps = self.gmm.sample(self.num_morphologies)
        self._pending = (params, comps)

        specs = [self._spec_factory(g) for g in self.gmm.to_genes(params)]
        replicas = n_envs // self.num_morphologies
        _, fields, genes = self._batch_builder(specs, replicas)
        import jax.numpy as jp

        fields = {
            k: jp.reshape(v, (n_devices, -1) + v.shape[1:]) for k, v in fields.items()
        }
        genes = jp.reshape(genes, (n_devices, -1) + genes.shape[1:])
        return fields, genes

    def scores_from(self, env_state) -> tuple[np.ndarray, np.ndarray]:
        """Mean score per design (HIGHER IS BETTER), and episodes counted.

        Lanes are BLOCKED, not interleaved -- `build_design_batch` repeats each
        spec `replicas` times contiguously -- so a flat reshape recovers the
        per-design grouping.

        TWO OBJECTIVES, and the default is NOT the upstream one:

        `objective="time"` (default) scores a design by how long it takes to
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
        shape = (self.num_morphologies, -1)
        cnt = np.asarray(info["ep_count"]).reshape(shape)
        ret = np.asarray(info["ep_return_last"]).reshape(shape)
        counts = cnt.sum(axis=1)

        if self.objective == "return":
            self._aux = {}
            return ret.mean(axis=1), counts

        arrived = np.asarray(info["ep_arrived_last"]).reshape(shape) > 0
        decisions = (
            np.asarray(info["ep_len_last"]).reshape(shape) / self.action_repeat
        )
        shortfall = np.clip(1.0 - ret / self.return_ceiling, 0.0, 1.0)
        fitness = np.where(
            arrived, decisions, self.max_decisions * (1.0 + shortfall)
        )
        # Reported in the natural units as well as the negated score, because
        # "design/score_mean = -212" is not a number anyone can sanity-check
        # against a rollout, and 212 decisions is.
        self._aux = {
            "design/fitness_decisions": float(fitness.mean()),
            "design/arrival_rate": float(arrived.mean()),
            "design/arrival_decisions": (
                float(decisions[arrived].mean()) if arrived.any() else float("nan")
            ),
        }
        return -fitness.mean(axis=1), counts

    def finish_iteration(self, env_state, step: int) -> dict:
        """Score the population, take one REINFORCE step, chop on schedule.

        `step` arrives on the TRAINER's axis, which restarts at zero after a
        resume; `_step_offset` maps it onto the design axis. See __init__.
        """
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
        # the real one, which is why episodes_per_design is reported.
        valid = counts >= 1
        if not valid.all():
            metrics["design/UNSCORED_DESIGNS"] = float((~valid).sum())

        for c, s in zip(comps[valid], scores[valid]):
            prev = self._comp_scores.get(int(c))
            self._comp_scores[int(c)] = (
                float(s) if prev is None else self.ema * prev + (1 - self.ema) * float(s)
            )

        # Two valid designs is the minimum for a meaningful baseline: the
        # standardisation in `normalize_scores` is taken over the designs
        # sampled this iteration, and a lone sample standardises to 0.
        n_valid = int(valid.sum())
        active = (
            step >= self.steps_before_update
            and not self._frozen(step)
            and n_valid >= 2
        )
        if n_valid < 2:
            metrics["design/SKIPPED_UPDATE"] = 1.0
        if active:
            lr_frac = max(0.0, 1.0 - step / self.tmax)
            metrics.update(
                self.gmm.update(params[valid], comps[valid], scores[valid], lr_frac)
            )
            metrics["design/lr_frac"] = lr_frac
            if self.chop_freq and step - self._last_chop >= self.chop_freq:
                killed = self.gmm.chop(dict(self._comp_scores))
                if killed:
                    self._last_chop = step
                    self._comp_scores = {
                        k: v for k, v in self._comp_scores.items() if k not in killed
                    }
                    metrics["design/chopped"] = float(len(killed))
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

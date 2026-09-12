"""CMDP constraint-handling methods ("penalizers"), ported from ss2r.

Trimmed to CRPO and (PPO-)Lagrangian, the two penalizers currently wired up
for the PPO port. Ported from
safe-learning/ss2r/algorithms/penalizers.py.
"""

from typing import Any, NamedTuple, Optional, Protocol, TypeVar

import jax
import jax.numpy as jnp
import optax

Params = TypeVar("Params")


class Penalizer(Protocol):
    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: Params,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        ...


class CRPOParams(NamedTuple):
    burnin: int


class CRPO:
    def __init__(self, eta: float) -> None:
        self.eta = eta

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: CRPOParams,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], CRPOParams]:
        active = jnp.greater(constraint + self.eta, 0.0) | jnp.greater(params.burnin, 0)
        if rest is not None:
            loss_constraint = rest
        else:
            loss_constraint = constraint
        actor_loss = jnp.where(
            active,
            actor_loss,
            -loss_constraint,
        )
        # positional min: the a_min/a_max kwargs were removed in jax 0.10
        new_params = CRPOParams(jnp.clip(params.burnin - 1, -1))
        aux = {
            "crpo/burnin_counter": new_params.burnin,
            "crpo/active": active,
        }
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: Params
    ) -> tuple[dict[str, Any], Params]:
        return {}, params


class LagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    optimizer_state: optax.OptState


class Lagrangian:
    def __init__(self, multiplier_lr: float) -> None:
        self.optimizer = optax.adam(learning_rate=multiplier_lr)
        self.learning_rate = multiplier_lr

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: LagrangianParams,
        *,
        rest: Any,
    ) -> tuple[jax.Array, dict[str, Any], LagrangianParams]:
        cost_advantage = -rest
        lagrange_multiplier = params.lagrange_multiplier
        actor_loss += lagrange_multiplier * cost_advantage
        aux: dict[str, Any] = {}
        new_params = params
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: LagrangianParams
    ) -> tuple[jax.Array, LagrangianParams]:
        new_lagrange_multiplier = update_lagrange_multiplier(
            constraint, params.lagrange_multiplier, self.learning_rate
        )
        aux = {"lagrange_multiplier": new_lagrange_multiplier}
        return aux, LagrangianParams(new_lagrange_multiplier, params.optimizer_state)


def update_lagrange_multiplier(
    constraint: jax.Array, lagrange_multiplier: jax.Array, learning_rate: float
) -> jax.Array:
    new_multiplier = jnp.maximum(lagrange_multiplier - learning_rate * constraint, 0.0)
    return new_multiplier


class ScheduledPenaltyParams(NamedTuple):
    updates: jax.Array  # how many penalizer.update() calls have happened


class ScheduledPenalty:
    """Lever (c): a cost penalty whose STRENGTH IS A SCHEDULE, not a learned
    multiplier. Same loss form as Lagrangian -- `actor_loss += kappa *
    cost_advantage`, so the reward term is never removed -- but kappa follows a
    fixed geometric ramp in time instead of integrating the violation.

    WHY. Every constrained run before 2026-09-11 collapsed the same way: the
    constraint applied full pressure from step 0, while the policy could not
    yet walk, and the cost-minimal behaviour for a policy that cannot walk is
    to stay put. CRPO reached 1.2 m of an 11 m corridor at 50M; the Lagrangian
    arm that finally traversed did so only because its multiplier lr was
    dropped to 1e-5, i.e. because the pressure ramped slowly BY ACCIDENT. This
    makes the ramp explicit and controllable: near zero while locomotion is
    learned, rising to a cap once it can walk. It is the mechanism behind P3O
    (progressively increasing penalty), one of the two strongest baselines on
    CRAX, reduced to its essential part.

    GATED ON VIOLATION so the budget still means something: the penalty is
    applied only while `constraint < 0`. Without the gate this would be a plain
    penalty method with no notion of a budget at all. `constraint` arrives
    already normalised -- and, under --adaptive_budget_horizon, already
    corrected for episode length -- so the gate is on the same quantity the
    Lagrangian multiplier responds to.

    The schedule is geometric from kappa_init to kappa_max over `ramp_updates`
    calls to update(), then held. Geometric rather than linear because the
    useful range spans decades (0.01 -> 5) and the early part of the ramp is
    the part that matters -- linear would sit at ~0 for most of the run and
    then jump.

    SIZING kappa_max. The Lagrangian b50 arm reached lambda 4.16 at 50M with
    reward still ~19-21, so a cap near 4-5 is anchored on a multiplier the
    policy is known to tolerate on this task. Above ~6-11 (the b30 run) reward
    collapsed, though that was under the old budget semantics.
    """

    def __init__(self, kappa_init: float, kappa_max: float, ramp_updates: int):
        if not (0.0 < kappa_init <= kappa_max):
            raise ValueError(
                f"need 0 < kappa_init <= kappa_max, got {kappa_init}, {kappa_max}"
            )
        if ramp_updates < 1:
            raise ValueError(f"ramp_updates must be >= 1, got {ramp_updates}")
        self.kappa_init = float(kappa_init)
        self.kappa_max = float(kappa_max)
        self.ramp_updates = int(ramp_updates)

    def kappa(self, updates: jax.Array) -> jax.Array:
        frac = jnp.clip(updates.astype(jnp.float32) / self.ramp_updates, 0.0, 1.0)
        # geometric interpolation: kappa_init * (kappa_max/kappa_init) ** frac
        log_k = jnp.log(self.kappa_init) + frac * (
            jnp.log(self.kappa_max) - jnp.log(self.kappa_init)
        )
        return jnp.exp(log_k)

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: ScheduledPenaltyParams,
        *,
        rest: Any,
    ) -> tuple[jax.Array, dict[str, Any], ScheduledPenaltyParams]:
        cost_advantage = -rest
        kappa = self.kappa(params.updates)
        violating = (constraint < 0.0).astype(jnp.float32)
        actor_loss += kappa * violating * cost_advantage
        aux = {
            "scheduled/kappa": kappa,
            "scheduled/violating": violating,
        }
        return actor_loss, aux, params

    def update(
        self, constraint: jax.Array, params: ScheduledPenaltyParams
    ) -> tuple[dict[str, Any], ScheduledPenaltyParams]:
        return {}, ScheduledPenaltyParams(params.updates + 1)


def get_penalizer(
    name: Optional[str],
    *,
    eta: float = 0.0,
    burnin: int = 0,
    multiplier_lr: float = 7e-7,
    initial_lagrange_multiplier: float = 0.01,
    penalty_kappa_init: float = 0.01,
    penalty_kappa_max: float = 5.0,
    penalty_ramp_updates: int = 1,
) -> tuple[Optional[Penalizer], Optional[Params]]:
    """Build a penalizer and its initial state from simple keyword args.

    multiplier_lr/initial_lagrange_multiplier default to ss2r's own values
    (agent/penalizer/ppo_lagrangian.yaml), not empirically tuned for this
    repo's exact setup -- see train_ppo.py's --lagrangian_multiplier_lr help
    text for the caveat (never actually validated against go_to_goal by
    ss2r's own authors either, since their go_to_goal reference uses Saute).

    Unlike ss2r's Hydra-driven `get_penalizer`, this takes explicit
    keyword arguments so it can be used without a config framework.
    """
    if name is None:
        return None, None
    if name == "crpo":
        penalizer = CRPO(eta)
        penalizer_state = CRPOParams(burnin)
    elif name == "ppo_lagrangian":
        penalizer = Lagrangian(multiplier_lr)
        penalizer_state = LagrangianParams(
            jnp.asarray(initial_lagrange_multiplier),
            penalizer.optimizer.init(jnp.asarray(initial_lagrange_multiplier)),
        )
    elif name == "scheduled":
        penalizer = ScheduledPenalty(
            penalty_kappa_init, penalty_kappa_max, penalty_ramp_updates
        )
        penalizer_state = ScheduledPenaltyParams(jnp.asarray(0, dtype=jnp.int32))
    else:
        raise ValueError(f"Unknown penalizer {name!r}")
    return penalizer, penalizer_state

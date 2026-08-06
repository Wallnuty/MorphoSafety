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


def get_penalizer(
    name: Optional[str],
    *,
    eta: float = 0.0,
    burnin: int = 0,
    multiplier_lr: float = 7e-7,
    initial_lagrange_multiplier: float = 0.01,
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
    else:
        raise ValueError(f"Unknown penalizer {name!r}")
    return penalizer, penalizer_state

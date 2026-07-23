from typing import Any, Dict, Mapping, Union
from flax import struct
import jax
import mujoco.mjx as mjx

Observation = Union[jax.Array, Mapping[str, jax.Array]]


@struct.dataclass
class State:
    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
    def single_step(data, _):
        # jax.debug.print("Before replacing ctrl: {x}", x=data.ctrl)
        data = data.replace(ctrl=action)
        # jax.debug.print("Before replacing ctrl: {x}", x=data.ctrl)
        data = mjx.step(model, data)
        return data, None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]

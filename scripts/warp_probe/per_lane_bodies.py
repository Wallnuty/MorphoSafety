import os, sys, numpy as np, jax, jax.numpy as jnp
from mujoco import mjx
from mjx_safety_gym import morphology as M, backend
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
kw = dict(T.robot_env_kwargs("ant")); kw["morphology_conditioning"] = True
e = Minefield(**kw)
specs = [M.MorphologySpec.nominal(), M.MorphologySpec(np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.1]))]
batched, in_axes = M.build_batch(specs, model_builder=e.build_morphology_model)
mj0 = e.build_morphology_model(specs[0])
acts = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, (60, 8)), jnp.float32)
def roll(model):
    d = mjx.forward(model, backend.make_data(model, mj0))
    def body(d, a):
        d = mjx.step(model, d.replace(ctrl=a))
        return d, jnp.concatenate([d.qpos[:3], d.xpos[1]])
    return jax.lax.scan(body, d, acts)[1]
traj = np.asarray(jax.jit(jax.vmap(roll, in_axes=(in_axes,)))(batched))   # (lanes, T, 6)
np.save(sys.argv[1], traj)
print(backend.IMPL, "torso z at steps 10/30/60, lane0:", np.round(traj[0, [9, 29, 59], 2], 4), " lane1:", np.round(traj[1, [9, 29, 59], 2], 4))

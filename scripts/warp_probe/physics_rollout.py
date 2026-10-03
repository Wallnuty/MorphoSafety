import sys, numpy as np, jax, jax.numpy as jnp
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
from mujoco import mjx
kw = dict(T.robot_env_kwargs("ant"))
e = Minefield(**kw)
st = jax.jit(e.reset)(jax.random.PRNGKey(0))
d = st.data
acts = np.random.default_rng(0).uniform(-1, 1, (50, e.action_size)).astype(np.float32)
step = jax.jit(lambda d, a: mjx.step(e._mjx_model, d.replace(ctrl=a)))
out = {}
for i, a in enumerate(acts):
    d = step(d, a)
    if i + 1 in (1, 5, 50):
        out[f"qpos{i+1}"] = np.asarray(d.qpos); out[f"qvel{i+1}"] = np.asarray(d.qvel)
np.savez(sys.argv[1], **out)
print("saved", sys.argv[1], "nq", e._mj_model.nq, "opt", e._mj_model.opt.iterations, e._mj_model.opt.ls_iterations, e._mj_model.opt.integrator)

import os
import numpy as np, jax
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
kw = dict(T.robot_env_kwargs("ant")); kw.update(solver_iterations=100, solver_ls_iterations=50)
e = Minefield(**kw)
n = 1024
st = jax.jit(jax.vmap(e.reset))(jax.random.split(jax.random.PRNGKey(0), n))
step = jax.jit(jax.vmap(e.step))
imp = st.data._impl
print("counter fields:", [k for k in ("nacon", "nefc", "ncon", "ne", "nf", "nl") if hasattr(imp, k)])
peak = {}
for i in range(300):
    a = jax.random.uniform(jax.random.PRNGKey(i), (n, e.action_size), minval=-1, maxval=1)
    st = step(st, a)
    if i % 10 == 0:
        for k in ("nacon", "nefc"):
            if hasattr(st.data._impl, k):
                peak[k] = max(peak.get(k, 0), int(np.asarray(getattr(st.data._impl, k)).max()))
print("PEAK", peak, "naconmax", 64 * n, "njmax 256")

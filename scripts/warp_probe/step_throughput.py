import sys, time, numpy as np, jax, jax.numpy as jnp
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
it, ls = int(sys.argv[1]), int(sys.argv[2])
kw = dict(T.robot_env_kwargs("ant")); kw.update(solver_iterations=it, solver_ls_iterations=ls)
e = Minefield(**kw)
n = 1024
st = jax.jit(jax.vmap(e.reset))(jax.random.split(jax.random.PRNGKey(0), n))
@jax.jit
def run(st, key):
    def body(c, k):
        a = jax.random.uniform(k, (n, e.action_size), minval=-1, maxval=1)
        return jax.vmap(e.step)(c, a), None
    return jax.lax.scan(body, st, jax.random.split(key, 100))[0]
st = run(st, jax.random.PRNGKey(1)); jax.block_until_ready(st.obs)     # compile + warm up
t = time.time(); st = run(st, jax.random.PRNGKey(2)); jax.block_until_ready(st.obs); dt = time.time() - t
# one env.step = one decision = action_repeat physics steps inside our env? count env steps * n
print(f"solver {it}/{ls}: {100 * n / dt:,.0f} env-steps/s (random actions, {n} envs)")

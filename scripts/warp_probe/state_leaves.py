import jax, numpy as np
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
env = T.wrap_for_brax_training(Minefield(**dict(T.robot_env_kwargs("ant"))), episode_length=100, action_repeat=4)
keys = jax.random.split(jax.random.PRNGKey(0), 8).reshape(1, 8, 2)   # (devices, envs, 2) like the trainer
st = jax.jit(jax.vmap(env.reset))(keys)
bad = []
for path, leaf in jax.tree_util.tree_flatten_with_path(st)[0]:
    shp = getattr(leaf, "shape", None)
    if shp is None or len(shp) == 0 or shp[0] != 1:
        bad.append((jax.tree_util.keystr(path), shp))
print(len(bad), "leaves without a leading device axis of 1:")
for p, s in bad[:40]:
    print(" ", p, s)

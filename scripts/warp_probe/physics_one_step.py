import sys, numpy as np, jax
from mjx_safety_gym.envs.minefield import Minefield
from mjx_safety_gym.algorithms import train_ppo as T
from mujoco import mjx
mode, path = sys.argv[1], sys.argv[2]
e = Minefield(**dict(T.robot_env_kwargs("ant")))
m = e._mjx_model
st = jax.jit(e.reset)(jax.random.PRNGKey(0))
acts = np.random.default_rng(0).uniform(-1, 1, (60, e.action_size)).astype(np.float32)
step = jax.jit(lambda d, a: mjx.step(m, d.replace(ctrl=a)))
if mode == "record":           # old env: roll 60 steps, keep the states to start from
    d, qs, vs = st.data, [], []
    for a in acts:
        qs.append(np.asarray(d.qpos)); vs.append(np.asarray(d.qvel)); d = step(d, a)
    np.savez(path, qpos=np.array(qs), qvel=np.array(vs))
    sys.exit()
src = np.load(sys.argv[3])     # both envs: ONE step from each recorded state, cold warmstart
fwd = jax.jit(lambda d: mjx.forward(m, d))
out_q, out_v, ncon = [], [], []
for k in range(len(acts)):
    d = st.data.replace(qpos=src["qpos"][k], qvel=src["qvel"][k], qacc_warmstart=0 * st.data.qacc_warmstart)
    d = fwd(d)
    d2 = step(d, acts[k])
    out_q.append(np.asarray(d2.qpos)); out_v.append(np.asarray(d2.qvel))
np.savez(path, qpos=np.array(out_q), qvel=np.array(out_v))
print("done", mode)

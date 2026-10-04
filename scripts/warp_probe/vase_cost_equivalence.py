"""Does the sensor-based vase cost equal the old data.contact one, step for step?

The vase collision cost moved from reading `data.contact` (collision.py's
geoms_colliding, unavailable on MuJoCo Warp) to one contact sensor per vase
(world.add_vase_contact_sensors), 2026-10-04. This rolls the `run` task (10
vases) on the JAX backend, where both can be computed from the same step, with
the vases moved into a ring around each ant so its legs actually hit them, and
compares the two costs at every step. Then it checks Warp against JAX on the
IDENTICAL (deeply interpenetrating) start state, pair by pair: comparing whole
rollouts across backends is meaningless here, because the first step from such
overlaps is violent and the backends' box-capsule contact depths differ by
millimetres, which chaos amplifies at once.

Measured 2026-10-04: JAX 19,200 env-steps, 3,491 with contact, old and sensor
cost both 5,401, 0 differing steps; Warp vs JAX on the start state: 320 of
8,320 pairs touching on both, 0 disagreements, distances within 3.7 mm.

    python scripts/warp_probe/vase_cost_equivalence.py
"""
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

from mjx_safety_gym import backend


def start_state(impl: str, n_env: int = 64):
    """Reset n_env ants and move the 10 vases into a ring around each torso."""
    backend.configure(impl=impl, nworld=n_env)
    from mjx_safety_gym.algorithms import train_ppo as T
    from mjx_safety_gym.envs.run_forward import RunForward

    env = RunForward(**dict(T.robot_env_kwargs("ant")))
    m, mx = env._mj_model, env._mjx_model
    nvase = env.spec["vases"].num_objects
    st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), n_env))
    qpos = np.array(st.data.qpos)
    for i in range(nvase):                      # ring of vases around each torso
        adr = m.jnt_qposadr[m.body(f"vase_{i}").jntadr[0]]
        ang = 2 * np.pi * i / nvase
        r = 0.30 + 0.05 * (i % 3)
        qpos[:, adr:adr + 3] = qpos[:, 0:3] * [1, 1, 0] + [r * np.cos(ang), r * np.sin(ang), 0.1]
        qpos[:, adr + 3:adr + 7] = [1, 0, 0, 0]
    d = jax.vmap(lambda d: mjx.forward(mx, d))(st.data.replace(qpos=jp.asarray(qpos)))
    return env, d


def start_state_sensors(impl: str):
    env, d = start_state(impl)
    adr = env._vase_touch_adr
    return np.asarray(d.sensordata[:, adr]), np.asarray(d.sensordata[:, adr + 1])


def rollout(impl: str, n_env: int = 64, n_steps: int = 300, with_old: bool = True):
    env, d = start_state(impl, n_env)
    mx = env._mjx_model

    from mjx_safety_gym.collision import geoms_colliding
    vase_geoms = env._collision_obstacle_geoms_ids
    robot_geoms = env._robot_collision_geom_ids
    adr = jp.asarray(env._vase_touch_adr)

    def costs(d):
        new = jp.sum(jp.any((d.sensordata[adr] > 0) & (d.sensordata[adr + 1] < 0), axis=-1))
        if not with_old:
            return new, new
        old = jp.sum(jp.array([jp.any(jp.array([geoms_colliding(d, v, g) for g in robot_geoms]))
                               for v in vase_geoms]))
        return old, new

    @jax.jit
    def step(d, key):
        a = jax.random.uniform(key, (n_env, env.action_size), minval=-1, maxval=1)
        d = jax.vmap(lambda d, a: mjx.step(mx, d.replace(ctrl=a)))(d, a)
        return d, jax.vmap(costs)(d)

    olds, news = [], []
    for k in jax.random.split(jax.random.PRNGKey(1), n_steps):
        d, (o, n) = step(d, k)
        olds.append(np.asarray(o)); news.append(np.asarray(n))
    return np.array(olds), np.array(news)


if __name__ == "__main__":
    old, new = rollout("jax")
    touching = int((old > 0).sum())
    print(f"JAX: {old.size} env-steps, {touching} with a robot-vase contact "
          f"(old cost total {int(old.sum())}, sensor cost total {int(new.sum())}); "
          f"steps where they differ: {int((old != new).sum())}")
    if jax.default_backend() == "gpu":
        j, w = start_state_sensors("jax"), start_state_sensors("warp")
        hj = (j[0] > 0) & (j[1] < 0)
        hw = (w[0] > 0) & (w[1] < 0)
        both = (j[0] > 0) & (w[0] > 0)
        print(f"Warp vs JAX on the identical start state: {int(hj.sum())} / {int(hw.sum())} of "
              f"{hj.size} (env, vase, robot geom) pairs touching, {int((hj != hw).sum())} disagree; "
              f"max distance difference {float(np.abs(j[1][both] - w[1][both]).max()):.4f} m")

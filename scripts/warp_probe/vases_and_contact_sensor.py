import os, numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
impl = os.environ.get("MJX_IMPL", "jax")
# (1) our env with vases: does the contact-based vase cost run on this backend?
try:
    from mjx_safety_gym.envs.run_forward import RunForward
    from mjx_safety_gym.algorithms import train_ppo as T
    e = RunForward(**{k: v for k, v in T.robot_env_kwargs("ant").items()})
    print(impl, "RunForward vases:", e.spec["vases"].num_objects, "| data.contact type:",
          type(jax.jit(e.reset)(jax.random.PRNGKey(0)).data.contact).__name__)
    st = jax.jit(jax.vmap(e.reset))(jax.random.split(jax.random.PRNGKey(0), 4))
    st = jax.jit(jax.vmap(e.step))(st, jnp.zeros((4, e.action_size)))
    print(impl, "  env step with vases OK; cost", np.asarray(st.info.get("cost", st.reward))[:2])
except Exception as ex:
    print(impl, "  env with vases FAILED:", type(ex).__name__, str(ex).splitlines()[0][:160])
# (2) a contact SENSOR on a toy scene: capsule dropped onto a box
xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
<body name="vase" pos="0 0 .1"><freejoint/><geom name="vase" type="box" size=".1 .1 .1"/></body>
<body name="foot" pos="0 0 .35"><freejoint/><geom name="foot" type="capsule" size=".03 .05"/></body>
</worldbody><sensor><contact name="touch" geom1="foot" geom2="vase" data="found"/></sensor></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
try:
    mx = mjx.put_model(m, impl=impl)
    d = mjx.make_data(m if impl == "warp" else mx, impl=impl, **({"naconmax": 64, "njmax": 64} if impl == "warp" else {}))
    step = jax.jit(lambda d: mjx.step(mx, d))
    found = []
    for i in range(400):
        d = step(d)
        if i % 50 == 0: found.append(float(np.asarray(d.sensordata)[0]))
    # CPU MuJoCo as the reference
    dc = mujoco.MjData(m); ref = []
    for i in range(400):
        mujoco.mj_step(m, dc)
        if i % 50 == 0: ref.append(float(dc.sensordata[0]))
    print(impl, "  contact sensor 'found' every 50 steps:", found, "| CPU MuJoCo:", ref)
except Exception as ex:
    print(impl, "  contact sensor FAILED:", type(ex).__name__, str(ex).splitlines()[0][:200])

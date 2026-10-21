"""GPU stepping harness, shared by the MorphoSafety and mjx-safety-gym 0.1.6
benchmarks so that the two are timed by identical code (2026-10-19).

Pure JAX, no package imports: bench_mjxsg.py runs in a separate environment
whose package is ALSO called mjx_safety_gym.

What is timed: `n` environments stepped with uniform random actions in
[-1, 1], `steps_per_call` steps per jitted call (a lax.scan), every lane reset
to its initial state every `episode` steps (a cached-state select, which is
what brax's auto-reset does). pmap over one device with vmap inside, as the
trainer runs it (MuJoCo Warp needs pmap there). Compile and the first call are
excluded; then calls repeat until `min_time` seconds and `min_calls` calls.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def measure(reset_fn, step_fn, action_size, n, steps_per_call=100, episode=1000,
            min_time=10.0, min_calls=3, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)[None]
    t0 = time.time()
    state = jax.pmap(jax.vmap(reset_fn))(keys)
    jax.block_until_ready(state)
    first = state

    def run(state, first, t, key):
        def body(carry, k):
            s, t = carry
            a = jax.random.uniform(k, (n, action_size), minval=-1.0, maxval=1.0)
            s = jax.vmap(step_fn)(s, a)
            t = t + 1
            done = (t % episode) == 0
            s = jax.tree_util.tree_map(lambda f, x: jnp.where(done, f, x), first, s)
            return (s, t), None

        (state, t), _ = jax.lax.scan(body, (state, t), jax.random.split(key, steps_per_call))
        return state, t

    run = jax.pmap(run)
    t = jnp.zeros((1,), jnp.int32)
    key = jax.random.PRNGKey(seed + 1)
    key, k = jax.random.split(key)
    state, t = run(state, first, t, k[None])
    jax.block_until_ready(state)
    compile_s = time.time() - t0
    calls, t1 = 0, time.time()
    while calls < min_calls or time.time() - t1 < min_time:
        key, k = jax.random.split(key)
        state, t = run(state, first, t, k[None])
        jax.block_until_ready(state)
        calls += 1
    elapsed = time.time() - t1
    return {
        "n": n,
        "steps_per_s": calls * steps_per_call * n / elapsed,
        "compile_s": compile_s,
        "calls": calls,
        "elapsed_s": elapsed,
        "steps_per_call": steps_per_call,
    }


def device_info() -> dict:
    d = jax.devices()[0]
    return {"jax": jax.__version__, "device": str(getattr(d, "device_kind", d))}

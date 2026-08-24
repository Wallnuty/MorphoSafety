"""Verify design-state checkpointing and the gene-block normalizer.

Neither fix is observable from a training log: a resume that silently restarts
the design search still prints plausible `design/*` metrics, and the
normalizer branch is dead until someone turns `normalize_observations` on. So
both are checked here directly, and each check is MUTATION-TESTED -- a round
trip that quietly drops a field passes a naive "did it load?" assertion.

Runs on CPU in a couple of seconds; no physics, no GPU.

    python scripts/verify_design_checkpoint.py
"""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from mjx_safety_gym import design as D

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def make_loop(n_components=8, n_params=7, seed=3):
    gmm = D.GmmDesignDistribution(
        n_params=n_params, n_components=n_components, std_init=0.577,
        lr=1e-3, momentum=0.9, seed=seed,
    )
    return D.DesignLoop(
        gmm=gmm, spec_factory=lambda g: g, batch_builder=lambda s, r: (None, None, None),
        wrapper=None, num_morphologies=4, num_envs=32, tmax=1_000_000,
        chop_freq=1, objective="return",
    )


def exercise(loop, rng_draws=3):
    """Drive the loop far enough that every mutable field is non-default."""
    gmm = loop.gmm
    for it in range(4):
        params, comps = gmm.sample(6)
        scores = np.linspace(-1.0, 1.0, 6) + 0.1 * it
        gmm.update(params, comps, scores, lr_frac=0.8)
        for c, s in zip(comps, scores):
            prev = loop._comp_scores.get(int(c))
            loop._comp_scores[int(c)] = (
                float(s) if prev is None else 0.9 * prev + 0.1 * float(s)
            )
    gmm.chop(dict(loop._comp_scores))          # kills half; resets both Adams
    for it in range(2):                        # re-dirty the Adam moments
        params, comps = gmm.sample(6)
        gmm.update(params, comps, np.linspace(-1, 1, 6), lr_frac=0.5)
    loop._last_chop, loop._last_step = 123_456, 234_567
    for _ in range(rng_draws):                 # advance the sampling stream
        gmm.sample(2)


print("=== 1. full round trip ===")
orig = make_loop()
exercise(orig)
with tempfile.TemporaryDirectory() as d:
    orig.save(d)
    restored = make_loop(seed=999)             # DIFFERENT seed on purpose:
    assert not np.allclose(restored.gmm.means, orig.gmm.means)  # nothing shared
    check("load() found the sidecar", restored.load(d))

    g0, g1 = orig.gmm, restored.gmm
    check("means", np.array_equal(g0.means, g1.means))
    check("log_stds", np.array_equal(g0.log_stds, g1.log_stds))
    check("log_mixprobs (chop flags)", np.array_equal(g0.log_mixprobs, g1.log_mixprobs),
          f"{g1.components_left()} of {g1.n_components} live")
    check("adam_mean m/v/t",
          np.array_equal(g0._adam_mean.m, g1._adam_mean.m)
          and np.array_equal(g0._adam_mean.v, g1._adam_mean.v)
          and g0._adam_mean.t == g1._adam_mean.t, f"t={g1._adam_mean.t}")
    check("adam_logstd m/v/t",
          np.array_equal(g0._adam_logstd.m, g1._adam_logstd.m)
          and np.array_equal(g0._adam_logstd.v, g1._adam_logstd.v)
          and g0._adam_logstd.t == g1._adam_logstd.t)
    check("component score EMA", orig._comp_scores == restored._comp_scores,
          f"{len(restored._comp_scores)} components scored")
    check("last_chop / last_step",
          (orig._last_chop, orig._last_step) == (restored._last_chop, restored._last_step))

    # The ONLY real proof the 128-bit word encoding is right: keep drawing.
    a = [orig.gmm.sample(3) for _ in range(4)]
    b = [restored.gmm.sample(3) for _ in range(4)]
    check("RNG stream continues identically",
          all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
              for x, y in zip(a, b)),
          "4 further draws")

print("=== 2. the round trip actually distinguishes a dropped field ===")
# Without this, every check above would also pass on a state_dict that silently
# omitted the Adam moments -- the fields it does copy would still match.
with tempfile.TemporaryDirectory() as d:
    orig2 = make_loop(); exercise(orig2); orig2.save(d)
    import pathlib
    path = pathlib.Path(d) / D.DesignLoop.SIDECAR
    with np.load(path) as f:
        kept = {k: f[k] for k in f.files if k != "gmm_adam_mean_m"}
    np.savez(path, **kept)
    maimed = make_loop(seed=999)
    maimed.load(d)                              # warns, does not crash
    check("dropped adam_mean_m is detected, not silently accepted",
          not np.array_equal(orig2.gmm._adam_mean.m, maimed.gmm._adam_mean.m)
          and np.allclose(maimed.gmm._adam_mean.m, 0.0))
    check("...and the rest still restored", np.array_equal(orig2.gmm.means, maimed.gmm.means))

print("=== 3. a config mismatch is fatal, not silently reshaped ===")
with tempfile.TemporaryDirectory() as d:
    make_loop(n_components=8).save(d)
    try:
        make_loop(n_components=4).load(d)
        check("8-component state into a 4-component run raises", False, "it did not")
    except ValueError as exc:
        check("8-component state into a 4-component run raises", True, str(exc)[:60])
    try:
        make_loop(n_params=5).load(d)
        check("7-gene state into a 5-gene run raises", False, "it did not")
    except ValueError as exc:
        check("7-gene state into a 5-gene run raises", True, str(exc)[:60])

print("=== 4. missing sidecar reports False, does not crash ===")
with tempfile.TemporaryDirectory() as d:
    check("load() on a bare directory returns False", make_loop().load(d) is False)
try:
    make_loop().save("/nonexistent/checkpoint/dir")
    check("save() into a missing directory raises", False, "it did not")
except FileNotFoundError:
    check("save() into a missing directory raises", True)

print("=== 5. _last_step survives, so the frozen phase survives a resume ===")
# `sample()` used to read the step off in-memory `history`, which a resume
# leaves empty -- so the first post-resume iteration of a run in its final
# fine-tune phase would sample from the distribution instead of the mode.
with tempfile.TemporaryDirectory() as d:
    late = make_loop()
    late.steps_after_update = 100_000
    late._last_step = late.tmax - 10          # deep in the frozen phase
    late.save(d)
    fresh = make_loop(); fresh.steps_after_update = 100_000
    check("a fresh loop is NOT frozen", not fresh._frozen(fresh._last_step))
    fresh.load(d)
    check("a restored loop IS frozen", fresh._frozen(fresh._last_step))

print("=== 6. the gene block is normalized correctly with and without Saute ===")
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from mjx_safety_gym.algorithms.ppo import train as ppo_train

NG, WIDTH = 7, 20
spec = specs.Array((WIDTH,), jnp.float32)
stats = running_statistics.init_state(spec)
batch = np.random.default_rng(0).normal(size=(256, WIDTH)) * 5.0 + 3.0
stats = running_statistics.update(stats, jnp.asarray(batch, dtype=jnp.float32))
x = jnp.asarray(batch[:4], dtype=jnp.float32)

for label, offset, gene_lo, gene_hi in (
    ("no saute (genes last)", 0, WIDTH - NG, WIDTH),
    ("saute (genes then budget)", 1, WIDTH - NG - 1, WIDTH - 1),
):
    fn = ppo_train._passthrough_block_normalizer(NG, offset)
    out = np.asarray(fn(x, stats))
    ref = np.asarray(running_statistics.normalize(x, stats))
    raw = np.asarray(x)
    passed = np.isclose(out, raw, atol=1e-5) & ~np.isclose(ref, raw, atol=1e-5)
    idx = sorted(set(np.flatnonzero(passed.any(axis=0)).tolist()))
    check(f"{label}: exactly dims {gene_lo}-{gene_hi - 1} pass through raw",
          idx == list(range(gene_lo, gene_hi)), f"got {idx}")
    check(f"{label}: shape preserved", out.shape == raw.shape)
    others = [i for i in range(WIDTH) if not (gene_lo <= i < gene_hi)]
    check(f"{label}: every other dim is normalized",
          np.allclose(out[:, others], ref[:, others], atol=1e-6))

print("=== 7. the chop schedule survives a resume (trainer step restarts at 0) ===")
# THE REGRESSION THIS GUARDS. `ppo/train.py` does not restore `env_steps`, so
# `current_step` counts from 0 again after a resume. Restoring `_last_chop` as
# a raw absolute step then makes `step - _last_chop` negative for the whole
# resumed run and the mixture NEVER CHOPS AGAIN -- it never commits to a
# design, silently, with every metric still looking healthy.


class FakeState:
    def __init__(self, n):
        self.info = {
            "ep_count": np.ones(n),
            "ep_return_last": np.linspace(0.0, 1.0, n),
        }


def drive(loop, trainer_step):
    loop._pending = loop.gmm.sample(loop.num_morphologies)
    return loop.finish_iteration(FakeState(loop.num_envs), trainer_step)


CHOP = 1000
with tempfile.TemporaryDirectory() as d:
    pre = make_loop()
    pre.chop_freq = CHOP
    # ONE chop, then two iterations too close together to trigger another --
    # so the mixture still has components to lose and `_last_chop` sits well
    # behind `_last_step`, which is the situation a crash actually leaves.
    for s in (CHOP, 1500, 1800):
        drive(pre, s)
    check("components were chopped before the crash",
          pre.gmm.components_left() < pre.gmm.n_components,
          f"{pre.gmm.components_left()} of {pre.gmm.n_components} live, "
          f"last chop at {pre._last_chop}")
    pre.save(d)

    post = make_loop(); post.chop_freq = CHOP
    post.load(d)
    check("design axis resumed at the saved step",
          post._step_offset == pre._last_step, f"offset={post._step_offset}")
    before = post.gmm.components_left()
    drive(post, 200)                          # trainer step 200, axis 2000
    check("a chop can still fire after the resume",
          post.gmm.components_left() < before,
          f"{before} -> {post.gmm.components_left()} live")

    # Mutation: zero the offset, i.e. the pre-fix behaviour, and confirm the
    # check above would have caught it.
    naive = make_loop(); naive.chop_freq = CHOP
    naive.load(d)
    naive._step_offset = 0
    before = naive.gmm.components_left()
    for s in (200, 400, 600, 800):            # a short resume, as after a crash
        drive(naive, s)
    check("...and WITHOUT the offset a short resume never chops (mutation)",
          naive.gmm.components_left() == before,
          f"stuck at {before} live; would need trainer step "
          f">= {pre._last_chop + CHOP}")

print()
if FAILURES:
    raise SystemExit(f"FAILED: {FAILURES}")
print("all checks passed")

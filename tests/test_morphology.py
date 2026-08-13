"""Batched-morphology models: the substrate the whole evolutionary search sits on.

The design premise of this project's morphology search is that MJX can step
many DIFFERENT bodies inside one batched `mjx.step`, which turns the cost of
evaluating a population from "one PPO run per candidate" (~1000 runs) into
"one batch of rollouts". That premise is only true if the batched model is
EXACT -- if any morphology-varying field is left unbatched, every lane
silently shares the base model's value for it and the whole population becomes
partially identical, with no error anywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

from mjx_safety_gym import morphology
from mjx_safety_gym.morphology import (
    NUM_GENES,
    MorphologySpec,
    batch_models,
    build_mj_model,
    randomization_fn,
)


@pytest.mark.parametrize("robot", ["ant", "ant_gym"])
def test_randomization_fn_runs(robot):
    """Regression guard: this raised NameError for every caller.

    `build_batch` referenced `integrator` and `robot` as if they were
    parameters when they were not, so ANY use of --num_morphologies died with
    `NameError: name 'integrator' is not defined`. It was introduced when
    those two arguments were threaded into `build_mj_model` for ant_gym and
    `build_batch` was not updated -- i.e. the batched-morphology path, the
    core of the research plan, was dead and nothing said so, because nothing
    exercised it.
    """
    batched, in_axes, genes = randomization_fn(
        None, jax.random.PRNGKey(0), num_morphologies=2, num_envs=4, robot=robot
    )
    assert genes.shape == (4, NUM_GENES)
    del batched, in_axes


def test_each_env_gets_its_morphologys_genes():
    """`num_envs // num_morphologies` replicas of each body, genes aligned.

    The policy is morphology-CONDITIONED: genes are appended to the
    observation, so a lane whose genes do not describe its body trains the
    network on a lie. Misalignment here is invisible -- training runs fine and
    just learns nothing useful from the conditioning.
    """
    n_morph, n_envs = 2, 6
    batched, _, genes = randomization_fn(
        None, jax.random.PRNGKey(1), num_morphologies=n_morph, num_envs=n_envs
    )
    genes = np.asarray(genes)
    replicas = n_envs // n_morph

    for m in range(n_morph):
        block = genes[m * replicas : (m + 1) * replicas]
        np.testing.assert_array_equal(
            block, np.repeat(block[:1], replicas, axis=0),
            err_msg="replicas of one morphology have different genes",
        )

    assert not np.allclose(genes[0], genes[replicas]), (
        "distinct morphologies produced identical genes -- sampling is not varying"
    )

    # And the bodies really differ, not just the gene vectors describing them.
    masses = np.asarray(batched.body_mass).sum(axis=1)
    assert masses[0] != pytest.approx(masses[replicas]), (
        "distinct genes but identical body mass -- the model is not being rebuilt"
    )


def test_batched_fields_covers_exactly_what_varies_with_morphology():
    """The real correctness proof for `_BATCHED_FIELDS`, and it is exact.

    If a morphology-varying field is missing from that list, every lane
    silently inherits the BASE model's value for it: the batch stops
    describing the population, the search ranks bodies it is not actually
    simulating, and nothing raises. So enumerate what genuinely differs
    between independently compiled models and require the two lists to
    account for all of it -- no floating-point comparison, no thresholds.

    Both directions matter. Unaccounted-for fields are the silent-corruption
    bug above; declared-but-constant fields mean the batch is carrying
    per-lane copies of identical data, which is pure memory and a sign the
    list has drifted from the model.

    (An earlier version of this test compared batched physics against
    individually compiled physics and asserted agreement at ~1e-8. That was
    the wrong instrument: batching THREE IDENTICAL MODELS already disagrees
    with the unbatched result by 8.08e-07, because vmap changes the float
    association XLA picks. See the test below.)
    """
    rng = np.random.default_rng(0)
    models = [
        mjx.put_model(build_mj_model(MorphologySpec.sample(rng))) for _ in range(4)
    ]

    varying = []
    for field in dir(models[0]):
        if field.startswith("_"):
            continue
        try:
            values = [getattr(m, field) for m in models]
        except Exception:
            continue
        if not all(hasattr(v, "shape") for v in values):
            continue
        if any(v.shape != values[0].shape for v in values) or any(
            not np.array_equal(np.asarray(v), np.asarray(values[0])) for v in values
        ):
            varying.append(field)

    declared = set(morphology._BATCHED_FIELDS) | set(morphology._STATIC_PINNED_FIELDS)
    missing = sorted(set(varying) - declared)
    spurious = sorted(declared - set(varying))

    assert not missing, (
        f"fields vary with morphology but are neither batched nor pinned: "
        f"{missing}. Every lane silently shares the base model's value for "
        "these, so the batch does not describe the population."
    )
    assert not spurious, (
        f"declared in _BATCHED_FIELDS/_STATIC_PINNED_FIELDS but identical "
        f"across morphologies: {spurious}"
    )


@pytest.mark.slow
def test_batched_physics_tracks_individually_compiled_physics():
    """Sanity check on the batched model, with an HONESTLY calibrated bound.

    Batching cannot be expected to reproduce unbatched physics bitwise: vmap
    changes which reduction order XLA emits. Measured on this machine, three
    IDENTICAL models batched together already differ from their unbatched
    counterparts by 8.08e-07 at step 1 -- so that, not float32 eps, is the
    floor any threshold here has to clear.

    Checked at STEP 1 on purpose. Contact dynamics amplify float32 roundoff by
    ~5 orders of magnitude within 20 steps (1.9e-08 -> 9.1e-04 measured), so a
    longer-horizon comparison would fail for reasons that are not bugs. That
    same amplification is why nothing anywhere may assume bitwise
    reproducibility across batch compositions: a morphology's fitness is
    reproducible only with the seed list AND the batch layout held fixed.

    Field completeness is proven exactly by the test above; this one only
    catches gross breakage in the stacking itself.
    """
    rng = np.random.default_rng(0)
    mj_models = [build_mj_model(MorphologySpec.sample(rng)) for _ in range(3)]

    batched, in_axes = batch_models(mj_models)
    singles = [mjx.put_model(m) for m in mj_models]
    ctrl = jp.tile(jp.linspace(-0.3, 0.3, mj_models[0].nu)[None, :], (len(mj_models), 1))

    def one_step(model, control):
        return mjx.step(model, mjx.make_data(model).replace(ctrl=control))

    batched_data = jax.jit(jax.vmap(one_step, in_axes=(in_axes, 0)))(batched, ctrl)
    err = max(
        float(
            np.max(
                np.abs(
                    np.asarray(batched_data.qpos)[i]
                    - np.asarray(jax.jit(one_step)(m, ctrl[i]).qpos)
                )
            )
        )
        for i, m in enumerate(singles)
    )
    assert err < 1e-4, (
        f"batched vs individually-compiled physics disagree by {err:.2e} after "
        "one step -- two orders above the measured vmap association noise "
        "(8e-07), so the stacking itself is wrong."
    )


def test_static_pinned_fields_are_not_in_batched_fields():
    """The two lists must be disjoint, and both are load-bearing.

    `geom_aabb` and `geom_rbound_hfield` live in the pytree TREEDEF rather
    than its leaves, so they cannot be batched at all -- stacking
    independently compiled models raises a tree-prefix metadata error. They
    are pinned on the base model instead, which is only safe because neither
    is read in the MJX physics path for a `plane` floor.
    """
    assert not (
        set(morphology._BATCHED_FIELDS) & set(morphology._STATIC_PINNED_FIELDS)
    )


def test_total_mass_measures_the_robot_not_the_arena():
    """`total_mass` summed the WHOLE model, arena included.

    Invisible at 42 kg robot / 0.95 kg arena, fatal at ant_gym's 0.911 kg
    robot against a 35.7 kg scaled arena -- 97% obstacles. The mass band that
    bounds the search would then have gated on hazard geometry rather than on
    the morphology, rejecting every sampled body and producing a silently
    empty search.
    """
    spec = MorphologySpec.sample(np.random.default_rng(0))
    with_arena = build_mj_model(spec, robot="ant_gym")
    mass = morphology.total_mass(with_arena)
    assert mass < 10.0, (
        f"total_mass {mass:.1f} kg for one ant_gym morphology -- that is the "
        "arena being counted, not the robot"
    )

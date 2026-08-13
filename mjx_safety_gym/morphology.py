"""Parametric ant morphology: build, batch, and bound scaled ant bodies.

Ported from exploratory verification done for the morphology-optimization
plan, not derived here. Three things were checked before this module was
written and are relied on rather than re-justified inline:

  - Exactly 13 `mjx.Model` fields vary with limb scaling (`_BATCHED_FIELDS`
    below); 2 more are static aux data that must be pinned to a shared value
    before batching (`_STATIC_PINNED_FIELDS`), because they live in the pytree
    treedef rather than its leaves.
  - Batched-vs-individually-compiled physics agree to 1.85e-08 after one
    `mjx.step` (float32 roundoff), confirming that field list is complete.
  - Structural fields (`ngeom`, `nbody`, `nq`, `nv`, `nu`, `njnt`, `geom_type`,
    ...) are identical across every morphology this module can generate,
    because only capsule sizes/positions are scaled -- never added or removed.

See the project plan for the full derivation and numbers.
"""

from __future__ import annotations

import dataclasses
from importlib.resources import files
from typing import Sequence

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
from mujoco import mjx

from mjx_safety_gym.world import ObjectSpec, apply_integrator, build_arena

_XML_DIR = files("mjx_safety_gym.envs.xmls")
_ANT_XML = _XML_DIR / "ant.xml"

# Per-robot facts this module needs, deliberately DUPLICATED from
# _ROBOT_CONFIGS in envs/go_to_goal.py rather than imported: go_to_goal imports
# NUM_GENES from here, so importing it back would be a cycle. Keep the two in
# step -- if a robot's arena_scale or vase_mass changes there, change it here.
#
# mass_band is this module's own, and is NOT in _ROBOT_CONFIGS: it bounds which
# sampled morphologies are worth evaluating at all. The ant band (15-90 kg
# around a 42.255 kg nominal) is 0.36x-2.13x nominal; ant_gym's is the same
# ratio band around its 0.911 kg nominal. Getting this wrong is silent and
# total -- an ant_gym run against the ant band would reject EVERY sampled
# morphology, since none of them come close to 15 kg.
_MORPH_ROBOTS: dict[str, dict] = {
    "ant": {
        "xml": "ant.xml",
        "arena_scale": 1.0,
        "vase_mass": None,
        "mass_band": (15.0, 90.0),
    },
    "ant_gym": {
        "xml": "ant_gym.xml",
        "arena_scale": 4.0,
        "vase_mass": 0.05,
        "mass_band": (0.3, 2.0),
    },
}


def _arena_spec_for(robot: str) -> dict[str, ObjectSpec]:
    """Placement keepouts, scaled to the robot -- mirrors GoToGoal.__init__."""
    a = _MORPH_ROBOTS[robot]["arena_scale"]
    return {
        "robot": ObjectSpec(0.4 * a, 1),
        "goal": ObjectSpec(0.305 * a, 1),
        "hazards": ObjectSpec(0.18 * a, 10),
        "vases": ObjectSpec(0.15 * a, 10),
    }


_DEFAULT_ARENA_SPEC: dict[str, ObjectSpec] = _arena_spec_for("ant")

# Each of the ant's 3 leg segments (aux -> leg -> ankle) is 4-fold symmetric
# across the 4 legs, so one length + one radius gene per segment covers the
# whole leg. Ankle geoms are leaves (nothing is parented past their tip), so
# they have no entry in _SEGMENT_CHILD_BODIES.
_SEGMENT_GEOMS: dict[str, tuple[str, ...]] = {
    "aux": ("aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"),
    "leg": ("left_leg_geom", "right_leg_geom", "back_leg_geom", "rightback_leg_geom"),
    "ankle": (
        "left_ankle_geom",
        "right_ankle_geom",
        "third_ankle_geom",
        "fourth_ankle_geom",
    ),
}
_SEGMENT_CHILD_BODIES: dict[str, tuple[str, ...]] = {
    "aux": ("aux_1", "aux_2", "aux_3", "aux_4"),
    "leg": ("front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"),
}

GENE_NAMES: tuple[str, ...] = (
    "aux_len",
    "leg_len",
    "ankle_len",
    "aux_rad",
    "leg_rad",
    "ankle_rad",
    "torso_rad",
)
NUM_GENES = len(GENE_NAMES)

# Scale-factor bounds applied to the nominal ant.xml geometry, symmetric
# around 1.0 so gene == 0.5 reproduces the unmodified ant exactly (useful as a
# regression anchor). Kept inside the range exercised during verification
# (0.6-1.6 tested batched-vs-individual): pushing wider is unverified and mass
# was already observed to swing ~13x across a similar band.
SCALE_LO = 0.6
SCALE_HI = 1.4

# Reject specs outside a band around the robot's nominal mass: too light barely
# touches the floor under gravity, too heavy is dead weight against a fixed
# actuator_gear=150 that cannot move it -- either way, a wasted fitness
# evaluation. Per-robot bands live in _MORPH_ROBOTS; this is the "ant" one,
# kept as a module constant for backwards compatibility with existing callers.
MASS_BAND: tuple[float, float] = (15.0, 90.0)

_BATCHED_FIELDS: tuple[str, ...] = (
    "body_pos",
    "body_ipos",
    "body_mass",
    "body_subtreemass",
    "body_inertia",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "geom_size",
    "geom_rbound",
    "geom_pos",
    "actuator_acc0",
    "light_poscom0",
)
_STATIC_PINNED_FIELDS: tuple[str, ...] = ("geom_aabb", "geom_rbound_hfield")


@dataclasses.dataclass(frozen=True, eq=False)
class MorphologySpec:
    """A single ant morphology as 7 normalized genes in [0, 1].

    Denormalized via `scales`: each gene maps linearly onto
    [SCALE_LO, SCALE_HI] and multiplies the corresponding nominal ant.xml
    dimension. `eq=False` because the default dataclass __eq__ would compare
    the genes array with `==` and return an array, not a bool.
    """

    genes: np.ndarray  # (NUM_GENES,), values in [0, 1]

    def __post_init__(self) -> None:
        genes = np.clip(np.asarray(self.genes, dtype=np.float64), 0.0, 1.0)
        if genes.shape != (NUM_GENES,):
            raise ValueError(f"expected shape ({NUM_GENES},), got {genes.shape}")
        object.__setattr__(self, "genes", genes)

    @classmethod
    def nominal(cls) -> "MorphologySpec":
        return cls(genes=np.full(NUM_GENES, 0.5))

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "MorphologySpec":
        return cls(genes=rng.uniform(0.0, 1.0, size=NUM_GENES))

    @property
    def scales(self) -> dict[str, float]:
        s = SCALE_LO + self.genes * (SCALE_HI - SCALE_LO)
        return dict(zip(GENE_NAMES, s.tolist()))


def build_mj_model(
    spec: MorphologySpec,
    arena_spec: dict[str, ObjectSpec] | None = None,
    integrator: str | None = None,
    robot: str = "ant",
) -> mj.MjModel:
    """Compile a single ant with the given morphology, arena included.

    Mirrors GoToGoal.__init__'s own compile path (envs/go_to_goal.py) so a
    model built here has identical topology to (and is thus batchable with)
    what the env normally produces. `integrator` must therefore be threaded
    through to match whatever the env was built with -- under morphology
    randomization these models REPLACE the env's `_mjx_model`, so if only
    GoToGoal honoured the override the randomized run would silently step
    different physics from the nominal one.
    """
    cfg = _MORPH_ROBOTS[robot]
    if arena_spec is None:
        arena_spec = _arena_spec_for(robot)
    scales = spec.scales

    s = mj.MjSpec.from_file(str(_XML_DIR / cfg["xml"]))
    apply_integrator(s, integrator)
    geoms = {g.name: g for g in s.geoms}
    bodies = {b.name: b for b in s.bodies}

    for segment, geom_names in _SEGMENT_GEOMS.items():
        len_scale = scales[f"{segment}_len"]
        rad_scale = scales[f"{segment}_rad"]
        for name in geom_names:
            g = geoms[name]
            fromto = np.array(g.fromto)
            g.fromto = np.concatenate([fromto[:3], fromto[3:] * len_scale])
            g.size = np.array([g.size[0] * rad_scale, 0.0, 0.0])
        for child_name in _SEGMENT_CHILD_BODIES.get(segment, ()):
            b = bodies[child_name]
            b.pos = np.array(b.pos) * len_scale

    torso = geoms["torso_geom"]
    torso.size = np.array([torso.size[0] * scales["torso_rad"], 0.0, 0.0])

    build_arena(
        s,
        objects=arena_spec,
        visualize=True,
        obstacle_scale=cfg["arena_scale"],
        vase_mass=cfg["vase_mass"],
    )
    return s.compile()


def total_mass(mj_model: mj.MjModel) -> float:
    """Mass of the ROBOT, not of the whole compiled model.

    `body_mass.sum()` would also count the arena -- ten vases, ten hazards and
    the goal cylinder are all worldbody children with real geom-derived mass.
    That was invisible while the ant weighed 42 kg against a 0.95 kg arena (2%),
    but it is fatal at other scales: ant_gym's robot is 0.911 kg against a
    ~35.7 kg scaled arena, so the whole-model sum is 97% obstacles and the mass
    band below would be gating on hazard geometry rather than on the morphology
    being evaluated. Reading the robot body's subtree mass counts the torso and
    all four legs and nothing else.
    """
    return float(mj_model.body_subtreemass[mj_model.body("robot").id])


def mass_in_band(
    mj_model: mj.MjModel,
    band: tuple[float, float] | None = None,
    robot: str = "ant",
) -> bool:
    """Is this morphology worth evaluating? Band is per-robot; see _MORPH_ROBOTS."""
    if band is None:
        band = _MORPH_ROBOTS[robot]["mass_band"]
    m = total_mass(mj_model)
    return band[0] <= m <= band[1]


def batch_models(mj_models: Sequence[mj.MjModel]) -> tuple[mjx.Model, mjx.Model]:
    """Stack independently-compiled ant models into one batched `mjx.Model`.

    Returns `(batched_model, in_axes)` -- `in_axes` is the pytree
    `jax.vmap`/brax's `DomainRandomizationVmapWrapper` expect: `None`
    everywhere except the 13 fields that actually vary with morphology, which
    get axis 0.

    All `mj_models` must share topology (ngeom, nbody, nq, ...) -- true for
    anything from `build_mj_model` in this module, since only capsule sizes
    and positions are ever scaled, never added or removed.
    """
    mjx_models = [mjx.put_model(m) for m in mj_models]
    base = mjx_models[0]
    # The 2 static fields live in the pytree treedef, not the leaves, so they
    # cannot be stacked -- pin them to a shared value first, or vmap raises a
    # tree-prefix metadata error. Safe to pin arbitrarily: neither field is
    # read anywhere in the MJX physics path for a `plane` floor (the ant's
    # floor geom), only inside the hfield branch of collision broad-phase.
    pinned = {
        f: np.maximum.reduce([np.asarray(getattr(m, f)) for m in mjx_models])
        for f in _STATIC_PINNED_FIELDS
    }
    base = base.tree_replace(pinned)
    in_axes = jax.tree_util.tree_map(lambda _: None, base)
    in_axes = in_axes.tree_replace({f: 0 for f in _BATCHED_FIELDS})
    stacked = {f: jp.stack([getattr(m, f) for m in mjx_models]) for f in _BATCHED_FIELDS}
    batched = base.tree_replace(stacked)
    return batched, in_axes


def build_batch(
    specs: Sequence[MorphologySpec],
    arena_spec: dict[str, ObjectSpec] | None = None,
    integrator: str | None = None,
    robot: str = "ant",
) -> tuple[mjx.Model, mjx.Model]:
    """Compile every spec and stack them into one batched model.

    `integrator` and `robot` are threaded through to `build_mj_model` rather
    than defaulted here, because every model in a batch must be compiled
    IDENTICALLY except for morphology -- a mismatched integrator or robot XML
    changes topology, and `batch_models` would then either fail to stack or
    silently produce a batch whose lanes are not comparable.
    """
    mj_models = [
        build_mj_model(spec, arena_spec, integrator, robot) for spec in specs
    ]
    return batch_models(mj_models)


def randomization_fn(
    mjx_model: mjx.Model,
    rng: jax.Array,
    num_morphologies: int,
    num_envs: int,
    arena_spec: dict[str, ObjectSpec] | None = None,
    integrator: str | None = None,
    robot: str = "ant",
) -> tuple[mjx.Model, mjx.Model, jax.Array]:
    """Sample `num_morphologies` ant bodies and repeat each to fill `num_envs`.

    Matches the single-positional-arg calling convention
    `mujoco_playground._src.wrapper.BraxDomainRandomizationVmapWrapper` uses
    (`randomization_fn(self.mjx_model)`) via a `functools.partial` the caller
    builds -- see `mjx_safety_gym.algorithms.train_ppo`. Returns a THIRD value
    beyond what that wrapper consumes: the per-env gene batch, shape
    `(num_envs, NUM_GENES)`. The caller feeds `(batched_model, in_axes)` to
    the wrapper and the gene batch to `MorphologyObsWrapper`
    (`mjx_safety_gym.algorithms.wrappers`) separately -- genes cannot be
    recovered from the compiled `mjx.Model` itself, only geometry derived
    from them.

    `mjx_model` (the env's own nominal model) is accepted only to match that
    calling convention; every morphology here is rebuilt from scratch via
    `build_mj_model`, so its value is unused.

    `num_envs` must be a multiple of `num_morphologies`; each sampled body is
    replicated `num_envs // num_morphologies` times so every field's leading
    dimension matches what `reset`'s per-env `rng` array requires. Sampling
    itself is host-side numpy (MjSpec compilation isn't traceable), seeded
    once from `rng` -- this function runs eagerly at wrapper-construction
    time, not under jit, so the population is fixed for the run's lifetime.
    """
    del mjx_model  # unused; every morphology is rebuilt from ant.xml directly
    if num_envs % num_morphologies != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be a multiple of num_morphologies "
            f"({num_morphologies})"
        )
    replicas = num_envs // num_morphologies
    seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
    np_rng = np.random.default_rng(seed)
    specs = [MorphologySpec.sample(np_rng) for _ in range(num_morphologies)]
    batched, in_axes = build_batch(specs, arena_spec, integrator, robot)
    genes = np.stack([s.genes for s in specs])  # (num_morphologies, NUM_GENES)

    batched = batched.tree_replace(
        {f: jp.repeat(getattr(batched, f), replicas, axis=0) for f in _BATCHED_FIELDS}
    )
    genes = np.repeat(genes, replicas, axis=0)  # (num_envs, NUM_GENES)
    return batched, in_axes, jp.asarray(genes, dtype=jp.float32)

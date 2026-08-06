"""Regression check: does `max_geom_pairs=16` (ant.xml) ever silently drop a
real limb-vase collision, at the largest morphology scale this project
generates (SCALE_HI=1.4, mjx_safety_gym/morphology.py)?

Why this exists: `mjx_safety_gym/collision.py`'s `geoms_colliding()` looks a
specific (geom1, geom2) pair up directly in `mjx.Data.contact`. If MJX's
broad-phase cull (the `max_geom_pairs` <numeric>, see ant.xml's own comment)
drops that pair before exact collision runs, the lookup silently reports "not
colliding" for a pair that may genuinely be touching -- a direct hit on the
safety-cost signal (`GoToGoal.get_cost`), not a downstream approximation.
ant.xml's own comment already flagged this as verified only for the nominal
(unscaled) ant standing on the floor; morphology randomization scales body
size, so a larger ant has more physical reach, making more simultaneous
close contacts geometrically plausible even though the pair-*group* size is
fixed (topology never changes, only capsule dimensions do).

Method: build two otherwise-identical compiled models that differ only in
the max_geom_pairs numeric -- the real 16, and 150 (comfortably above the
scene's theoretical max of 130 possible capsule-vase (120) + torso-vase (10)
pairs, i.e. effectively uncapped). Construct a single, fully deterministic
qpos (no dynamics, no integration) where all 10 vases sit at exact midpoints
between adjacent limb-capsule centres, guaranteeing deep (not
boundary-grazing) overlap with at least 2 capsules each. A single
`mjx.forward()` per model reads off the resulting contacts -- no chaotic RK4
rollout involved, so this is immune to the run-to-run floating-point noise
near collision boundaries that an earlier (organic-dynamics) version of this
test ran into. Then compares the per-vase "is anything touching me" boolean
`get_cost` actually relies on, between the two models, at that shared state.

Result as of 2026-08-06 (recorded here since this has no CI to pin it):
this construction produces 102 simultaneous real (dist < 0) limb-vase pairs
at max scale -- 6x the 16-slot cap, and a substantially denser pileup than
default arena placement (10 vases, keepout 0.15) would ever organically
produce. Even so, all 10 vases still registered correctly under the capped
model: the torso geom (large, centrally located) reliably held a broad-phase
slot against every vase, giving each vase redundant coverage even after 60
of the 102 real pairs were dropped. This is an empirical result tied to this
specific (compact, legs-folded) rest pose, not a formal proof for every pose
reachable during a rollout -- rerun this after any change to morphology
scale bounds (SCALE_LO/SCALE_HI) or arena vase count/keepout.

Usage:
    python -m scripts.verify_contact_capping
"""

import numpy as np
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx

from mjx_safety_gym import jax_cache

jax_cache.configure()

from mjx_safety_gym import morphology
from mjx_safety_gym.world import ObjectSpec, build_arena
from mjx_safety_gym.collision import geoms_colliding

ARENA = {
    "robot": ObjectSpec(0.4, 1),
    "goal": ObjectSpec(0.305, 1),
    "hazards": ObjectSpec(0.18, 10),
    "vases": ObjectSpec(0.15, 10),
}

ROBOT_GEOMS = [
    "torso_geom",
    "aux_1_geom", "left_leg_geom", "left_ankle_geom",
    "aux_2_geom", "right_leg_geom", "right_ankle_geom",
    "aux_3_geom", "back_leg_geom", "third_ankle_geom",
    "aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom",
]

# Adjacent-on-the-same-limb (or torso-adjacent) capsule pairs. Each vase is
# placed at one pair's midpoint, guaranteeing deep overlap with both.
LEG_PAIRS = [
    ("aux_1_geom", "left_leg_geom"), ("left_leg_geom", "left_ankle_geom"),
    ("aux_2_geom", "right_leg_geom"), ("right_leg_geom", "right_ankle_geom"),
    ("aux_3_geom", "back_leg_geom"), ("back_leg_geom", "third_ankle_geom"),
    ("aux_4_geom", "rightback_leg_geom"), ("rightback_leg_geom", "fourth_ankle_geom"),
    ("torso_geom", "aux_1_geom"), ("torso_geom", "aux_3_geom"),
]

# ant.xml's own standing pose (its "init_qpos" custom numeric): root xyz +
# quat, then 8 hip/ankle joint angles.
INIT_QPOS = np.array(
    [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
)


def _build(spec: morphology.MorphologySpec, cap: int) -> mj.MjModel:
    """Mirrors morphology.build_mj_model, with the cap overridable for this
    test (build_mj_model itself always uses ant.xml's baked-in value)."""
    s = mj.MjSpec.from_file(str(morphology._ANT_XML))
    geoms = {g.name: g for g in s.geoms}
    bodies = {b.name: b for b in s.bodies}
    scales = spec.scales
    for segment, geom_names in morphology._SEGMENT_GEOMS.items():
        len_scale = scales[f"{segment}_len"]
        rad_scale = scales[f"{segment}_rad"]
        for name in geom_names:
            g = geoms[name]
            fromto = np.array(g.fromto)
            g.fromto = np.concatenate([fromto[:3], fromto[3:] * len_scale])
            g.size = np.array([g.size[0] * rad_scale, 0.0, 0.0])
        for child_name in morphology._SEGMENT_CHILD_BODIES.get(segment, ()):
            b = bodies[child_name]
            b.pos = np.array(b.pos) * len_scale
    torso = geoms["torso_geom"]
    torso.size = np.array([torso.size[0] * scales["torso_rad"], 0.0, 0.0])
    build_arena(s, objects=ARENA, visualize=True)
    s.numeric("max_geom_pairs").data = [float(cap)]
    return s.compile()


def _per_vase_collisions(data, robot_geom_ids, vase_ids) -> list[bool]:
    return [
        any(bool(geoms_colliding(data, vid, rg)) for rg in robot_geom_ids)
        for vid in vase_ids
    ]


def _count_true_touches(data, robot_geom_ids, vase_ids) -> int:
    geom = np.array(data.contact.geom)
    dist = np.array(data.contact.dist)
    robot_set, vase_set = set(robot_geom_ids), set(vase_ids)
    hits = (dist < 0) & np.array(
        [
            (g0 in robot_set and g1 in vase_set) or (g1 in robot_set and g0 in vase_set)
            for g0, g1 in geom.tolist()
        ]
    )
    return int(hits.sum())


def main() -> None:
    spec_max = morphology.MorphologySpec(genes=np.ones(morphology.NUM_GENES))
    mj_uncapped = _build(spec_max, cap=150)
    mj_capped = _build(spec_max, cap=16)
    assert mj_uncapped.ngeom == mj_capped.ngeom == 35
    print(f"mass at max scale: {morphology.total_mass(mj_uncapped):.2f} kg")

    mx_uncapped = mjx.put_model(mj_uncapped)
    mx_capped = mjx.put_model(mj_capped)

    robot_geom_ids = [mj_uncapped.geom(n).id for n in ROBOT_GEOMS]
    assert robot_geom_ids == [mj_capped.geom(n).id for n in ROBOT_GEOMS]
    vase_ids = [mj_uncapped.geom(f"vase_{i}_geom").id for i in range(10)]
    assert vase_ids == [mj_capped.geom(f"vase_{i}_geom").id for i in range(10)]
    robot_geom_ids_arr = jp.array(robot_geom_ids)

    qpos = np.array(mjx.make_data(mx_uncapped).qpos)
    qpos[:15] = INIT_QPOS
    d0 = mjx.forward(mx_uncapped, mjx.make_data(mx_uncapped).replace(qpos=jp.asarray(qpos)))
    capsule_xpos = dict(zip(ROBOT_GEOMS, np.array(d0.geom_xpos[robot_geom_ids_arr])))

    for i, (a, b) in enumerate(LEG_PAIRS):
        mid = (capsule_xpos[a] + capsule_xpos[b]) / 2.0
        adr = mj_uncapped.jnt_qposadr[mj_uncapped.joint(f"vase_{i}_joint").id]
        qpos[adr : adr + 3] = mid
        qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

    d_uncapped = mjx.forward(mx_uncapped, mjx.make_data(mx_uncapped).replace(qpos=jp.asarray(qpos)))
    d_capped = mjx.forward(mx_capped, mjx.make_data(mx_capped).replace(qpos=jp.asarray(qpos)))

    n_true = _count_true_touches(d_uncapped, robot_geom_ids, vase_ids)
    print(f"real (dist<0) limb-vase pairs at this pose: {n_true} (cap is 16)")

    hits_uncapped = _per_vase_collisions(d_uncapped, robot_geom_ids, vase_ids)
    hits_capped = _per_vase_collisions(d_capped, robot_geom_ids, vase_ids)
    print(f"per-vase hits, uncapped (ground truth): {hits_uncapped}")
    print(f"per-vase hits, capped (cap=16):         {hits_capped}")

    if hits_uncapped == hits_capped:
        print(
            f"\nPASS: per-vase cost signal matches despite {n_true} real pairs "
            "exceeding the 16-slot cap."
        )
    else:
        missed = [i for i in range(10) if hits_uncapped[i] and not hits_capped[i]]
        raise AssertionError(
            f"max_geom_pairs=16 silently drops cost for vase(es) {missed}, "
            "which ARE touching under the uncapped ground truth. Raise "
            "max_geom_pairs in ant.xml."
        )


if __name__ == "__main__":
    main()

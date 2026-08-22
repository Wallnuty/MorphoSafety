from collections import defaultdict
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jp

import mujoco as mj
from mjx_safety_gym import lidar


_EXTENTS = (-2.0, -2.0, 2.0, 2.0)


class ObjectSpec(NamedTuple):
    keepout: float
    num_objects: int


INTEGRATORS = {
    "rk4": mj.mjtIntegrator.mjINT_RK4,
    "implicitfast": mj.mjtIntegrator.mjINT_IMPLICITFAST,
    "euler": mj.mjtIntegrator.mjINT_EULER,
}


def apply_integrator(spec: mj.MjSpec, integrator: str | None) -> None:
    """Override the integrator an XML declares, in place, before compile().

    Exists because the integrator is the single largest throughput lever in
    this workload and the two robots disagree on it by accident of ancestry:
    `ant.xml` declares `integrator="RK4"` (inherited verbatim from
    safety-gymnasium) while `point.xml` declares nothing and gets MuJoCo's
    default Euler. RK4 does FOUR force evaluations per step against one, and
    physics is ~90% of this workload's per-decision cost -- measured 3.5x
    cheaper stepping on CPU for implicitfast/euler.

    It is NOT a free win, which is why this is an explicit override rather than
    a change to the XML: the dynamics genuinely differ. Measured on the nominal
    ant at the shipped timestep of 0.01, a scripted gait travels 1.010 m under
    RK4 and 0.770 m under implicitfast (~25% less), though passive settling is
    near-identical (equilibrium torso height 0.159 vs 0.161 m, max trajectory
    deviation 0.0225 m) and neither shows instability. implicitfast and euler
    agree with each other to the digits printed. Standard practice would pair a
    single-evaluation integrator with a smaller timestep; implicitfast at
    dt=0.005 would still be ~2x cheaper than RK4 at dt=0.01 and more accurate
    than either -- untested here.

    Applied to the MjSpec rather than the compiled model so that every compile
    path picks it up identically: GoToGoal.__init__ and morphology.py's
    build_mj_model both compile from the same XML, and if only one honoured the
    override, morphology-randomized runs would silently step different physics
    from the base env.

    Passing None leaves whatever the XML declares, so existing behaviour and
    every checkpoint trained under it are untouched by default.
    """
    if integrator is None:
        return
    key = integrator.lower()
    if key not in INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. Available: {sorted(INTEGRATORS)}"
        )
    spec.option.integrator = INTEGRATORS[key]


# sample_layout(vase: [10, 5], hazard: [20, 2], goal : []): [-2, -2, 2, 2]-> (vase: [x y theta])
def build_arena(
    spec: mj.MjSpec,
    objects: dict[str, ObjectSpec],
    visualize: bool = False,
    floor_half_size: tuple[float, float] | None = None,
    obstacle_scale: float = 1.0,
    vase_mass: float | None = None,
    lidar_groups=None,
):
    """Build the arena (currently, just adds Lidar rings). Future: dynamically add obstacles, hazards, objects, goal here

    `floor_half_size` overrides the square floor derived from `_EXTENTS`. It
    exists for corridor-shaped tasks (see envs/run_forward.py), which need a
    floor much longer in x than wide in y -- the default square 2.1 m half-
    extent is barely wider than one episode of ant travel. Passing None keeps
    the historical behaviour byte-for-byte.

    `obstacle_scale` multiplies every obstacle dimension. The hazard/vase/goal
    sizes below were chosen for safety-gym's ~0.1 m point robot and are
    meaningless against a robot of a very different size -- the ant_gym robot
    has a 3.6 m leg span, so unscaled 0.2 m hazards would sit under its feet as
    rounding errors rather than obstacles. Robots declare their scale via
    `arena_scale` in `_ROBOT_CONFIGS` (envs/go_to_goal.py) and it arrives here.
    1.0 reproduces the original arena exactly.

    `vase_mass` sets an explicit vase mass instead of letting MuJoCo derive one
    from geom density. It matters because the robot XMLs set
    `inertiafromgeom="true"`, so the `mass=` passed to add_body below is IGNORED
    and a vase weighs density * volume -- which scales as obstacle_scale**3. At
    obstacle_scale=4 that is 2.56 kg per vase against the 0.911 kg ant_gym
    robot, i.e. an immovable wall, where the original arena had 0.04 kg vases
    against a 42 kg ant. Passing None keeps the density-derived mass and so
    reproduces the original behaviour exactly for point and ant.
    """
    # Set floor size
    maybe_floor = spec.worldbody.geoms[0]
    assert maybe_floor.name == "floor"
    if floor_half_size is None:
        size = max(_EXTENTS)
        floor_half_size = (size + 0.1, size + 0.1)
    maybe_floor.size = jp.array([floor_half_size[0], floor_half_size[1], 0.1])

    # Reposition robot
    for i in range(objects["vases"].num_objects):
        vase_half = 0.1 * obstacle_scale
        volume = vase_half**3
        density = 0.001
        vase = spec.worldbody.add_body(
            name=f"vase_{i}",
            mass=volume * density,
        )

        vase_geom_kwargs = {} if vase_mass is None else {"mass": vase_mass}
        vase.add_geom(
            name=f"vase_{i}_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[vase_half, vase_half, vase_half],
            rgba=[0, 1, 1, 1],
            userdata=jp.ones(1),
            **vase_geom_kwargs,
        )

        # Free joint bug in visualizer: https://github.com/google-deepmind/mujoco/issues/2508
        vase.add_freejoint(name=f"vase_{i}_joint")

    for i in range(objects["hazards"].num_objects):
        hazard = spec.worldbody.add_body(name=f"hazard_{i}", mocap=True)
        hazard.add_geom(
            name=f"hazard_{i}_geom",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2 * obstacle_scale, 0.01 * obstacle_scale, 0],
            rgba=[0.0, 0.0, 1.0, 0.25],
            userdata=jp.ones(1),
            contype=jp.zeros(()),
            conaffinity=jp.zeros(()),
        )

    goal = spec.worldbody.add_body(name="goal", mocap=True)
    goal.add_geom(
        name="goal_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.3 * obstacle_scale, 0.15 * obstacle_scale, 0],
        rgba=[0, 1, 0, 0.25],
        contype=jp.zeros(()),
        conaffinity=jp.zeros(()),
    )

    # Visualize lidar rings -- ONLY the ones the env actually emits. Passing
    # None draws all of LIDAR_GROUPS, which leaves faint ghost rings (alpha 0.1)
    # hovering over a robot that does not have them.
    if visualize:
        lidar.add_lidar_rings(spec, lidar_groups)


def placement_not_valid(xy, object_keepout, other_xy, other_keepout):
    def check_single(other_xy, other_keepout):
        dist = jp.linalg.norm(xy - other_xy)
        return dist < (other_keepout + object_keepout)

    validity_checks = jax.vmap(check_single)(other_xy, other_keepout)
    return jp.any(validity_checks)


def draw_until_valid(rng, object_keepout, other_xy, other_keepout):
    def cond_fn(val):
        i, conflicted, *_ = val
        return jp.logical_and(i < 1000, conflicted)

    def body_fn(val):
        i, _, _, rng = val
        rng, rng_ = jax.random.split(rng)
        xy = draw_placement(rng_, object_keepout)
        conflicted = placement_not_valid(xy, object_keepout, other_xy, other_keepout)
        return i + 1, conflicted, xy, rng

    # Initial state: (iteration index, conflicted flag, placeholder for xy)
    init_val = (0, True, jp.zeros((2,)), rng)  # Assuming xy is a 2D point
    i, _, xy, *_ = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return xy, i


def _sample_layout(
    rng: jax.Array, objects_spec: dict[str, ObjectSpec]
) -> dict[str, list[tuple[int, jax.Array]]]:
    num_objects = sum(spec.num_objects for spec in objects_spec.values())
    all_placements = jp.ones((num_objects, 2)) * 100.0
    all_keepouts = jp.zeros(num_objects)
    layout = defaultdict(list)
    flat_idx = 0
    for _, (name, object_spec) in enumerate(objects_spec.items()):
        rng, rng_ = jax.random.split(rng)
        keys = jax.random.split(rng_, object_spec.num_objects)
        for _, key in enumerate(keys):
            xy, iter_ = draw_until_valid(
                key, object_spec.keepout, all_placements, all_keepouts
            )
            # TODO (yarden): technically should quit if not valid sampling.
            all_placements = all_placements.at[flat_idx, :].set(xy)
            all_keepouts = all_keepouts.at[flat_idx].set(object_spec.keepout)
            layout[name].append((flat_idx, xy))
            flat_idx += 1

            # Warn via a callback rather than lax.cond: under vmap (which is how
            # resets run during training) a batched cond lowers to a select that
            # executes *both* branches, so a debug.print in the failure branch
            # fires on every reset regardless of `iter_`. debug.callback is
            # mapped per batch element with concrete values, so it only warns
            # when sampling genuinely gave up.
            jax.debug.callback(
                partial(_warn_invalid_sample, name=name), iter_ >= 1000
            )
    return layout


def _warn_invalid_sample(failed, *, name: str) -> None:
    if failed:
        print(f"Failed to find a valid sample for {name}")


def constrain_placement(placement: tuple, keepout: float) -> tuple:
    """Helper function to constrain a single placement by the keepout radius"""
    xmin, ymin, xmax, ymax = placement
    return xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout


def draw_placement(rng: jax.Array, keepout) -> jax.Array:
    choice = constrain_placement(_EXTENTS, keepout)
    xmin, ymin, xmax, ymax = choice
    min_ = jp.hstack((xmin, ymin))
    max_ = jp.hstack((xmax, ymax))
    pos = jax.random.uniform(rng, shape=(2,), minval=min_, maxval=max_)
    return pos

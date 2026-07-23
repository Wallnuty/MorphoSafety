from collections import defaultdict
from typing import NamedTuple
import jax
import jax.numpy as jp

import mujoco as mj
from mjx_safety_gym import lidar


_EXTENTS = (-2.0, -2.0, 2.0, 2.0)


class ObjectSpec(NamedTuple):
    keepout: float
    num_objects: int


# sample_layout(vase: [10, 5], hazard: [20, 2], goal : []): [-2, -2, 2, 2]-> (vase: [x y theta])
def build_arena(
    spec: mj.MjSpec, objects: dict[str, ObjectSpec], visualize: bool = False
):
    """Build the arena (currently, just adds Lidar rings). Future: dynamically add obstacles, hazards, objects, goal here"""
    # Set floor size
    maybe_floor = spec.worldbody.geoms[0]
    assert maybe_floor.name == "floor"
    size = max(_EXTENTS)
    maybe_floor.size = jp.array([size + 0.1, size + 0.1, 0.1])

    # Reposition robot
    for i in range(objects["vases"].num_objects):
        volume = 0.1**3
        density = 0.001
        vase = spec.worldbody.add_body(
            name=f"vase_{i}",
            mass=volume * density,
        )

        vase.add_geom(
            name=f"vase_{i}_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.1, 0.1],
            rgba=[0, 1, 1, 1],
            userdata=jp.ones(1),
        )

        # Free joint bug in visualizer: https://github.com/google-deepmind/mujoco/issues/2508
        vase.add_freejoint(name=f"vase_{i}_joint")

    for i in range(objects["hazards"].num_objects):
        hazard = spec.worldbody.add_body(name=f"hazard_{i}", mocap=True)
        hazard.add_geom(
            name=f"hazard_{i}_geom",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2, 0.01, 0],
            rgba=[0.0, 0.0, 1.0, 0.25],
            userdata=jp.ones(1),
            contype=jp.zeros(()),
            conaffinity=jp.zeros(()),
        )

    goal = spec.worldbody.add_body(name="goal", mocap=True)
    goal.add_geom(
        name="goal_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.3, 0.15, 0],
        rgba=[0, 1, 0, 0.25],
        contype=jp.zeros(()),
        conaffinity=jp.zeros(()),
    )

    # Visualize lidar rings
    if visualize:
        lidar.add_lidar_rings(spec)


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

            jax.lax.cond(
                iter_ >= 1000,
                lambda _: jax.debug.print(f"Failed to find a valid sample for {name}"),
                lambda _: None,
                operand=None,
            )
    return layout


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

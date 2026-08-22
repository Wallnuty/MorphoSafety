import jax
import mujoco as mj
import jax.numpy as jp

# Every ring this module knows how to COMPUTE. "object" is safety-gym's slot for
# the Push task's movable box, which this repo does not implement -- GoToGoal
# assigns `_object_body_ids = []` in _post_init and never writes to it, so the
# ring is 16 constant zeros wherever it is enabled. Kept here (not deleted) so
# that older checkpoints can still be loaded by asking for it explicitly, and so
# a future Push task has a name to switch back on.
LIDAR_GROUPS = ["obstacle", "goal", "object"]

# What an env gets when it does not ask for anything specific. Excludes the dead
# "object" ring -- MEASURED 2026-08-22 over ~7,700 real observations on the goal
# task: obstacle 16/16 live, goal 16/16 live, object 0/16, max value exactly
# 0.0000. Dropping it narrows the goal task's observation by 16.
#
# NOT a reason to trim the obstacle ring as well: the same audit found only
# 11/16 of its bins lit, but those are BEARING bins that a barely-moving random
# policy never happened to point at, not structurally dead dimensions. Removing
# bins would break the ring's rotational structure.
DEFAULT_LIDAR_GROUPS = ["obstacle", "goal"]

# For calculations
NUM_LIDAR_BINS = 16
LIDAR_MAX_DIST = 2.0

# Visualisation
BASE_OFFSET = 0.5
OFFSET_STEP = 0.06
RADIANS = 0.15
LIDAR_SIZE = 0.025


def compute_lidar(
    robot_pos: jax.Array, robot_mat: jax.Array, targets_pos: jax.Array
) -> jax.Array:
    obs = jp.zeros(NUM_LIDAR_BINS)

    def ego_xy(pos):
        robot_3vec = robot_pos
        """Transforms world position to ego-centric robot frame in 2D."""
        pos_3vec = jp.concatenate(
            [pos, jp.array([0.0])]
        )  # Add zero z-coordinate -- not needed I thin
        world_3vec = pos_3vec - robot_3vec  # make sure obstacle pos is 3D
        return jp.matmul(world_3vec, robot_mat)[:2]  # Extract X, Y onl

    for pos in targets_pos:
        #   pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]  # Truncate Z coordinate

        z = jax.lax.complex(*ego_xy(pos))

        dist = jp.abs(z)
        angle = jp.angle(z) % (jp.pi * 2)

        bin_size = (jp.pi * 2) / NUM_LIDAR_BINS
        bin_ = (angle / bin_size).astype(jp.int32)
        bin_angle = bin_size * bin_

        sensor = jp.maximum(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST

        obs = obs.at[bin_].set(jp.maximum(obs[bin_], sensor))
        alias = (angle - bin_angle) / bin_size

        bin_plus = (bin_ + 1) % NUM_LIDAR_BINS
        bin_minus = (bin_ - 1) % NUM_LIDAR_BINS
        obs = obs.at[bin_plus].set(jp.maximum(obs[bin_plus], alias * sensor))
        obs = obs.at[bin_minus].set(jp.maximum(obs[bin_minus], (1 - alias) * sensor))

    return obs


# Each ring is drawn in the colour of the thing it SENSES, matching the arena
# geoms built in world.build_arena. Previously the colour was just
# `rgba[i] = 1` over the group index, which made the obstacle ring RED while
# hazards are blue -- and made the (dead) object ring the blue one, so the only
# blue ring in the viewer was the one carrying no information at all.
#
# The obstacle ring senses hazards AND vases (`_obstacle_body_ids`), which are
# blue and cyan respectively. It is drawn hazard-blue: on `minefield` there are
# no vases, so it is exact there, and hazards dominate on `run` too.
RING_RGBA = {
    "obstacle": [0.0, 0.0, 1.0, 1.0],   # hazards: world.py rgba [0, 0, 1, 0.25]
    "goal": [0.0, 1.0, 0.0, 1.0],       # goal:    world.py rgba [0, 1, 0, 0.25]
    "object": [0.0, 1.0, 1.0, 1.0],     # vases:   world.py rgba [0, 1, 1, 1]
}


def add_lidar_rings(spec: mj.MjSpec, groups=None):
    """Add the viewer's lidar ring sites for `groups` only.

    `groups` must be the env's ACTUAL `lidar_groups`. It used to add all of
    LIDAR_GROUPS unconditionally, on the reasoning that unused sites are visual,
    collide with nothing, and simply stay dark -- but "dark" is alpha 0.1, not
    invisible (`update_lidar_rings` sets alpha to `value + 0.1`), so a disabled
    ring still shows up as a faint circle of dots hovering over the robot. On
    the corridor tasks that meant two ghost rings above a robot that only has
    one, including the permanently-empty `object` ring.

    Height is keyed off the group's index in LIDAR_GROUPS rather than its
    position in `groups`, so a ring sits at the same height whichever others
    are enabled.
    """
    robot_body = spec.body("robot")
    groups = LIDAR_GROUPS if groups is None else list(groups)

    for category in groups:
        i = LIDAR_GROUPS.index(category)
        lidar_body = robot_body.add_body(name=f"lidar_{category}")
        for bin in range(NUM_LIDAR_BINS):
            theta = 2 * jp.pi * (bin + 0.5) / NUM_LIDAR_BINS
            binpos = jp.array(
                [
                    jp.cos(theta) * RADIANS,
                    jp.sin(theta) * RADIANS,
                    BASE_OFFSET + OFFSET_STEP * i,
                ]
            )
            lidar_body.add_site(
                name=f"lidar_{category}_{bin}",
                size=LIDAR_SIZE * jp.ones(3),  # Size of the lidar site
                rgba=list(RING_RGBA[category]),  # colour of what this ring senses
                pos=binpos,  # Position of the lidar site
            )


def update_lidar_rings(lidar_values: jax.Array, model: mj.MjModel, groups=None):
    """Light up the viewer's lidar rings from an observation slice.

    `groups` names which rings `lidar_values` holds, in order, and must match
    whatever the env actually put in the observation -- envs no longer
    necessarily emit all of LIDAR_GROUPS (see GoToGoal's `lidar_groups`). It
    defaults to all of them for callers that still pass a full stack.

    Note the SITES for every group exist regardless; only the observation
    shrinks. Sites are visual and collide with nothing, so leaving the unused
    rings in the model costs nothing and keeps model structure stable across
    configurations. They simply stay dark.
    """
    groups = LIDAR_GROUPS if groups is None else list(groups)
    if len(lidar_values) != len(groups):
        raise ValueError(
            f"got {len(lidar_values)} lidar rings for groups {groups} -- the "
            f"caller sliced the observation with the wrong ring count"
        )
    # Update data just for viewer
    for lidars, category in zip(lidar_values, groups):
        for i, value in enumerate(lidars):
            lidar_site_id = model.site(f"lidar_{category}_{i}").id
            model.site_rgba[lidar_site_id][3] = min(1.0, value + 0.1)  # Change alpha

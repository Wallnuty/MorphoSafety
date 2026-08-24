"""Lasers: RunForward over a corridor of full-width tripwire beams.

WHY THIS TASK EXISTS
--------------------
`Minefield` was already drifting from a routing problem toward a GAIT problem
-- evenly spaced discs at a fixed pitch, where the answer is a stride that
misses them rather than a route that goes around them. It never got all the
way there, because discs leave gaps: the widest disc-free lane on the default
lattice is 0.210 m, so a lateral weave is still part of the solution and a
policy can trade "walk straighter" against "cost".

Lasers remove that escape entirely. Each beam spans the FULL FLOOR WIDTH, so
there is no y at which a beam can be dodged. Lateral position is irrelevant to
cost, and the only remaining degree of freedom is WHEN a foot comes down.
That makes this a pure foot-placement/stride-timing problem, which is the
cleanest version of the question the minefield was groping at.

It is solvable for exactly the reason the ant already exploits on minefield:
cost is charged only for a geom that is ON THE GROUND inside a beam (see
`--hazard_step_on`), so a limb swung through the air over a beam is free. The
observed "gallop over the mines" behaviour is not a loophole here, it is the
intended and only solution.

GEOMETRY, AND WHY FULL-WIDTH MATTERS FOR CORRECTNESS
----------------------------------------------------
Beams are axis-aligned: a thin slab in x, spanning the whole floor in y, lying
on the ground. Because a beam is effectively infinite in y over everything the
robot can reach, the distance from a robot geom to a beam is a ONE-DIMENSIONAL
problem -- the distance from the geom's swept x-interval to the beam's x. No
segment-segment distance is needed, and no special case for crossing segments.

That exactness depends on full width. A partial-width beam would be dodgeable
around its ends, and this cost function would wrongly charge for it, so the
drawn geometry deliberately spans the floor rather than just the corridor.
Do not narrow the beams without replacing the distance test.

WHAT IT SHARES WITH MINEFIELD
-----------------------------
Everything except the obstacles: reward, corridor, walls, boundary cost,
upright bonus, flip termination, goal sensing and observation layout are all
inherited from `RunForward` unchanged. There are no hazards and no vases, so
`nq`/`nv` match Minefield's and the beams -- like hazards -- carry
contype=0/conaffinity=0 and add no contacts and no DOFs.

OBSERVATION WIDTH IS UNCHANGED (47 for the ant with the lidar ring), so
checkpoints transfer to and from `run`/`minefield` with no network surgery.
NOTE the beams are NOT in a lidar group: the obstacle ring stays empty of them
deliberately, because a full-width beam has no bearing -- every direction is
equally blocked -- so a ring would carry no information a range sensor could
use. What the policy has instead is its own x, recoverable from goal bearing +
range + magnetometer yaw, against beams at a FIXED, known pitch.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np

from mjx_safety_gym.envs.run_forward import RunForward
from mjx_safety_gym.world import build_arena


class Lasers(RunForward):
    """`RunForward` over evenly spaced, full-width laser beams."""

    # 10 beams over the ant's ~10 m obstacle band is a ~1.0 m pitch, matching
    # the minefield lattice's x-pitch so the two tasks pose the same stride
    # frequency and their costs are on comparable footing.
    DEFAULT_NUM_LASERS = 10
    # FULL thickness in x, before arena scaling. 0.10 gives a 0.05 half-width;
    # against the ant's 0.02 foot radius the forbidden band is 0.07 wide at a
    # 1.0 m pitch, i.e. 7% of the runway is untouchable.
    DEFAULT_LASER_WIDTH = 0.10

    def __init__(
        self,
        robot: str = "ant",
        num_lasers: int | None = None,
        laser_width: float | None = None,
        **kwargs,
    ):
        self._num_lasers = (
            self.DEFAULT_NUM_LASERS if num_lasers is None else int(num_lasers)
        )
        if self._num_lasers < 1:
            # Same reasoning as Minefield's hazard check: a corridor with no
            # beams has no obstacle cost at all, so cost could only ever come
            # from the boundary term. Fail loudly rather than train that blind.
            raise ValueError(
                f"Lasers needs at least one beam, got {self._num_lasers}. "
                "A beam-free corridor has no obstacle cost signal."
            )
        self._laser_width = (
            self.DEFAULT_LASER_WIDTH if laser_width is None else float(laser_width)
        )
        super().__init__(robot=robot, num_hazards=0, num_vases=0, **kwargs)

    # -- geometry ---------------------------------------------------------

    def _laser_positions(self) -> np.ndarray:
        """Beam x centres. Deterministic and evenly spaced, like the minefield
        lattice -- a fixed pitch is what makes the task a stride problem rather
        than a "find this episode's gap" problem."""
        lo, hi = self._obstacle_x_lo, self._obstacle_x_hi
        # Half-step inset at both ends so the first beam is not flush against
        # the start of the obstacle band, matching the lattice's cell-centre
        # placement (which is what puts minefield's first row 1.25 m ahead).
        step = (hi - lo) / self._num_lasers
        return lo + step * (np.arange(self._num_lasers) + 0.5)

    def _add_lasers(self, spec: mj.MjSpec) -> None:
        a = self._arena_scale
        half_w = 0.5 * self._laser_width * a
        # Spans the FULL FLOOR, not just the corridor -- see the module
        # docstring: the cost test ignores y, and that is only exact if a beam
        # cannot be walked around.
        floor_half_y = self._corridor_half_width + 1.5 * a
        half_h = 0.005 * a          # matches the hazard discs' halved height
        for i, x in enumerate(self._laser_positions()):
            # Its own material so the viewer can light one beam at a time; a
            # shared material could only ever light all of them together.
            # Base emission is nonzero because a laser should LOOK like a laser
            # even when unbroken -- unlike the hazard discs, which rest dark.
            spec.add_material(name=f"laser_{i}_mat", emission=0.35)
            spec.worldbody.add_geom(
                name=f"laser_{i}_geom",
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[half_w, floor_half_y, half_h],
                pos=[float(x), 0.0, half_h],
                material=f"laser_{i}_mat",
                rgba=[1.0, 0.12, 0.12, 0.85],
                # Non-colliding, exactly like hazards: no contacts, no DOFs, no
                # broad-phase entry. The beam is a COST region, not an object.
                contype=0,
                conaffinity=0,
            )

    def _build_arena(self, mjSpec: mj.MjSpec) -> None:
        build_arena(
            mjSpec,
            objects=self.spec,
            visualize=True,
            floor_half_size=(
                0.5 * self._corridor_length + 1.0 * self._arena_scale,
                self._corridor_half_width + 1.5 * self._arena_scale,
            ),
            obstacle_scale=self._arena_scale,
            lidar_groups=self._lidar_groups,
            hazard_size=self._hazard_size,
        )
        self._add_lasers(mjSpec)
        # MUST be repeated here -- this OVERRIDES RunForward._build_arena, so
        # that method's own trailing call does not run. This is the exact trap
        # that made the corridor walls silently absent on Minefield's first
        # test (ngeom 35 with and without them).
        if self._corridor_walls:
            self._add_corridor_walls(mjSpec)

    def _post_init(self, *args, **kwargs):
        super()._post_init(*args, **kwargs)
        # Read the beam x centres and half-width BACK OFF THE COMPILED MODEL
        # rather than recomputing them, so the charged band cannot drift from
        # the band that is drawn -- the same discipline as `_hazard_radius`.
        self._laser_x = jp.array(
            [
                float(self._mj_model.geom(f"laser_{i}_geom").pos[0])
                for i in range(self._num_lasers)
            ]
        )
        self._laser_half_width = float(
            self._mj_model.geom("laser_0_geom").size[0]
        )

    # -- cost -------------------------------------------------------------

    def laser_distances(self, data) -> jax.Array:
        """Distance in x from each beam to the robot's nearest surface, (L,).

        One-dimensional by construction: a beam spans the whole floor in y, so
        y cannot separate the robot from it. For each robot collision geom the
        swept x-interval is [min(ax,bx) - r, max(ax,bx) + r]; the distance from
        a beam at x_L is how far outside that interval it falls.
        """
        radius, half_len = self._robot_geom_extent()
        geom_pos = data.geom_xpos[self._robot_collision_geom_ids_arr]      # (G,3)
        geom_axis = data.geom_xmat[self._robot_collision_geom_ids_arr].reshape(
            -1, 3, 3
        )[:, :, 2]
        ax = geom_pos[:, 0] + geom_axis[:, 0] * half_len
        bx = geom_pos[:, 0] - geom_axis[:, 0] * half_len
        x_lo = jp.minimum(ax, bx) - radius                                  # (G,)
        x_hi = jp.maximum(ax, bx) + radius
        lx = self._laser_x[:, None]                                         # (L,1)
        d = jp.maximum(jp.maximum(x_lo[None, :] - lx, lx - x_hi[None, :]), 0.0)

        if self._hazard_step_on:
            # STEP-ON, and it is what makes the task solvable at all: without
            # it every crossing is charged, because a leg must sweep through
            # the beam's x whatever the gait. Gated on the geom's LOWEST POINT,
            # not on mjx contacts, for the same reason as the hazard version --
            # `max_geom_pairs=16` can truncate a real contact out of existence,
            # and geometry cannot be truncated.
            lowest_z = (
                geom_pos[:, 2] - jp.abs(geom_axis[:, 2]) * half_len - radius
            )
            grounded = lowest_z <= self._ground_contact_eps
            d = jp.where(grounded[None, :], d, 1e6)
        return jp.min(d, axis=1)

    def laser_contacts(self, data) -> jax.Array:
        """Boolean (L,): which beams are currently charging cost."""
        return self.laser_distances(data) <= self._laser_half_width

    def foot_obstacle_observations(self, data):
        """Per-foot clearance to the nearest BEAM: 4 entries per foot.

        Same contract as the hazard version (see GoToGoal), so the observation
        width is identical and checkpoints transfer between the two tasks. The
        geometry is simpler: a beam spans the whole floor, so clearance is
        one-dimensional in x and the "bearing" is just ahead or behind --
        cos = +-1, sin = 0 exactly. Those two constant-ish entries are kept
        rather than dropped precisely SO the width matches minefield.
        """
        if not self._foot_obstacle_obs or self._foot_geom_cols.size == 0:
            return None
        cols = self._robot_collision_geom_ids_arr[self._foot_geom_cols]
        radius, half_len = self._robot_geom_extent()
        r = radius[self._foot_geom_cols]
        hl = half_len[self._foot_geom_cols]
        gp = data.geom_xpos[cols]
        ga = data.geom_xmat[cols].reshape(-1, 3, 3)[:, :, 2]
        ax = gp[:, 0] + ga[:, 0] * hl
        bx = gp[:, 0] - ga[:, 0] * hl
        x_lo = jp.minimum(ax, bx) - r                       # (F,)
        x_hi = jp.maximum(ax, bx) + r
        lx = self._laser_x[:, None]                          # (L,1)
        # UNGATED by step-on, deliberately: a foot in the air must still see
        # what it is about to land on. Signed gap to the beam's edge.
        d = jp.maximum(jp.maximum(x_lo[None, :] - lx, lx - x_hi[None, :]), 0.0)
        nearest = jp.argmin(d, axis=0)                       # (F,)
        clearance = jp.min(d, axis=0) - self._laser_half_width
        scale = 2.0 * self._arena_scale
        clearance = jp.clip(clearance / scale, -1.0, 1.0)
        # Ahead (+x in the torso yaw frame) or behind.
        foot_x = 0.5 * (x_lo + x_hi)
        delta_x = self._laser_x[nearest] - foot_x
        mat = data.xmat[self._robot_body_id].reshape(3, 3)
        yaw = jp.arctan2(mat[1, 0], mat[0, 0])
        rel = jp.arctan2(jp.zeros_like(delta_x), delta_x) - yaw
        return jp.stack(
            [clearance, jp.cos(rel), jp.sin(rel), self._scaled_ground_gap(data)],
            axis=-1,
        ).flatten()

    def get_cost(self, data) -> jax.Array:
        # super() contributes the boundary term (RunForward) plus a zero
        # obstacle term (no hazards, no vases).
        return (
            super().get_cost(data) + jp.sum(self.laser_contacts(data))
        ).astype(jp.float32)

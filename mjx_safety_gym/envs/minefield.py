"""Minefield: RunForward with the dynamic obstacles taken out.

WHY THIS TASK EXISTS
--------------------
Identical to `RunForward` in reward, corridor, posture handling and
observation layout. The single difference is the arena: **no vases, more
hazards.** It exists to make training runs cheap enough to iterate on, and it
is worth being precise about where that saving comes from, because the two
obstacle classes are not remotely equal in cost:

    vases    dynamic. One free joint each = 7 qpos and 6 qvel per vase, plus
             real contacts against the floor, each other and every robot limb.
             At the default 10 they are ~82% of nq and ~81% of nv for the ant.
    hazards  mocap bodies with contype=0 and conaffinity=0. No DOFs, no
             contacts, no solver work. They cost one mocap_pos entry, one
             lidar bin contribution, and one row of the (H, G) distance
             matrix in get_cost.

So the vases dominate physics time while the hazards are close to free. This
task drops the former and doubles the latter, keeping the obstacle COUNT at 20
while removing every dynamic body from the scene.

The safety signal survives the trade because hazards were never collision-
based to begin with. `GoToGoal.get_cost` charges vases through
`geoms_colliding` (real contacts) but charges hazards through a swept-sphere
distance test against the robot's limbs -- pure geometry, no contacts
involved. Removing vases removes the contact-based half of the cost and leaves
the distance-based half intact and unchanged.

WHAT THIS BUYS, AND WHAT IT COSTS
---------------------------------
Buys: a much smaller state vector and a much smaller contact problem, and the
elimination of the one part of the scene that made `max_geom_pairs=16` worth
policing (limb-vs-vase pairs -- see scripts/verify_contact_capping.py). With
no vases the only contacts left are robot-vs-floor and robot self-collision.

Costs: the cost signal becomes purely proximity-based. There is nothing left
in the arena to knock over, so a policy cannot be punished for disturbing the
world -- only for entering a region. That is a genuinely weaker notion of
safety, and it is the reason `RunForward` is kept rather than replaced: this
task is the fast iteration loop, `run` is the one with the richer constraint.

OBSERVATION WIDTH IS UNCHANGED, DELIBERATELY
--------------------------------------------
Lidar is binned, so its width depends on the number of RINGS (3), never on the
number of objects in a ring. Obstacle count therefore does not enter
`observation_size` at all, and this task reports the same width as `run` and
`goal` (76 for the ant). Checkpoints transfer between all three without
network surgery -- which is the point: train fast here, then warm-start `run`.
"""

from __future__ import annotations

import mujoco as mj

from mjx_safety_gym.envs.run_forward import RunForward
from mjx_safety_gym.world import build_arena


class Minefield(RunForward):
    """`RunForward` over a corridor of hazards, with no vases at all.

    Reward, corridor geometry, boundary cost, upright bonus and flip
    termination are inherited unchanged -- see envs/run_forward.py for the
    reasoning behind each. Only the arena differs.
    """

    # Default obstacle count. 20 keeps the total number of obstacles equal to
    # RunForward's 10 hazards + 10 vases, so the corridor is not made emptier
    # by the swap -- only cheaper. Over the ant's 10.0 m x 2.0 m obstacle band
    # that is ~12.6% of the floor inside a hazard's cost radius.
    DEFAULT_NUM_HAZARDS = 20

    def __init__(
        self,
        robot: str = "ant",
        num_hazards: int | None = None,
        **kwargs,
    ):
        if num_hazards is None:
            num_hazards = self.DEFAULT_NUM_HAZARDS
        if int(num_hazards) < 1:
            # _post_init reads the cost threshold off `hazard_0_geom`, and with
            # no vases either there would be no obstacles at all -- a corridor
            # with nothing in it, where cost can only ever come from the
            # boundary term. Fail loudly rather than train that by accident.
            raise ValueError(
                f"Minefield needs at least one hazard, got {num_hazards}. "
                "A hazard-free corridor has no obstacle cost signal."
            )
        super().__init__(
            robot=robot,
            num_hazards=int(num_hazards),
            num_vases=0,
            **kwargs,
        )

    def _build_arena(self, mjSpec: mj.MjSpec) -> None:
        # Same corridor floor as RunForward, but `vase_mass` is not passed:
        # with num_vases=0 the vase loop in build_arena never runs, so the
        # argument would be inert. Left off so it is obvious from the call
        # site that this arena has no dynamic bodies.
        build_arena(
            mjSpec,
            objects=self.spec,
            visualize=True,
            floor_half_size=(
                0.5 * self._corridor_length + 1.0 * self._arena_scale,
                self._corridor_half_width + 1.5 * self._arena_scale,
            ),
            obstacle_scale=self._arena_scale,
        )

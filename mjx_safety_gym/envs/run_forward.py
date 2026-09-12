"""RunForward: get as far down a corridor as possible without hitting anything.

WHY THIS TASK EXISTS
--------------------
`GoToGoal` turned out to be unlearnable for the ant, and not because of a bug.
Its reward is a signed distance delta to a randomly placed goal, which sums over
an episode to `d_initial - d_final`. Distance-to-a-point is CONVEX, so by
Jensen's inequality any zero-mean displacement *increases* expected distance:
undirected motion has strictly negative expected return while standing still
scores exactly 0. Freezing is therefore the optimal policy reachable without a
working gait, and PPO correctly converged to it -- the trained policy moved
0.011 m per episode against 0.313 m for uniform random actions.

But the decisive measurement is this one, taken in GoToGoal itself over 16
seeds with the ant, comparing episode return against a do-nothing policy:

    zero action     -0.0000 +- 0.0000
    random actions  +0.0441 +- 0.1351     (standard error 0.034)
    scripted gait   +0.0781 +- 0.6715     (standard error 0.168)

A SCRIPTED GAIT THAT WALKS SEVERAL METRES EARNS STATISTICALLY ZERO RETURN.
Because the goal is placed in a uniformly random direction, walking is as
likely to move away from it as toward it, so locomotion by itself is worth
nothing: gait and steering have to be discovered *simultaneously* before any
reward appears, and there is no gradient rewarding the first without the
second. (The convexity effect above is real but second-order at these
displacements -- it is this that dominates.)

There was also no way to bootstrap from the sparse term. Goals spawn ~0.8 m
away with a 0.3 m capture radius, so the +1 bonus essentially never fired.

This task removes both traps by making the reward LINEAR in position:

    reward = (x_t - x_{t-1}) * forward_reward_weight

Summed over an episode that telescopes to `x_final - x_initial` -- literally
"how far did it get". Because x-displacement is linear rather than convex,
zero-mean motion has expected reward 0 instead of negative, so exploration is
no longer punished, and ANY perturbation of the policy that produces net +x
motion is immediately rewarded. There is one fixed direction, known from the
first step, so gait and steering no longer have to be discovered together.

This is the same shape of reward that makes Gym's `Ant-v4` and
safety-gymnasium's own `SafetyAntVelocity` work -- and notably, those are the
ONLY ant tasks either benchmark ships results for. safety-gymnasium publishes no
`SafetyAntGoal` benchmark, and CRAX registers `safe_goal_point` but no ant
navigation task at all. Nobody has demonstrated an ant learning goal navigation
in this family of benchmarks, so this is not us failing to reproduce a known
result.

THE DEGENERATE SOLUTION, AND WHY THERE IS A BOUNDARY COST
---------------------------------------------------------
With obstacles confined to a band and reward depending only on x, the optimal
"safe" policy is to walk sideways out of the band and then run +x in clean air,
collecting full reward at zero cost. That would make the safety question
vacuous. Rather than build physical walls -- which would add wall-vs-limb geom
pairs to a contact buffer already capped at `max_geom_pairs=16` in ant.xml, the
exact thing scripts/verify_contact_capping.py exists to police -- leaving the
corridor is charged as COST. That keeps the constraint in the place the research
question lives, and it composes correctly with the existing penalizers: an
unconstrained run is then *expected* to sprint wide with high cost (which is
what makes it a useful --safety_budget calibration), while a constrained one has
to weave through the obstacles.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
  * No control cost by default. A control cost is paid by any moving policy and
    not by a frozen one, so a positive weight re-creates the very freeze
    attractor this task was written to escape. It is available as a parameter
    but defaults to 0.0; raise it only once the ant reliably walks.
  * No height-based termination. Gym's Ant terminates outside a torso-z range
    of (0.2, 1.0), but that threshold is tuned to a robot whose torso radius is
    0.25 m; safety-gymnasium's ant is 4x smaller and settles at z ~= 0.13-0.16,
    so the same numbers would terminate it on the first step. See the posture
    section below for what replaced it.

WHAT WAS ADDED LATER, AND WHY (2026-08-12)
------------------------------------------
This module originally shipped with NO upright bonus and NO termination, on the
reasoning that "a flipped ant simply stops making progress, which this reward
already handles". THAT REASONING WAS WRONG, and measuring it is what unblocked
the project: over 32 full episodes the ant was inverted for 94.4% of steps
untrained and 94.1% after 1.5M steps of training. It does not stop making
progress when flipped -- it crawls on its back, earning enough to have no
gradient pressure to get up. Training moved uprightness by 0.3 points in 1.5M
steps because nothing in the reward ever mentioned it.

`healthy_reward` and `terminate_on_flip` (both defaulted ON for ants in
train_ppo._ROBOT_DEFAULTS, OFF for the point) are the fix. Every working ant in
safety-gymnasium, CRAX and Gym pairs this morphology with a healthy bonus AND
termination; we had adopted the morphology alone. See the posture section
below.
  * The goal body is NOT removed, even though nothing respawns it. It is parked
    once at the far end of the corridor as a fixed beacon.

OBSERVATION NARROWED TO ONE LIDAR RING (2026-08-15)
---------------------------------------------------
This task originally emitted all three lidar rings, purely so its observation
width matched GoToGoal's and checkpoints stayed interchangeable. Both of the
extra rings turned out to be identically zero here, measured over 10,000 real
observations from a trained policy:

    obstacle ring   16/16 dims live
    goal     ring    0/16 -- max 0.0000, in EVERY sample
    object   ring    0/16 -- `_object_body_ids` is [] and never written

The goal ring is dead because the goal is parked 11 m away (44 m for ant_gym)
against `LIDAR_MAX_DIST = 2.0`; closest approach measured over the whole run was
6.81 m. The object ring is safety-gym's Push-task slot, which this repo never
implements. Together they were 32 of 76 observation entries -- 42% of the input
was a constant.

`lidar_groups` defaults to ("obstacle",) here -- the OBSTACLE ring only. It was
briefly () between 2026-08-22 and 2026-08-23; see train_ppo's --hazard_lidar
for the measurement that put it back (it helps at a 256x4 policy and hurts at
32x4). The goal ring stays off regardless. The goal is still parked at
the far end, and its DIRECTION is still available -- via `goal_observation`,
which supplies bearing and range as three numbers rather than as a ring that
could not see it. GoToGoal keeps all three rings: its goal actually moves and
comes into range, so there the ring carries information.

CONSEQUENCE: run/minefield checkpoints no longer interchange with goal ones.
run and minefield still match each other, which is the pairing that matters
(train fast on minefield, warm-start run).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import mujoco as mj
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import _ROBOT_CONFIGS, GoToGoal
from mjx_safety_gym.mjx_env import State, step
from mjx_safety_gym.world import build_arena, placement_not_valid


# Corridor-wall colour. The second entry is the VIEWING style, selected by
# `draw_corridor_lines`: alpha 0 leaves the collider present but undrawn, and
# a flat `corridor_line_*` stripe marks the boundary instead. See
# RunForward._add_corridor_walls.
_WALL_RGBA_SOLID = [0.35, 0.35, 0.40, 0.6]
_WALL_RGBA_HIDDEN = [0.35, 0.35, 0.40, 0.0]


class RunForward(GoToGoal):
    """Run as far as possible in +x along an obstacle-strewn corridor.

    Inherits GoToGoal's observation stack, hazard/vase cost geometry and lidar
    wholesale; only the layout, the reward and the episode bookkeeping differ.
    """

    def __init__(
        self,
        robot: str = "ant",
        corridor_length: float | None = None,
        corridor_half_width: float | None = None,
        start_margin: float | None = None,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.0,
        boundary_cost_weight: float = 1.0,
        corridor_walls: bool = True,
        draw_corridor_lines: bool = False,
        healthy_reward: float = 0.0,
        terminate_on_flip: bool = False,
        terminate_on_goal: bool = True,
        goal_radius: float | None = None,
        goal_reward_weight: float = 0.0,
        goal_observation: bool = False,
        lidar_groups=("obstacle",),
        # Half-range of the robot's start-line y position, as a FRACTION of
        # corridor_half_width. 0.0 = the same spawn every episode. Was 0.5
        # (i.e. +-0.5 m) until 2026-09-12, user's call to fix it: with the
        # hazard lattice already deterministic since 2026-08-22, this was the
        # ONLY per-episode randomness left on minefield, and removing it makes
        # every body face the identical problem -- the right setting for a
        # controlled morphology comparison. The trade is stated in the
        # docstring of _hazard_lattice: with nothing random at all, "avoids
        # the mines" and "memorised one trajectory" become indistinguishable.
        # --start_y_jitter 0.5 reproduces every earlier run.
        start_y_jitter: float = 0.0,
        **kwargs,
    ):
        # Corridor dimensions default to a multiple of the robot's own arena
        # scale, because a corridor is only meaningful relative to the robot in
        # it: ant_gym has a 3.6 m leg span and would not fit in the 2 m-wide
        # corridor that suits ant. Explicit values are taken as-is (already in
        # metres), so a caller who passes numbers is never silently rescaled.
        # Read straight from _ROBOT_CONFIGS rather than waiting for
        # super().__init__ to set self._arena_scale, because _build_arena needs
        # these dimensions and runs *inside* that call.
        a = float(_ROBOT_CONFIGS[robot]["arena_scale"])
        # Set before super().__init__ because _build_arena runs inside it.
        self._corridor_length = 12.0 * a if corridor_length is None else float(corridor_length)
        self._corridor_half_width = (
            1.0 * a if corridor_half_width is None else float(corridor_half_width)
        )
        self._start_margin = 1.0 * a if start_margin is None else float(start_margin)
        self._forward_reward_weight = float(forward_reward_weight)
        self._ctrl_cost_weight = float(ctrl_cost_weight)
        self._boundary_cost_weight = float(boundary_cost_weight)
        self._corridor_walls = bool(corridor_walls)
        self._draw_corridor_lines = bool(draw_corridor_lines)
        self._healthy_reward = float(healthy_reward)
        self._terminate_on_flip = bool(terminate_on_flip)
        self._terminate_on_goal = bool(terminate_on_goal)
        if not 0.0 <= float(start_y_jitter) <= 1.0:
            raise ValueError(f"start_y_jitter must be in [0, 1], got {start_y_jitter}")
        self._start_y_jitter = float(start_y_jitter)
        # Matches world.build_arena's goal cylinder, 0.3 * obstacle_scale, so
        # the capture radius is the thing you can actually see in the viewer.
        self._goal_radius = (
            0.3 * float(_ROBOT_CONFIGS[robot]["arena_scale"])
            if goal_radius is None
            else float(goal_radius)
        )
        self._goal_reward_weight = float(goal_reward_weight)
        self._goal_observation = bool(goal_observation)

        # Robots start at -L/2 + margin and run toward +L/2. Obstacles fill the
        # span between the start line and the far end.
        self._start_x = -0.5 * self._corridor_length + self._start_margin
        self._finish_x = 0.5 * self._corridor_length
        self._obstacle_x_lo = self._start_x + 0.75 * a  # clear runway to get moving
        self._obstacle_x_hi = self._finish_x - 0.25 * a

        super().__init__(robot=robot, lidar_groups=lidar_groups, **kwargs)
        # After super(), because it needs self.spec (hazard count) and
        # self._arena_scale, both of which GoToGoal.__init__ sets. Computed once
        # rather than per reset: it is a constant, and doing it here keeps it out
        # of every trace.
        self._hazard_lattice_xy = self._hazard_lattice()

    # -- arena -------------------------------------------------------------

    def _build_arena(self, mjSpec: mj.MjSpec) -> None:
        # Floor must cover the whole corridor plus a margin, otherwise the robot
        # runs off the edge of the plane and falls out of the world. The default
        # square floor is only 2.1 m half-extent -- less than one episode of
        # travel for a healthy ant.
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
            vase_mass=_ROBOT_CONFIGS[self._robot]["vase_mass"],
        )
        if self._corridor_walls:
            self._add_corridor_walls(mjSpec)

    def _add_corridor_walls(self, spec: mj.MjSpec) -> None:
        """Two static boxes that make the corridor PHYSICAL rather than a cost.

        ON BY DEFAULT since 2026-08-22. Pass corridor_walls=False (CLI:
        --no-corridor_walls) to reproduce anything measured before that date.

        WHY THIS REPLACES THE BOUNDARY COST. Leaving a corridor is a task-scope
        violation, not a safety violation, and charging it as `cost` put it in
        the same signal as hazard proximity -- so a constrained run optimised
        "stay in your lane" rather than "avoid the mines". It was also an
        unobservable constraint for most of this project's history. Worse, with
        the boundary cost simply removed, the constrained optimum becomes a ~4%
        longer path bowing out to |y| ~ 1.4 where no hazard is ever sampled
        (hazards are drawn uniformly in |y| <= corridor_half_width). A wall
        makes that exploit IMPOSSIBLE rather than merely expensive, and leaves
        `cost` as pure hazard proximity.

        BOXES, NOT VERTICAL PLANES, and that is the load-bearing choice.
        `max_geom_pairs=16` caps the broad phase PER COLLISION-TYPE GROUP. The
        ant's geoms against the floor already are the capsule-vs-plane group,
        and it is already truncated -- so wall planes would compete with the
        FLOOR for those 16 slots and could evict foot contacts whenever the
        robot came near a wall, i.e. sink it through the ground exactly when it
        is against a wall. Boxes land in the separate capsule-vs-box group,
        which on `minefield` is completely empty (no vases), so the walls
        cannot displace a single floor contact. Verify with
        `scripts/verify_contact_capping.py` rather than trusting this note.

        Static geoms on the worldbody: no joints, so `nq`/`nv` are unchanged
        and minefield keeps its speed (vases were expensive because they were
        FREE BODIES, not because they collided). Not added to any lidar group,
        so the observation width does not change and every existing checkpoint
        still loads.
        """
        a = self._arena_scale
        # Half-length matches the floor's own x half-extent, so there is no gap
        # to round the end of a wall through.
        half_len = 0.5 * self._corridor_length + 1.0 * a
        half_thick = 0.05 * a
        # Tall enough that the ant cannot climb or ballistically clear it: the
        # ant's torso stands at ~0.33 m at arena_scale 1, ant_gym's at ~0.67.
        half_height = 0.5 * a
        # VIEWING STYLE, off by default (2026-08-24). With
        # `draw_corridor_lines=True` the collider is drawn fully transparent and
        # a flat blue stripe is laid on the floor in its place, so the corridor
        # reads as a lane marking rather than a canyon -- the ant is small in
        # frame and two 0.5 m walls hid it from most camera angles.
        #
        # OFF IN TRAINING BECAUSE IT IS NOT FREE. The stripes never collide
        # (contype=0/conaffinity=0, the same mechanism that keeps the 20 hazards
        # out of the broad phase) and add no DOFs, so physics is BIT-IDENTICAL --
        # verified by sha256 over qpos after 40 steps, max abs diff 0.0. But
        # they raise ngeom 37 -> 39, so geom_xpos/geom_xmat carry two more rows
        # through kinematics every step, and that measured -1.27% on env-only
        # stepping (512 envs, laptop GPU, 2 alternated reps: -0.93% / -1.61%).
        # That is only ~0.4% of training/sps once diluted by the non-physics
        # two-thirds of a training step -- but it buys nothing during training,
        # where nothing is rendered at all.
        #
        # main.py turns it on; every other caller gets the fast path by default.
        wall_rgba = _WALL_RGBA_HIDDEN if self._draw_corridor_lines else _WALL_RGBA_SOLID
        half_thick_line = 0.02 * a
        half_height_line = 0.002 * a
        for sign, side in ((1.0, "left"), (-1.0, "right")):
            spec.worldbody.add_geom(
                name=f"corridor_wall_{side}",
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[half_len, half_thick, half_height],
                # Inner FACE sits exactly on +-corridor_half_width, so the wall
                # stands where the boundary cost used to be charged rather than
                # a wall-thickness away from it.
                pos=[
                    0.0,
                    sign * (self._corridor_half_width + half_thick),
                    half_height,
                ],
                rgba=wall_rgba,
            )
            if not self._draw_corridor_lines:
                continue
            # Centred ON the boundary, not on the wall's centre, so the stripe
            # marks where the collider's inner face actually is. Not in any
            # lidar group, and both cost-path id lists are built by explicit
            # NAME lookup, so this cannot leak into the cost signal.
            spec.worldbody.add_geom(
                name=f"corridor_line_{side}",
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[half_len, half_thick_line, half_height_line],
                pos=[0.0, sign * self._corridor_half_width, half_height_line],
                contype=0,
                conaffinity=0,
                # Azure, opaque. Deliberately NOT the hazards' pure blue,
                # which is [0, 0, 1] at alpha 0.25 -- a translucent disc
                # against an opaque line, so the two stay tellable apart.
                rgba=[0.15, 0.45, 0.95, 1.0],
            )

    def _hazard_lattice(self) -> np.ndarray:
        """Evenly spaced hazard centres. Deterministic, computed once.

        Replaces rejection sampling (2026-08-22, user's call). A random field
        makes the corridor a ROUTING problem -- find the gap this episode
        happens to have -- which needs long-range directional sensing. An even
        lattice makes it a GAIT problem: obstacles arrive at a fixed pitch, and
        the answer is a stride that misses them. That is why the hazard lidar
        ring is off by default on this task; see `lidar_groups`.

        Laid out as a STAGGERED lattice, alternate columns offset a quarter
        pitch either side of centre, so that fewer straight-line paths run clear
        from end to end than a plain grid would leave. Rows x columns is the factorisation of
        `num_hazards` with the squarest cell, so the pitch the robot meets in x
        is comparable to the spacing it must thread in y.

        Inset in y by the hazard radius, so no disc is half-buried in a corridor
        wall where only part of it is reachable.

        NOTE the layout is IDENTICAL every episode -- that is what "evenly
        spaced" means. The only per-episode variation is the robot's jittered
        start y, which changes its phase relative to the rows. If that turns out
        to be too little variation, offsetting the whole lattice by a per-episode
        random phase is a one-line change.
        """
        n = self.spec["hazards"].num_objects
        if n == 0:
            # A task may replace hazards entirely (envs/lasers.py). The loop
            # below divides by the column count, so this is a ZeroDivisionError
            # at CONSTRUCTION time rather than anything subtle.
            return np.empty((0, 2), dtype=float)
        r = self._hazard_size * self._arena_scale
        x_lo, x_hi = self._obstacle_x_lo, self._obstacle_x_hi
        y_lo = -self._corridor_half_width + r
        y_hi = self._corridor_half_width - r
        span_x, span_y = x_hi - x_lo, y_hi - y_lo

        rows, cols = 1, n
        best = None
        for rr in range(1, n + 1):
            if n % rr:
                continue
            cc = n // rr
            # squarest cell, scored in log space so 2x too wide and 2x too tall
            # are penalised equally
            score = abs(np.log((span_x / cc) / (span_y / rr)))
            if best is None or score < best:
                best, rows, cols = score, rr, cc

        dx, dy = span_x / cols, span_y / rows
        out = np.empty((n, 2), dtype=float)
        k = 0
        for i in range(cols):
            x = x_lo + dx * (i + 0.5)
            # SYMMETRIC stagger, -+ a quarter pitch, not a one-sided half
            # pitch. Shifting odd columns upward only pushes the top row onto
            # the wall inset and leaves the bottom open: measured at 20
            # hazards, that gave a 0.43 m disc-free lane hugging one wall.
            # Splitting the offset either side of centre keeps the pattern
            # symmetric and halves the widest free lane.
            shift = dy * (0.25 if i % 2 else -0.25)
            for j in range(rows):
                out[k] = (x, y_lo + dy * (j + 0.5) + shift)
                k += 1
        self._hazard_grid = (rows, cols)
        return out

    def _sample_corridor_layout(
        self, rng: jax.Array
    ) -> tuple[dict[str, list[tuple[int, jax.Array]]], jax.Array]:
        """Place the robot on the start line and scatter obstacles ahead of it.

        Not `world._sample_layout`: that samples every object uniformly from the
        single global `_EXTENTS` square, which would put a third of the hazards
        *behind* the start line where they can never be encountered, and would
        let the robot spawn at the far end with nothing left to run.

        Rejection sampling mirrors `world.draw_until_valid` but over this task's
        rectangle. The loop is bounded (`i < 100`) so it stays a fixed-cost
        `while_loop` under vmap; on giving up it returns the last candidate,
        matching the upstream behaviour of tolerating a rare overlap rather than
        failing a reset.
        """
        layout: dict[str, list[tuple[int, jax.Array]]] = {}
        n_haz = self.spec["hazards"].num_objects
        n_vase = self.spec["vases"].num_objects
        n_obs = n_haz + n_vase

        placed = jp.full((n_obs, 2), 1e3)
        keepouts = jp.zeros((n_obs,))

        def draw_one(key, keepout, placed, keepouts):
            def cond_fn(val):
                i, conflicted, *_ = val
                return jp.logical_and(i < 100, conflicted)

            def body_fn(val):
                i, _, _, k = val
                k, k_ = jax.random.split(k)
                xy = jax.random.uniform(
                    k_,
                    (2,),
                    minval=jp.array([self._obstacle_x_lo, -self._corridor_half_width]),
                    maxval=jp.array([self._obstacle_x_hi, self._corridor_half_width]),
                )
                return i + 1, placement_not_valid(xy, keepout, placed, keepouts), xy, k

            _, _, xy, _ = jax.lax.while_loop(
                cond_fn, body_fn, (0, True, jp.zeros((2,)), key)
            )
            return xy

        idx = 0
        # HAZARDS ARE NOT SAMPLED. They sit on the fixed lattice built by
        # _hazard_lattice(). They are still written into `placed`/`keepouts`,
        # because vases ARE still rejection-sampled and must keep clear of them.
        haz_keepout = self.spec["hazards"].keepout
        entries = []
        for xy_np in self._hazard_lattice_xy:
            xy = jp.asarray(xy_np)
            placed = placed.at[idx].set(xy)
            keepouts = keepouts.at[idx].set(haz_keepout)
            entries.append((idx, xy))
            idx += 1
        layout["hazards"] = entries

        vase_keepout = self.spec["vases"].keepout
        entries = []
        rng, sub = jax.random.split(rng)
        for key in jax.random.split(sub, n_vase):
            xy = draw_one(key, vase_keepout, placed, keepouts)
            placed = placed.at[idx].set(xy)
            keepouts = keepouts.at[idx].set(vase_keepout)
            entries.append((idx, xy))
            idx += 1
        layout["vases"] = entries

        # Robot on the start line. Jittered in y (so it cannot memorise one
        # lane) only if start_y_jitter > 0; the default is now a FIXED spawn at
        # y = 0. The rng is split either way so the key stream downstream is
        # identical whichever setting is used.
        rng, rk = jax.random.split(rng)
        half = self._start_y_jitter * self._corridor_half_width
        robot_xy = jp.array([
            self._start_x,
            jax.random.uniform(rk, (), minval=-half, maxval=half),
        ])
        layout["robot"] = [(0, robot_xy)]
        # Fixed beacon at the far end -- never moves, never respawns.
        layout["goal"] = [(0, jp.array([self._finish_x, 0.0]))]
        return layout, rng

    # -- goal sensing ------------------------------------------------------
    #
    # MEASURED 2026-08-15, and the reason this exists. Rolling the 1.68M-step
    # checkpoint for 16 episodes:
    #
    #     cost is 99.1% BOUNDARY, 0.8% hazard, 0.1% vase
    #     first leaves the corridor at decision 34 of 625 (~1 m of travel)
    #     |y| reaches a median of 11.6 m against a half-width of 1.0
    #     net +x median +1.6 m, range -7.8 to +11.9 -- half go BACKWARDS
    #
    # It walks; it just walks in an arbitrary direction. Two causes: nothing in
    # the observation correlated with where it was (the sensor block is
    # bit-identical at y=0, y=0.99 and y=3.0), and forward progress is
    # `speed * cos(theta)`, whose derivative at theta=0 is ZERO -- a gait 20
    # degrees off axis still earns 94%. The goal lidar ring that would have
    # supplied a heading was dead in all 10,000 observations sampled, because
    # the goal sits 11 m away against LIDAR_MAX_DIST = 2.0.
    #
    # So the direction to the goal is handed to the policy directly. This is
    # deliberately NOT a realisable sensor -- it is privileged state, and that
    # is a considered trade: sim-to-real is not this project's question, and a
    # policy that cannot tell which way to run cannot produce a meaningful
    # safety measurement either.
    #
    # WHY A DISTANCE-DELTA REWARD IS SAFE HERE, given it is exactly the reward
    # that made GoToGoal unlearnable: distance-to-a-point is convex with
    # curvature 1/d, so the penalty it puts on undirected motion scales with
    # 1/d. Measured, as a fraction of what a working gait earns per decision:
    #
    #     goal 0.8 m away (GoToGoal)     15.3%    <- exploration really is punished
    #     goal 11 m away (here)           0.02%   <- negligible
    #
    # At 11 m the distance function is locally almost linear, so this behaves
    # like the +x reward it supplements rather than like GoToGoal's trap. Do
    # NOT carry that conclusion over to a goal placed close to the robot.

    def _goal_xy(self, data: mjx.Data) -> jax.Array:
        # mocap_pos, never xpos: _reset_goal writes mocap_pos without re-running
        # forward kinematics, so xpos lags by a step. RunForward never respawns
        # its goal, but the habit is what keeps that class of bug out.
        return data.mocap_pos[self._goal_mocap_id][:2]

    def goal_distance(self, data: mjx.Data) -> jax.Array:
        return jp.linalg.norm(self._goal_xy(data) - data.site_xpos[self._robot_site_id][:2])

    def at_goal(self, data: mjx.Data) -> jax.Array:
        """Has the robot reached the goal? 1.0/0.0, usable as a `done` term.

        The radius matches the goal cylinder actually DRAWN in the arena
        (`0.3 * obstacle_scale` in world.build_arena), so "arrived" means what
        it looks like in the viewer rather than some invisible threshold. Not a
        knife edge either: the trained conditioned policy closed to within
        0.01-0.09 m of the centre on all eight bodies.
        """
        return (self.goal_distance(data) <= self._goal_radius).astype(jp.float32)

    def task_observations(self, data: mjx.Data) -> jax.Array | None:
        """[cos, sin] of the goal's bearing in the robot's frame, plus range.

        Bearing is taken from the torso's YAW ALONE rather than by rotating
        through the full orientation matrix, which is what `lidar.ego_xy`
        does. That matters: the lidar path leaves the robot's own height in the
        vector it rotates, so when the torso tilts the vertical component
        bleeds into the horizontal reading -- measured at 60 degrees of pitch,
        a target 2.00 m away registers as 1.58 m. Yaw-only is immune to that,
        and a heading signal that degrades exactly when the ant is falling over
        would be worst where it is needed most.
        """
        if not self._goal_observation:
            return None
        delta = self._goal_xy(data) - data.site_xpos[self._robot_site_id][:2]
        mat = data.xmat[self._robot_body_id].reshape(3, 3)
        yaw = jp.arctan2(mat[1, 0], mat[0, 0])
        rel = jp.arctan2(delta[1], delta[0]) - yaw
        # Range is normalised by the corridor length so it stays O(1) --
        # observations are NOT normalised anywhere in this stack
        # (normalize_observations=False in ppo/train.py), so raw metres would
        # enter the first layer an order of magnitude above every other input.
        return jp.array([
            jp.cos(rel), jp.sin(rel),
            jp.linalg.norm(delta) / self._corridor_length,
        ])

    def task_observation_size(self) -> int:
        return 3 if self._goal_observation else 0

    # -- reward / cost -----------------------------------------------------

    def get_reward(self, data, prev_data) -> jax.Array:
        """Progress along +x between two consecutive states.

        Takes the PREVIOUS `mjx.Data` rather than a carried scalar in
        `state.info`, and that is a correctness requirement, not a style choice.
        `BraxAutoResetWrapper` (mujoco_playground) restores `data` and `obs` from
        `first_state`/`first_obs` when an episode ends, but leaves every OTHER
        info key untouched. A carried `last_x` therefore survives the episode
        boundary, and the first step of each new episode measures its reward
        against the PREVIOUS episode's final x -- a spurious reward roughly the
        size of a whole episode's travel (~1 m) against a typical per-step
        reward of ~0.001 m. That is a 1000x outlier landing in PPO's advantage
        statistics once per episode.

        Measured before the fix, with a 3-decision episode: at the boundary
        `last_x` read -4.9746 while the reset data read -5.0000, and the next
        reward came out -0.02472 where the correct value was +0.0007 -- wrong by
        25x and of the opposite sign.

        `state.data` is restored by the wrapper, so differencing against it is
        correct across boundaries by construction, with no carried state to keep
        in sync.

        NOTE: GoToGoal had the identical bug via its carried `last_goal_dist`,
        for every run this project has ever done including the point baselines.
        It is now fixed there too (2026-08-11), by recomputing the previous
        distance from `state.data.mocap_pos`. Point results recorded before
        that date describe a slightly different reward function and are not
        directly comparable.
        """
        x_prev = prev_data.site_xpos[self._robot_site_id][0]
        x_new = data.site_xpos[self._robot_site_id][0]
        reward = (x_new - x_prev) * self._forward_reward_weight
        if self._goal_reward_weight:
            # Telescopes to (d_initial - d_final), i.e. "how much closer did it
            # get". Unlike the +x term this charges for lateral motion, which
            # is the point -- +x alone is flat in y, so going straight was never
            # preferred over drifting. Differenced against prev_data for the
            # same reason the +x term is: nothing position-like may live in
            # state.info, or every episode boundary pays a spurious reward.
            reward = reward + self._goal_reward_weight * (
                self.goal_distance(prev_data) - self.goal_distance(data)
            )
        if self._healthy_reward:
            reward = reward + self._healthy_reward * self.is_upright(data)
        return reward

    # -- posture -----------------------------------------------------------
    #
    # MEASURED 2026-08-12, and the reason these exist at all. Over 32 full
    # episodes of ant_gym, the torso is inverted for 94.4% of steps untrained
    # and 94.1% trained (1.5M steps) -- i.e. training moved it by 0.3 points,
    # because nothing in the reward ever mentioned staying upright and falling
    # had no consequence. 100% / 96.9% of episodes ENDED inverted. The +x
    # progress the 1.5M policy did make was made on its back.
    #
    # This is not a surprise once ant_gym is recognised as the standard Gym Ant
    # (0.911 kg on gear-150 actuators, torque/kg 164.7): a random policy on
    # actuators that strong throws itself over immediately. Gym's own Ant does
    # the same, which is exactly why Gym pairs it with a healthy bonus AND
    # termination. We adopted the morphology and left both behind.
    #
    # Orientation, not torso height, because height thresholds are tied to the
    # robot's scale and would have to be re-derived for every morphology the
    # search produces -- the same trap that made Gym's (0.2, 1.0) z-range
    # unusable here. `xmat[2,2]` is the world-z component of the torso's own
    # z axis: +1 perfectly upright, 0 on its side, -1 fully inverted. Measured
    # +1.0000 at reset.

    # Bonus is paid above this; termination happens below zero. The gap is
    # deliberate -- an ant on its side is not earning, but is not yet dead.
    _UPRIGHT_BONUS_THRESHOLD = 0.5  # ~60 degrees of tilt

    def _torso_up(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._robot_body_id].reshape(3, 3)[2, 2]

    def is_upright(self, data: mjx.Data) -> jax.Array:
        return (self._torso_up(data) > self._UPRIGHT_BONUS_THRESHOLD).astype(jp.float32)

    def is_flipped(self, data: mjx.Data) -> jax.Array:
        return (self._torso_up(data) < 0.0).astype(jp.float32)

    def get_cost(self, data) -> jax.Array:
        """Hazard/vase cost, plus a cost for leaving the corridor.

        Without the boundary term the safe optimum is to step out of the
        obstacle band and run in clean air -- full reward, zero cost -- which
        would make the constraint vacuous. See the module docstring for why this
        is a cost rather than a wall.
        """
        cost = super().get_cost(data)
        y = data.site_xpos[self._robot_site_id][1]
        out_of_bounds = jp.abs(y) > self._corridor_half_width
        return cost + out_of_bounds.astype(jp.float32) * self._boundary_cost_weight

    # -- episode -----------------------------------------------------------

    def reset(self, rng) -> State:
        data = mjx.make_data(self._mjx_model)
        layout, rng = self._sample_corridor_layout(rng)
        data, rng = self.update_positions(data, layout, rng)
        data = mjx.forward(self._mjx_model, data)

        # Deliberately carries NO position state. Anything stored here would
        # survive the episode boundary (BraxAutoResetWrapper restores only
        # `data` and `obs`), so reward and metrics are derived from `data`
        # instead -- see get_reward. Episode return already equals total +x
        # displacement by construction, verified exactly, so a separate
        # "distance" metric would be a redundant thing to keep correct.
        info = {
            "rng": rng,
            "cost": jp.zeros(()),
            # Recomputed fresh every step, never carried. A key present in only
            # one of reset/step is a pytree structure mismatch for the
            # auto-reset wrapper, so both must list all of them.
            "out_of_bounds": jp.zeros(()),
            "upright": self.is_upright(data),
            # ARRIVAL LATCH, and it has to live in the env rather than be read
            # off `state.data` afterwards, for two independent reasons:
            #
            #   1. `BraxAutoResetWrapper` swaps `data` for the reset pose on the
            #      very step `done` is set, so anything calling `at_goal()` after
            #      the fact inspects the START pose and reads 0% arrived. That
            #      exact mistake void-ed a whole eval pass of
            #      scripts/eval_morphology.py, which now has to pin
            #      terminate_on_goal=False to work around it.
            #   2. `CostEpisodeWrapper` scans `action_repeat` inner steps and
            #      does NOT break on `done`, so a single post-scan reading can
            #      miss an arrival that happened two inner steps earlier. A
            #      running `jp.maximum` cannot.
            #
            # Cleared by `EpisodeStatsWrapper` on the done step -- nothing else
            # clears it, because auto-reset leaves `info` untouched. Anyone
            # adding another consumer must not clear it a second time.
            "arrived": jp.zeros(()),
        }
        return State(data, self.get_obs(data), jp.zeros(()), jp.zeros(()), {}, info)

    def step(self, state: State, action: jax.Array) -> State:
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        scaled = (action + 1.0) / 2.0 * (upper - lower) + lower

        data = step(self._mjx_model, state.data, scaled, n_substeps=2)
        reward = self.get_reward(data, state.data)
        if self._ctrl_cost_weight:
            reward = reward - self._ctrl_cost_weight * jp.sum(jp.square(action))

        cost = self.get_cost(data)
        done = (jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(jp.float32)
        # Ending the episode on a flip is the larger half of this fix. Without
        # it an ant that goes over at step 50 still contributes 2450 further
        # transitions from a state where forward reward is unobtainable -- 94%
        # of every batch was that (see is_upright's note).
        if self._terminate_on_flip:
            done = jp.maximum(done, self.is_flipped(data))

        # Arrival termination. ON by default since 2026-08-18. Pass
        # terminate_on_goal=False (CLI: --no-terminate_on_goal) to reproduce
        # any result recorded before that date.
        #
        # WHY IT IS THE DEFAULT. Measured on the 50M conditioned run: the
        # ants reach the goal at decision 248 of 625 on average, so 60% of every
        # episode is spent milling around a goal that pays nothing further --
        # the reward telescopes, so once the distance is closed there is no more
        # to earn. Terminating there is ~2.5x more useful experience per
        # env-step.
        #
        # NOT paired with an arrival bonus, deliberately: a bonus would break
        # the telescoping property that makes episode return readable directly
        # as metres travelled, which is the one thing that has made this reward
        # debuggable.
        at_goal = self.at_goal(data)
        if self._terminate_on_goal:
            done = jp.maximum(done, at_goal)

        # Latched unconditionally, including when terminate_on_goal is off, so
        # "did this episode ever reach the goal" is answerable either way. One
        # extra float per lane; the distance it needs was already computed.
        state.info["arrived"] = jp.maximum(state.info["arrived"], at_goal)
        state.info["cost"] = cost
        state.info["upright"] = self.is_upright(data)
        state.info["out_of_bounds"] = (
            jp.abs(data.site_xpos[self._robot_site_id][1]) > self._corridor_half_width
        ).astype(jp.float32)

        return State(
            data=data,
            obs=self.get_obs(data),
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=state.info,
        )

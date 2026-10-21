"""LaserWeb: a dense, criss-crossing web of laser beams, like a heist-film hall.

WHY THIS TASK EXISTS (2026-10-15, user's call)
----------------------------------------------
`LaserField`'s beams are sparse, evenly spaced bars that each cross the corridor
once -- "extremely sparse, predictable and basically placed". Here every
layout is a fresh web: NUM_BEAMS straight lines, each through a random point of
the corridor volume in a random plan direction (across, diagonal or nearly
lengthwise) at a random slope, so beams cross each other, run along the hall,
and dip into the floor (the part below it is simply hidden, which reads as a
beam hitting the floor). Lines are LINE_HALF_LEN either side of that point,
long enough to reach across the whole floor, as in the film still the user
showed.

PASSABILITY IS CHECKED, NOT ASSUMED: a random web can wall the corridor off.
Each candidate layout is rasterised onto a 2D grid of ant positions (GRID_STEP
spacing, the usable corridor only); a cell is blocked when some beam passes
within CLEAR_RADIUS of it horizontally at a height below CLEAR_HEIGHT (the
standing ant's body top, 0.22-0.24 m measured, plus the cost margin and slack),
allowing for the beam's slope over that radius. A fixed-iteration flood fill
from the spawn decides whether the finish is reachable. CANDIDATES layouts are
drawn per reset and the first passable one kept; if none passes, the last is
lifted clear of the ant (counted, see `layout_fallback`). Conservative: beams
low enough to step over count as walls.

Everything else -- mocap capsules with no contacts, the exact per-limb gap,
the graded cost (1 at contact to 0 at COST_MARGIN), hazard_steps = beams
touched -- is LaserField's, unchanged.

OBSERVATION
-----------
LaserField's segment features assume a beam crosses the corridor once, which
no longer holds. Instead a CEILING MAP: a MAP_FWD x MAP_LEFT grid around the
torso in its yaw frame (more of it ahead), each point holding the height of the
lowest beam passing over it (within MAP_CAPTURE, which covers the gap between
grid points), /MAP_HMAX and capped at 1 when nothing is overhead; -1 marks a
point outside the corridor, where the episode ends. Like the minefield's foot
grid, but for the space the body moves through. Plus LaserField's per-limb gap
and direction to the nearest beam, the goal sensing and the lateral offset.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import _ROBOT_CONFIGS
from mjx_safety_gym.envs.laser_field import LaserField, _quat_z_axis, _quat_z_to


class LaserWeb(LaserField):
    """`LaserField` with a dense random web of beams and a ceiling-map input."""

    DEFAULT_NUM_BEAMS = 32
    LINE_HALF_LEN = 2.0
    FLOOR_FRACTION = 0.7
    FLOOR_ELEVATION = (0.35, 0.80)   # radians, ~20-46 degrees
    HIGH_HEIGHT = (0.32, 0.90)
    HIGH_ELEVATION = 0.12
    # web_style="ant" (2026-10-15, user: "not much to avoid"): the "tall" web
    # above is human-scale -- its steep floor beams spend only ~0.3-0.7 m below
    # the 0.22 m ant. This one lives in the ant's own height band: shallow LOW
    # SWEEPS through body height (route around), TRIPWIRES just off the floor
    # (step over -- the passability check counts that as a way through), and
    # low OVERHEAD beams just above the nominal ant (walk under; a taller
    # evolved body would hit them).
    ANT_FRACTIONS = (0.55, 0.20, 0.25)     # sweeps, tripwires, overhead
    SWEEP_HEIGHT = (0.05, 0.30)
    SWEEP_ELEVATION = 0.20
    TRIP_HEIGHT = (0.010, 0.035)
    OVER_HEIGHT = (0.30, 0.55)
    OVER_ELEVATION = 0.08
    STEP_HEIGHT = 0.04            # beams below this can be stepped over
    ANT_NUM_BEAMS = 16            # 15% of candidates pass at 16 x 2 m beams,
    ANT_HALF_LEN = 1.0            # so draw 64 per reset (fallback ~3e-5)
    ANT_CANDIDATES = 64
    CLEAR_RADIUS = 0.22
    CLEAR_HEIGHT = 0.28
    GRID_STEP = 0.10
    FILL_ITERS = 260
    CANDIDATES = 16
    MAP_FWD = np.linspace(-0.3, 1.5, 13)
    MAP_LEFT = np.linspace(-0.6, 0.6, 9)
    MAP_CAPTURE = 0.11            # > half the 0.15 m grid diagonal
    MAP_HMAX = 0.60
    # web_obs="voxels": a cubic grid around the torso (yaw frame, ahead-biased,
    # absolute heights) of distance to the nearest beam, clipped and /VOX_RANGE;
    # -1 outside the corridor. The user's 2026-10-15 suggestion.
    VOX_FWD = np.linspace(-0.3, 1.2, 11)
    VOX_LEFT = np.linspace(-0.5, 0.5, 7)
    VOX_Z = np.array([0.04, 0.12, 0.20, 0.28])
    VOX_RANGE = 0.30
    # web_obs="wide": "both" with a longer, wider ceiling map (more foresight).
    WIDE_FWD = np.linspace(-0.3, 2.4, 19)
    WIDE_LEFT = np.linspace(-0.9, 0.9, 13)
    # web_obs="objects": "both" plus the OBJ_BEAMS nearest beams as lines;
    # "wide_objects" = both additions.
    OBJ_BEAMS = 6

    def __init__(self, robot: str = "ant", num_beams: int | None = None,
                 web_obs: str = "ceiling", web_style: str = "tall", **kwargs):
        if web_style not in ("tall", "ant"):
            raise ValueError(f"web_style must be 'tall' or 'ant', got {web_style!r}")
        self._web_style = web_style
        if web_style == "ant":
            num_beams = self.ANT_NUM_BEAMS if num_beams is None else num_beams
        self._candidates = self.ANT_CANDIDATES if web_style == "ant" else self.CANDIDATES
        if web_obs not in ("ceiling", "voxels", "both", "wide", "objects", "wide_objects"):
            raise ValueError(f"unknown web_obs {web_obs!r}")
        self._web_obs = web_obs
        wide = web_obs in ("wide", "wide_objects")
        self._map_fwd = self.WIDE_FWD if wide else self.MAP_FWD
        self._map_left = self.WIDE_LEFT if wide else self.MAP_LEFT
        self._use_ceiling = web_obs != "voxels"
        self._use_voxels = web_obs != "ceiling"
        self._use_objects = web_obs in ("objects", "wide_objects")
        kwargs.pop("beam_obs", None)
        super().__init__(robot=robot, num_beams=num_beams, **kwargs)

    def _initial_half_len(self, a: float, half_w: float) -> float:
        return (self.ANT_HALF_LEN if self._web_style == "ant" else self.LINE_HALF_LEN) * a

    # -- layout ------------------------------------------------------------

    def _grid(self):
        a = self._arena_scale
        xs = np.arange(self._start_x - 0.3 * a, self._finish_x + 0.3 * a, self.GRID_STEP * a)
        w_u = self._corridor_half_width - self.USABLE_MARGIN * a * 0.8
        ys = np.arange(-w_u, w_u + 1e-6, self.GRID_STEP * a)
        return jp.asarray(xs), jp.asarray(ys)

    def _blocked(self, a, b):
        """(X, Y) cells an ant centred there cannot occupy without hitting a beam.

        A scan over beams, OR-ing one (X, Y) mask at a time: vectorised over
        beams the (lanes x candidates x cells x beams) intermediates were a
        4 GB matmul at 1024 lanes (OOM at the first reset, 2026-10-15)."""
        xs, ys = self._grid()
        px, py = jp.meshgrid(xs, ys, indexing="ij")
        r = self.CLEAR_RADIUS * self._arena_scale
        top = self.CLEAR_HEIGHT * self._arena_scale

        def one(blocked, beam):
            ai, bi = beam
            ab = bi - ai
            denom = jp.maximum(ab[0] ** 2 + ab[1] ** 2, 1e-9)
            t = jp.clip(((px - ai[0]) * ab[0] + (py - ai[1]) * ab[1]) / denom, 0.0, 1.0)
            d = jp.sqrt((px - ai[0] - t * ab[0]) ** 2 + (py - ai[1] - t * ab[1]) ** 2)
            h = ai[2] + t * ab[2]
            slack = jp.abs(ab[2]) / jp.sqrt(denom) * r       # slope over the radius
            floor_ = self.STEP_HEIGHT * self._arena_scale if self._web_style == "ant" else 0.0
            return blocked | ((d < r) & (h < top + slack) & (h > floor_ - slack)), None

        blocked, _ = jax.lax.scan(one, jp.zeros(px.shape, dtype=bool), (a, b))
        return blocked

    def _passable(self, a, b):
        blocked = self._blocked(a, b)
        xs, ys = self._grid()
        free = ~blocked
        start = free & (jp.abs(xs[:, None] - self._start_x) < 0.15 * self._arena_scale) & (jp.abs(ys[None, :]) < 0.15)
        goal = (xs >= self._finish_x - 0.05 * self._arena_scale)[:, None]

        def grow(_, r):
            # Shifts WITHOUT wrap-around: jp.roll would join the finish column
            # to the start and pass walled-off layouts.
            f = jp.zeros_like(r[:1]); g = jp.zeros_like(r[:, :1])
            n = (r | jp.concatenate([f, r[:-1]], 0) | jp.concatenate([r[1:], f], 0)
                 | jp.concatenate([g, r[:, :-1]], 1) | jp.concatenate([r[:, 1:], g], 1))
            return n & free

        reach = jax.lax.fori_loop(0, self.FILL_ITERS, grow, start)
        return jp.any(reach & goal)

    def _draw_lines(self, rng):
        """FLOOR_FRACTION of the beams rise from a floor point at a steep angle
        (the part below the floor is hidden, so each reads as a beam striking
        the floor); the rest run near-horizontally overhead. A steep beam only
        blocks the ~0.3-0.7 m where it is below body height, so the web can be
        dense and still leave ways through -- shallow low beams are walls."""
        a = self._arena_scale
        n = self._num_beams
        k = jax.random.split(rng, 8)
        floor = jax.random.uniform(k[0], (n,)) < self.FLOOR_FRACTION
        x = jax.random.uniform(k[1], (n,), minval=self._obstacle_x_lo, maxval=self._obstacle_x_hi)
        y = jax.random.uniform(k[2], (n,), minval=-self._corridor_half_width - 0.3 * a,
                               maxval=self._corridor_half_width + 0.3 * a)
        z_high = jax.random.uniform(k[3], (n,), minval=self.HIGH_HEIGHT[0] * a, maxval=self.HIGH_HEIGHT[1] * a)
        z = jp.where(floor, 0.0, z_high)
        psi = jax.random.uniform(k[4], (n,), minval=0.0, maxval=2.0 * jp.pi)
        steep = jax.random.uniform(k[5], (n,), minval=self.FLOOR_ELEVATION[0], maxval=self.FLOOR_ELEVATION[1])
        flat = jax.random.uniform(k[6], (n,), minval=-self.HIGH_ELEVATION, maxval=self.HIGH_ELEVATION)
        phi = jp.where(floor, steep, flat)
        u = jp.stack([jp.cos(phi) * jp.cos(psi), jp.cos(phi) * jp.sin(psi), jp.sin(phi)], -1)
        # Centre the capsule on its visible half: a floor beam's centre sits
        # half a length up the beam, so nothing is wasted underground.
        c = jp.stack([x, y, z], -1) + jp.where(floor[:, None], self._beam_half_len * u, 0.0)
        return c, u

    def _draw_lines_ant(self, rng):
        a = self._arena_scale
        n = self._num_beams
        k = jax.random.split(rng, 9)
        kind = jax.random.choice(k[0], 3, (n,), p=jp.asarray(self.ANT_FRACTIONS))
        x = jax.random.uniform(k[1], (n,), minval=self._obstacle_x_lo, maxval=self._obstacle_x_hi)
        y = jax.random.uniform(k[2], (n,), minval=-self._corridor_half_width, maxval=self._corridor_half_width)
        z_sweep = jax.random.uniform(k[3], (n,), minval=self.SWEEP_HEIGHT[0] * a, maxval=self.SWEEP_HEIGHT[1] * a)
        z_trip = jax.random.uniform(k[4], (n,), minval=self.TRIP_HEIGHT[0] * a, maxval=self.TRIP_HEIGHT[1] * a)
        z_over = jax.random.uniform(k[5], (n,), minval=self.OVER_HEIGHT[0] * a, maxval=self.OVER_HEIGHT[1] * a)
        z = jp.select([kind == 0, kind == 1], [z_sweep, z_trip], z_over)
        psi = jax.random.uniform(k[6], (n,), minval=0.0, maxval=jp.pi)
        e = jax.random.uniform(k[7], (n,), minval=-1.0, maxval=1.0)
        phi = jp.select([kind == 0, kind == 1], [e * self.SWEEP_ELEVATION, jp.zeros_like(e)], e * self.OVER_ELEVATION)
        u = jp.stack([jp.cos(phi) * jp.cos(psi), jp.cos(phi) * jp.sin(psi), jp.sin(phi)], -1)
        return jp.stack([x, y, z], -1), u

    def sample_beams(self, rng):
        keys = jax.random.split(rng, self._candidates)
        draw = self._draw_lines_ant if self._web_style == "ant" else self._draw_lines
        cs, us = jax.vmap(draw)(keys)                                    # (C, N, 3)
        half = self._beam_half_len
        # lax.map, not vmap: one candidate at a time keeps reset memory flat.
        ok = jax.lax.map(lambda cu: self._passable(cu[0] - half * cu[1], cu[0] + half * cu[1]), (cs, us))
        first = jp.argmax(ok)
        c, u = cs[first], us[first]
        # Fallback: lift the last candidate clear of the ant.
        lift = jp.where(ok.any(), 0.0, (self.CLEAR_HEIGHT + 0.6) * self._arena_scale)
        c = c.at[:, 2].add(lift)
        u = jp.where(ok.any(), u, u.at[:, 2].set(0.0) / jp.linalg.norm(u.at[:, 2].set(0.0), axis=-1, keepdims=True))
        return c, _quat_z_to(u)

    def layout_passable(self, rng) -> jax.Array:
        """(CANDIDATES,) whether each candidate of this key passed (to measure rates)."""
        keys = jax.random.split(rng, self._candidates)
        draw = self._draw_lines_ant if self._web_style == "ant" else self._draw_lines
        cs, us = jax.vmap(draw)(keys)
        half = self._beam_half_len
        return jax.vmap(lambda c, u: self._passable(c - half * u, c + half * u))(cs, us)

    # -- observation -------------------------------------------------------

    def _ceiling_map(self, data: mjx.Data) -> jax.Array:
        p, c, s = self._yaw_frame(data)
        fx, ly = np.meshgrid(self._map_fwd, self._map_left, indexing="ij")
        fx, ly = jp.asarray(fx.ravel()) * self._arena_scale, jp.asarray(ly.ravel()) * self._arena_scale
        px = p[0] + c * fx - s * ly
        py = p[1] + s * fx + c * ly
        pts = jp.stack([px, py], -1)                                       # (M, 2)
        a, b = self.beam_segments(data)
        a2, ab2 = a[:, :2], b[:, :2] - a[:, :2]
        t = (pts @ ab2.T - jp.sum(a2 * ab2, -1)) / jp.sum(ab2 * ab2, -1)
        t = jp.clip(t, 0.0, 1.0)                                           # (M, N)
        near = a2[None] + t[..., None] * ab2[None]
        d = jp.linalg.norm(pts[:, None, :] - near, axis=-1)
        h = a[:, 2] + t * (b[:, 2] - a[:, 2])
        hmax = self.MAP_HMAX * self._arena_scale
        over = (d < self.MAP_CAPTURE * self._arena_scale) & (h > -0.02 * self._arena_scale)
        ceiling = jp.min(jp.where(over, jp.clip(h, 0.0, hmax), hmax), axis=-1) / hmax
        return jp.where(jp.abs(py) > self._corridor_half_width, -1.0, ceiling)

    def _voxel_distances(self, data: mjx.Data) -> jax.Array:
        p, c, s = self._yaw_frame(data)
        fx, ly, z = np.meshgrid(self.VOX_FWD, self.VOX_LEFT, self.VOX_Z, indexing="ij")
        a_ = self._arena_scale
        fx, ly, z = (jp.asarray(v.ravel()) * a_ for v in (fx, ly, z))
        px = p[0] + c * fx - s * ly
        py = p[1] + s * fx + c * ly
        pts = jp.stack([px, py, z], -1)                                    # (V, 3)
        a, b = self.beam_segments(data)
        ab = b - a
        t = jp.clip((pts @ ab.T - jp.sum(a * ab, -1)) / jp.sum(ab * ab, -1), 0.0, 1.0)  # (V, N)
        near = a[None] + t[..., None] * ab[None]
        d = jp.min(jp.linalg.norm(pts[:, None, :] - near, axis=-1), axis=-1) - self._beam_radius
        rng_ = self.VOX_RANGE * a_
        return jp.where(jp.abs(py) > self._corridor_half_width, -1.0, jp.clip(d, 0.0, rng_) / rng_)

    def _object_observations(self, data: mjx.Data) -> jax.Array:
        """The OBJ_BEAMS nearest beams: nearest point (forward, left, height/0.2)
        and direction (forward, left, up) in the yaw frame, sign fixed so the
        forward component is >= 0."""
        p, c, s = self._yaw_frame(data)
        a, b = self.beam_segments(data)
        ab = b - a
        t = jp.clip(jp.sum((p - a) * ab, -1) / jp.sum(ab * ab, -1), 0.0, 1.0)
        near = a + ab * t[:, None]
        order = jp.argsort(jp.linalg.norm(near - p, axis=-1))[: self.OBJ_BEAMS]
        q, u = near[order], (ab / jp.linalg.norm(ab, axis=-1, keepdims=True))[order]
        dx, dy = q[:, 0] - p[0], q[:, 1] - p[1]
        uf, ul = c * u[:, 0] + s * u[:, 1], -s * u[:, 0] + c * u[:, 1]
        sign = jp.where(uf < 0, -1.0, 1.0)
        return jp.stack([jp.clip(c * dx + s * dy, -3, 3), jp.clip(-s * dx + c * dy, -3, 3),
                         q[:, 2] / (self.HEIGHT_SCALE * self._arena_scale),
                         sign * uf, sign * ul, sign * u[:, 2]], -1).reshape(-1)

    def task_observations(self, data: mjx.Data) -> jax.Array:
        base = super(LaserField, self).task_observations(data)          # goal + lateral
        parts = [] if base is None else [base]
        if self._use_ceiling:
            parts.append(self._ceiling_map(data))
        if self._use_voxels:
            parts.append(self._voxel_distances(data))
        if self._use_objects:
            parts.append(self._object_observations(data))
        parts.append(self._limb_observations(data))
        return jp.concatenate(parts)

    def task_observation_size(self) -> int:
        n_geoms = len(_ROBOT_CONFIGS[self._robot]["collision_geoms"])
        size = super(LaserField, self).task_observation_size() + 4 * n_geoms
        if self._use_ceiling:
            size += self._map_fwd.size * self._map_left.size
        if self._use_voxels:
            size += self.VOX_FWD.size * self.VOX_LEFT.size * self.VOX_Z.size
        if self._use_objects:
            size += 6 * self.OBJ_BEAMS
        return size


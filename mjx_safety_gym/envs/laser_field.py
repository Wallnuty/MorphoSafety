"""LaserField: cross a corridor of 3D laser beams without touching one.

WHY THIS TASK EXISTS (2026-10-15, user's call)
----------------------------------------------
`Lasers` is floor-level tripwires: full-width stripes where only a foot on the
ground pays, i.e. a striped minefield. This is the "movie" version: beams run
DIAGONALLY across the corridor in plan and SLOPE in height, so each beam is a
wall at one end and something to walk under at the other. The ant has to pick
where along each beam to cross, which makes it a routing-and-posture problem
rather than a stride-timing one. It is also the co-design counterweight to the
minefield: mines reward long legs, low beams punish tall bodies.

GEOMETRY
--------
`num_beams` beams on a jittered ~1 m pitch over the obstacle band. Beam k runs
from (x_k - s_k, -W_e, z_-) to (x_k + s_k, +W_e, z_+), W_e just outside the
corridor, so a beam cannot be walked around its end (leaving the corridor ends
the episode). The plan skew |s_k| <= MAX_SKEW and the pitch jitter are sized so
neighbouring beams never cross in plan. Heights z_-, z_+ are drawn per layout
and the beam's height varies linearly along it.

EVERY BEAM IS AN OBSTACLE: it must also block at least MIN_BLOCK metres of the
usable corridor below BLOCK_HEIGHT (where it hits the standing ant); without
this 46% of beams sat wholly overhead and constrained nothing.

PASSABILITY IS GUARANTEED, not hoped for: every beam must leave a stretch of at
least MIN_PASSAGE metres, inside the usable corridor (an ant-half-width in from
each edge), where it is at or above PASS_HEIGHT -- the nominal ant's body top
(0.22 m standing, 0.24 m max under random actions, measured 2026-10-15) plus
the cost margin and a little slack, so walking under it costs nothing. Heights
are rejection-sampled per beam (CANDIDATES draws, first that passes; fully-high
fallback). Beams at a few centimetres can also be stepped over (the torso
bottom stands ~0.10 m up) but that is not counted on.

Beams are mocap capsules with contype=conaffinity=0: no contacts, no DOFs, like
the hazards. Their pose lives in data.mocap_pos / mocap_quat, which the
auto-reset wrapper restores with the rest of `data`, so nothing about a layout
is carried in `info`.

COST
----
Per beam, the smallest surface gap to any robot collision geom (capsule core
segment vs beam axis, minus both radii -- exact for capsules and spheres).
`hazard_depths` turns that into the graded cost the rest of the stack already
knows: 1 at contact, falling linearly to 0 at COST_MARGIN of clearance, summed
over beams by GoToGoal.get_cost's hazard_cost_shape logic. `hazard_contacts`
(gap <= 0, an actual touch) feeds info["hazard_steps"], so eval's hazard_steps
and eval_design's hazard_free read "beams touched" here. EVERY body part counts,
not just grounded ones: a movie laser does not care which limb broke it.

OBSERVATION (task block, before the morphology genes)
-----------------------------------------------------
* goal sensing (3) and lateral offset (1), from RunForward;
* the OBS_BEAMS nearest beams, sorted by distance from the torso: the nearest
  point on the beam and the two points where it crosses the corridor edges,
  each as (forward, left) in the torso's yaw frame plus absolute height / 0.2;
* per collision geom: gap to the nearest beam (clipped at PROXIMITY_RANGE,
  /range) and the unit vector toward it in the yaw frame. Sensing where cost is
  CHARGED is the lesson of the minefield's foot grid.
The hazard lidar, foot grids and foot clearance are forced off: with no hazards
they would only carry zeros.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import _ROBOT_CONFIGS
from mjx_safety_gym.envs.run_forward import RunForward
from mjx_safety_gym.world import build_arena


def _closest_points(p1, q1, p2, q2):
    """Closest points between segments p1q1 and p2q2 (Ericson, RTCD 5.1.9).

    Broadcasts over leading axes. The first segment may be degenerate (a
    sphere's centre); the second (a beam) never is.
    """
    eps = 1e-9
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = jp.sum(d1 * d1, -1)
    e = jp.sum(d2 * d2, -1)
    f = jp.sum(d2 * r, -1)
    c = jp.sum(d1 * r, -1)
    b = jp.sum(d1 * d2, -1)
    denom = a * e - b * b
    s = jp.where(denom > eps, jp.clip((b * f - c * e) / jp.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    safe_a = jp.maximum(a, eps)
    s = jp.where(t < 0.0, jp.clip(-c / safe_a, 0.0, 1.0), jp.where(t > 1.0, jp.clip((b - c) / safe_a, 0.0, 1.0), s))
    t = jp.clip(t, 0.0, 1.0)
    degenerate = a <= eps
    s = jp.where(degenerate, 0.0, s)
    t = jp.where(degenerate, jp.clip(f / e, 0.0, 1.0), t)
    return p1 + d1 * s[..., None], p2 + d2 * t[..., None]


def _quat_z_to(u):
    """Shortest-arc quaternion (w, x, y, z) taking +z onto unit vector(s) u."""
    q = jp.stack([1.0 + u[..., 2], -u[..., 1], u[..., 0], jp.zeros_like(u[..., 0])], -1)
    return q / jp.linalg.norm(q, axis=-1, keepdims=True)


def _quat_z_axis(q):
    """Image of +z under quaternion(s) q (w, x, y, z)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jp.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


class LaserField(RunForward):
    """`RunForward` through randomly drawn, sloped, diagonal laser beams."""

    DEFAULT_NUM_BEAMS = 10
    BEAM_RADIUS = 0.005      # drawn radius; touching it is a contact
    COST_MARGIN = 0.03       # graded cost from this clearance (0) to contact (1)
    PASS_HEIGHT = 0.28       # beam axis height the nominal ant walks under for free
    # Heights where a beam crosses the USABLE corridor's edges: one low (it
    # blocks there), one high (walk under there), side drawn per beam. Drawn at
    # the usable edges rather than the beam ends because the ends lie outside
    # where the ant can be: drawn there, a beam could not both block and leave
    # a passage inside the usable 1.5 m (70% fell back, 2026-10-15).
    LOW_RANGE = (0.03, 0.18)
    HIGH_RANGE = (0.32, 0.60)
    MAX_SKEW = 0.30          # x offset of each beam end from its centre
    PITCH_JITTER = 0.10      # fraction of the pitch
    EDGE_OVERHANG = 0.10     # beam ends sit this far outside the corridor
    USABLE_MARGIN = 0.25     # an ant half-width plus slack in from each edge
    MIN_PASSAGE = 0.50       # walk-under stretch every beam must leave
    BLOCK_HEIGHT = 0.20      # below this a beam hits the standing ant
    MIN_BLOCK = 0.30         # ... and every beam must block this much corridor
    CANDIDATES = 32
    OBS_BEAMS = 4
    PROXIMITY_RANGE = 0.30
    HEIGHT_SCALE = 0.20
    # beam_obs="profile": the next PROFILE_BEAMS beams AHEAD (by where they
    # cross the robot's own y), each sampled at PROFILE_POINTS lateral
    # positions across the usable corridor: (x - robot x, height) per point --
    # a 1D height map per beam, saying directly where it can be walked under.
    PROFILE_BEAMS = 3
    PROFILE_POINTS = 7

    def __init__(self, robot: str = "ant", num_beams: int | None = None,
                 beam_obs: str = "segments", **kwargs):
        if beam_obs not in ("segments", "profile"):
            raise ValueError(f"beam_obs must be 'segments' or 'profile', got {beam_obs!r}")
        self._beam_obs = beam_obs
        self._num_beams = self.DEFAULT_NUM_BEAMS if num_beams is None else int(num_beams)
        if self._num_beams < 1:
            raise ValueError(f"LaserField needs at least one beam, got {self._num_beams}")
        a = float(_ROBOT_CONFIGS[robot]["arena_scale"])
        half_w = kwargs.get("corridor_half_width")
        half_w = 1.0 * a if half_w is None else float(half_w)
        # Fixed capsule half-length, long enough for the most skewed, most
        # sloped beam to reach past both corridor edges.
        self._beam_half_len_init = self._initial_half_len(a, half_w)
        # No hazards: their lidar ring, foot grids and clearance would be zeros.
        kwargs.update(lidar_groups=(), foot_hazard_grid=0, foot_hazard_rings=0, foot_obstacle_obs=False)
        super().__init__(robot=robot, num_hazards=0, num_vases=0, **kwargs)

    def _initial_half_len(self, a: float, half_w: float) -> float:
        """Capsule half-length, fixed at model build: long enough for the most
        skewed, most sloped beam to reach past both corridor edges."""
        reach = half_w + self.EDGE_OVERHANG * a
        usable = half_w - self.USABLE_MARGIN * a
        dz = (self.HIGH_RANGE[1] - self.LOW_RANGE[0]) * a * reach / usable   # steepest beam, end to end
        return 1.02 * float(np.sqrt(reach**2 + (self.MAX_SKEW * a) ** 2 + (0.5 * dz) ** 2))

    # -- arena -------------------------------------------------------------

    def _add_beams(self, spec: mj.MjSpec) -> None:
        a = self._arena_scale
        for i in range(self._num_beams):
            spec.add_material(name=f"beam_{i}_mat", emission=0.8)
            beam = spec.worldbody.add_body(
                name=f"beam_{i}", mocap=True, pos=[0.0, 0.0, 0.5 * a],
                # Lying across the corridor (+z -> +y) until the first reset.
                quat=[0.70710678, -0.70710678, 0.0, 0.0],
            )
            beam.add_geom(
                name=f"beam_{i}_geom",
                type=mj.mjtGeom.mjGEOM_CAPSULE,
                size=[self.BEAM_RADIUS * a, self._beam_half_len_init, 0.0],
                material=f"beam_{i}_mat",
                rgba=[1.0, 0.08, 0.08, 0.9],
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
            hazard_height=self._ground_contact_eps,
        )
        self._add_beams(mjSpec)
        # Repeated from RunForward._build_arena, which this overrides (the trap
        # Minefield's and Lasers' docstrings describe).
        self._add_corridor_walls(mjSpec)
        if self._finish_line:
            self._add_finish_line(mjSpec)

    def _post_init(self, *args, **kwargs):
        super()._post_init(*args, **kwargs)
        m = self._mj_model
        self._beam_mocap_ids = jp.array([m.body(f"beam_{i}").mocapid[0] for i in range(self._num_beams)])
        # Read back off the compiled model, so cost and drawing cannot drift.
        self._beam_half_len = float(m.geom("beam_0_geom").size[1])
        self._beam_radius = float(m.geom("beam_0_geom").size[0])

    # -- layout ------------------------------------------------------------

    def _passage_length(self, z_minus, z_plus, height=None):
        """Length of the usable-corridor stretch where a beam is >= `height`
        (default PASS_HEIGHT)."""
        a = self._arena_scale
        w_e = self._corridor_half_width + self.EDGE_OVERHANG * a
        w_u = self._corridor_half_width - self.USABLE_MARGIN * a
        h = (self.PASS_HEIGHT if height is None else height) * a
        slope = (z_plus - z_minus) / (2.0 * w_e)                      # dz/dy
        flat = jp.abs(slope) < 1e-9
        y_star = -w_e + (h - z_minus) / jp.where(flat, 1.0, slope)   # z(y*) == h
        y_star = jp.clip(y_star, -w_u, w_u)
        up = w_u - y_star            # slope > 0: passable for y >= y*
        down = y_star + w_u          # slope < 0: passable for y <= y*
        return jp.where(flat, jp.where(z_minus >= h, 2.0 * w_u, 0.0), jp.where(slope > 0, up, down))

    def sample_beams(self, rng):
        """(N, 3) centres and (N, 4) quaternions of one layout's beams."""
        a = self._arena_scale
        n = self._num_beams
        lo, hi = self._obstacle_x_lo, self._obstacle_x_hi
        pitch = (hi - lo) / n
        k_jit, k_skew, k_z = jax.random.split(rng, 3)
        xc = lo + pitch * (jp.arange(n) + 0.5)
        xc = xc + jax.random.uniform(k_jit, (n,), minval=-1.0, maxval=1.0) * self.PITCH_JITTER * pitch
        skew = jax.random.uniform(k_skew, (n,), minval=-1.0, maxval=1.0) * self.MAX_SKEW * a
        w_e = self._corridor_half_width + self.EDGE_OVERHANG * a
        w_u = self._corridor_half_width - self.USABLE_MARGIN * a
        k_lo, k_hi, k_side = jax.random.split(k_z, 3)
        shape = (n, self.CANDIDATES)
        low = jax.random.uniform(k_lo, shape, minval=self.LOW_RANGE[0] * a, maxval=self.LOW_RANGE[1] * a)
        high = jax.random.uniform(k_hi, shape, minval=self.HIGH_RANGE[0] * a, maxval=self.HIGH_RANGE[1] * a)
        low_at_minus = jax.random.bernoulli(k_side, 0.5, (n,))[:, None]
        h_minus = jp.where(low_at_minus, low, high)          # at y = -w_u
        h_plus = jp.where(low_at_minus, high, low)           # at y = +w_u
        slope = (h_plus - h_minus) / (2.0 * w_u)
        z_minus = h_minus - slope * (w_e - w_u)              # extended to the beam ends
        z_plus = h_plus + slope * (w_e - w_u)
        # Every beam must leave a walk-under passage AND block part of the
        # corridor -- a beam the ant can ignore is not an obstacle.
        blocked = 2.0 * w_u - self._passage_length(z_minus, z_plus, self.BLOCK_HEIGHT)
        ok = ((self._passage_length(z_minus, z_plus) >= self.MIN_PASSAGE * a)
              & (blocked >= self.MIN_BLOCK * a))
        first = jp.argmax(ok, axis=1)[:, None]
        pick = lambda v: jp.take_along_axis(v, first, axis=1)[:, 0]
        # Fallback (0.08% of beams over 5000 layouts): the steepest beam, which passes.
        steep = (self.HIGH_RANGE[1] - self.LOW_RANGE[0]) * a / (2.0 * w_u)
        fb_minus = jp.where(low_at_minus[:, 0], self.LOW_RANGE[0] * a - steep * (w_e - w_u),
                            self.HIGH_RANGE[1] * a + steep * (w_e - w_u))
        fb_plus = jp.where(low_at_minus[:, 0], self.HIGH_RANGE[1] * a + steep * (w_e - w_u),
                           self.LOW_RANGE[0] * a - steep * (w_e - w_u))
        any_ok = ok.any(axis=1)
        z = jp.stack([jp.where(any_ok, pick(z_minus), fb_minus), jp.where(any_ok, pick(z_plus), fb_plus)], -1)
        e_minus = jp.stack([xc - skew, jp.full((n,), -w_e), z[:, 0]], -1)
        e_plus = jp.stack([xc + skew, jp.full((n,), w_e), z[:, 1]], -1)
        u = e_plus - e_minus
        u = u / jp.linalg.norm(u, axis=-1, keepdims=True)
        return 0.5 * (e_minus + e_plus), _quat_z_to(u)

    def update_positions(self, data, layout, rng):
        data, rng = super().update_positions(data, layout, rng)
        rng, key = jax.random.split(rng)
        pos, quat = self.sample_beams(key)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._beam_mocap_ids].set(pos),
            mocap_quat=data.mocap_quat.at[self._beam_mocap_ids].set(quat),
        )
        return data, rng

    # -- geometry ----------------------------------------------------------

    def beam_segments(self, data: mjx.Data):
        """(N, 3) start and end points of every beam's axis."""
        c = data.mocap_pos[self._beam_mocap_ids]
        u = _quat_z_axis(data.mocap_quat[self._beam_mocap_ids])
        return c - self._beam_half_len * u, c + self._beam_half_len * u

    def _gap_matrix(self, data: mjx.Data):
        """(G, N) surface gap from each robot geom to each beam, and the two
        closest points, (G, N, 3) each: on the geom's core, on the beam axis."""
        radius, half_len = self._robot_geom_extent()
        ids = self._robot_collision_geom_ids_arr
        gp = data.geom_xpos[ids]
        ga = data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        p1 = (gp - ga * half_len[:, None])[:, None, :]
        q1 = (gp + ga * half_len[:, None])[:, None, :]
        a, b = self.beam_segments(data)
        c1, c2 = _closest_points(p1, q1, a[None], b[None])
        gap = jp.linalg.norm(c1 - c2, axis=-1) - radius[:, None] - self._beam_radius
        return gap, c1, c2

    def beam_gaps(self, data: mjx.Data) -> jax.Array:
        """(N,) smallest surface gap from the robot to each beam."""
        return jp.min(self._gap_matrix(data)[0], axis=0)

    # GoToGoal.get_cost and RunForward.step read these, so cost shape, the
    # hazard_steps metric and eval's hazard_free all apply to beams unchanged.
    def hazard_depths(self, data: mjx.Data) -> jax.Array:
        margin = self.COST_MARGIN * self._arena_scale
        return jp.clip(1.0 - jp.maximum(self.beam_gaps(data), 0.0) / margin, 0.0, 1.0)

    def hazard_contacts(self, data: mjx.Data) -> jax.Array:
        return self.beam_gaps(data) <= 0.0

    # -- observation -------------------------------------------------------

    def _yaw_frame(self, data: mjx.Data):
        mat = data.xmat[self._robot_body_id].reshape(3, 3)
        yaw = jp.arctan2(mat[1, 0], mat[0, 0])
        return data.xpos[self._robot_body_id], jp.cos(yaw), jp.sin(yaw)

    def _beam_observations(self, data: mjx.Data) -> jax.Array:
        p, c, s = self._yaw_frame(data)
        a, b = self.beam_segments(data)
        ab = b - a
        t = jp.clip(jp.sum((p - a) * ab, -1) / jp.sum(ab * ab, -1), 0.0, 1.0)
        near = a + ab * t[:, None]
        order = jp.argsort(jp.linalg.norm(near - p, axis=-1))[: self.OBS_BEAMS]
        w = self._corridor_half_width
        ty_m = (-w - a[:, 1]) / ab[:, 1]     # ab_y > 0: beams run -y -> +y
        ty_p = (w - a[:, 1]) / ab[:, 1]
        pts = jp.stack([near, a + ab * ty_m[:, None], a + ab * ty_p[:, None]], 1)[order]  # (K, 3, 3)
        dx, dy = pts[..., 0] - p[0], pts[..., 1] - p[1]
        fwd = jp.clip(c * dx + s * dy, -3.0, 3.0)
        left = jp.clip(-s * dx + c * dy, -3.0, 3.0)
        height = pts[..., 2] / (self.HEIGHT_SCALE * self._arena_scale)
        return jp.stack([fwd, left, height], -1).reshape(-1)

    def _profile_observations(self, data: mjx.Data) -> jax.Array:
        a, b = self.beam_segments(data)
        ab = b - a
        p = data.site_xpos[self._robot_site_id]
        w_u = self._corridor_half_width - self.USABLE_MARGIN * self._arena_scale

        def at_y(y):                       # (N,) points where each beam crosses y
            t = (y - a[:, 1]) / ab[:, 1]
            return a + ab * t[:, None]

        ahead = at_y(p[1])[:, 0] - p[0]    # along-track distance at the robot's y
        # Beams more than 0.3 m behind are done with; the rest in order ahead.
        order = jp.argsort(jp.where(ahead < -0.3 * self._arena_scale, 1e3 + ahead, ahead))
        order = order[: self.PROFILE_BEAMS]
        ys = jp.linspace(-w_u, w_u, self.PROFILE_POINTS)
        pts = jax.vmap(at_y)(ys)[:, order]                        # (P, K, 3)
        fwd = jp.clip((pts[..., 0] - p[0]) / self._arena_scale, -3.0, 3.0)
        height = pts[..., 2] / (self.HEIGHT_SCALE * self._arena_scale)
        return jp.stack([fwd, height], -1).transpose(1, 0, 2).reshape(-1)

    def _limb_observations(self, data: mjx.Data) -> jax.Array:
        gap, c1, c2 = self._gap_matrix(data)
        j = jp.argmin(gap, axis=1)
        rows = jp.arange(gap.shape[0])
        g = gap[rows, j]
        v = c2[rows, j] - c1[rows, j]
        v = v / jp.maximum(jp.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
        _, c, s = self._yaw_frame(data)
        rng_ = self.PROXIMITY_RANGE * self._arena_scale
        return jp.stack([
            jp.clip(g, 0.0, rng_) / rng_,
            c * v[:, 0] + s * v[:, 1],
            -s * v[:, 0] + c * v[:, 1],
            v[:, 2],
        ], -1).reshape(-1)

    def task_observations(self, data: mjx.Data) -> jax.Array:
        base = super().task_observations(data)
        parts = [] if base is None else [base]
        beams = self._beam_observations(data) if self._beam_obs == "segments" else self._profile_observations(data)
        parts += [beams, self._limb_observations(data)]
        return jp.concatenate(parts)

    def task_observation_size(self) -> int:
        n_geoms = len(_ROBOT_CONFIGS[self._robot]["collision_geoms"])
        beams = 9 * self.OBS_BEAMS if self._beam_obs == "segments" else 2 * self.PROFILE_BEAMS * self.PROFILE_POINTS
        return super().task_observation_size() + beams + 4 * n_geoms

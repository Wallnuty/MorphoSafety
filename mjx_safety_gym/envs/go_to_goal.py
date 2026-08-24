from typing import Mapping, Optional, Sequence, Union
import warnings

import jax
from mujoco import mjx
import mujoco as mj
import jax.numpy as jp
from importlib.resources import files
import numpy as np

from ml_collections import config_dict
from mujoco_playground._src import mjx_env as playground_mjx_env

from mjx_safety_gym.collision import geoms_colliding
from mjx_safety_gym.mjx_env import State, step
import mjx_safety_gym.lidar as lidar
from mjx_safety_gym.morphology import NUM_GENES
from mjx_safety_gym.world import (
    ObjectSpec,
    _sample_layout,
    apply_integrator,
    build_arena,
    draw_until_valid,
)

_XML_DIR = files("mjx_safety_gym.envs.xmls")
_ROBOT_XMLS = {
    "point": "point.xml",
    "ant": "ant.xml",
    "ant_gym": "ant_gym.xml",
}

# Per-robot morphology description. `collision_geoms` are the robot geoms checked
# against obstacles for the cost function. Placement is driven by either a set of
# `slide_joints` (2-DOF planar robot) or a single `free_joint` (6-DOF, 7 qpos).
_ROBOT_CONFIGS = {
    "point": {
        # A sliding puck has no feet and no step-on semantics, so per-foot
        # clearance is meaningless for it.
        "foot_geoms": [],
        "collision_geoms": ["robot", "pointarrow"],
        "slide_joints": ["x", "y"],
        "free_joint": None,
        "spawn_height": None,
        # Multiplies every arena dimension -- obstacle geom sizes, placement
        # keepouts and drop heights. Exists because obstacle sizes were chosen
        # for the ~0.1 m point robot and are meaningless for a robot of a
        # different scale: ant_gym has a 3.6 m leg span, so unscaled 0.2 m
        # hazards would sit under its feet as rounding errors. Keep at 1.0 for
        # any robot whose size matches the original safety-gym arena.
        "arena_scale": 1.0,
        # Explicit vase mass, or None to let geom density decide (see
        # world.build_arena). Only needed where arena_scale != 1.
        "vase_mass": None,
        # Proprioceptive sensors appended to the observation (beyond BASE_SENSORS).
        "extra_sensors": [],
    },
    "ant": {
        # The geoms that actually bear weight, i.e. what `--hazard_step_on`
        # charges cost for. Used by foot_obstacle_observations to tell the
        # policy where ITS FEET are relative to obstacles -- the lidar ring is
        # computed from the torso alone and cannot express this.
        "foot_geoms": [
            "left_ankle_geom", "right_ankle_geom",
            "third_ankle_geom", "fourth_ankle_geom",
        ],
        "collision_geoms": [
            "torso_geom",
            "aux_1_geom", "left_leg_geom", "left_ankle_geom",
            "aux_2_geom", "right_leg_geom", "right_ankle_geom",
            "aux_3_geom", "back_leg_geom", "third_ankle_geom",
            "aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom",
        ],
        "slide_joints": None,
        "free_joint": "root",
        # Torso height at which the ant spawns upright (matches the body pos in ant.xml).
        "spawn_height": 0.18,
        "arena_scale": 1.0,
        "vase_mass": None,
        # Joint angles + joint velocities so the policy can perceive its own legs.
        "extra_sensors": [
            "hip_1", "ankle_1", "hip_2", "ankle_2",
            "hip_3", "ankle_3", "hip_4", "ankle_4",
            "hip_1_vel", "ankle_1_vel", "hip_2_vel", "ankle_2_vel",
            "hip_3_vel", "ankle_3_vel", "hip_4_vel", "ankle_4_vel",
        ],
    },
}

# ant_gym is ant.xml at 4x length scale with the ankle density hack removed --
# i.e. the standard Gym/Brax ant, which every geom/body/sensor/joint name is
# shared with, so this entry is ant's with only the two scale-dependent numbers
# changed. See envs/xmls/ant_gym.xml for the derivation and measurements.
_ROBOT_CONFIGS["ant_gym"] = {
    **_ROBOT_CONFIGS["ant"],
    "spawn_height": 0.75,
    "arena_scale": 4.0,
    # ~5% of this robot's 0.911 kg: knockable, but not free to barge through.
    # Without this a 4x-scaled vase weighs 2.56 kg and is effectively a wall.
    "vase_mass": 0.05,
}

Observation = Union[jax.Array, Mapping[str, jax.Array]]
BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=512,
      render_width=64,
      render_height=64,
      enabled_geom_groups=[0, 1, 2],
      use_rasterizer=False,
      history=3,
  )

def _segment_point_distance_2d(
    points: jax.Array, seg_a: jax.Array, seg_b: jax.Array
) -> jax.Array:
    """Distance in the xy-plane from each point to each line segment.

    `points` is (P, 2); `seg_a`/`seg_b` are (S, 2) segment endpoints. Returns
    (P, S). Degenerate segments (a == b, i.e. spheres) collapse to point-point
    distance, so a single code path covers both spheres and capsules.
    """
    ab = seg_b - seg_a  # (S, 2)
    ap = points[:, None, :] - seg_a[None, :, :]  # (P, S, 2)
    denom = jp.sum(ab * ab, axis=-1)  # (S,)
    # Guard the divide twice: once for the value, once so the *unselected*
    # branch is still finite (a NaN there would poison reverse-mode grads).
    safe_denom = jp.where(denom > 0.0, denom, 1.0)
    t = jp.where(denom > 0.0, jp.sum(ap * ab[None], axis=-1) / safe_denom, 0.0)
    t = jp.clip(t, 0.0, 1.0)  # (P, S)
    closest = seg_a[None] + t[..., None] * ab[None]  # (P, S, 2)
    return jp.linalg.norm(points[:, None, :] - closest, axis=-1)


def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
  """
  Intensity-weigh the colors.
  This expects the input to have the channels in the last dim.
  Values from ITU-R BT.60 standard for RGB to grayscale conversion.
  """
  r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

class GoToGoal(playground_mjx_env.MjxEnv):
    """GoToGoal safety task.

    Subclasses mujoco_playground's MjxEnv (without calling its __init__, matching
    the ss2r reference implementation) so it can be wrapped by
    mujoco_playground's brax-compatible training wrappers.
    """

    def __init__(
        self,
        robot: str = "point",
        vision: bool = False,
        vision_config=None,
        morphology_conditioning: bool = False,
        integrator: str | None = None,
        num_hazards: int = 10,
        num_vases: int = 10,
        lidar_groups: Optional[Sequence[str]] = None,
        hazard_size: float = 0.16,
        hazard_step_on: bool = True,
        foot_obstacle_obs: bool = False,
        ground_contact_eps: float | None = None,
    ):
        if robot not in _ROBOT_XMLS:
            raise ValueError(
                f"Unknown robot {robot!r}. Available: {sorted(_ROBOT_XMLS)}"
            )
        self._robot = robot
        self._xml_path = _XML_DIR / _ROBOT_XMLS[robot]

        # Which lidar rings actually enter the observation. Each is
        # NUM_LIDAR_BINS wide, so dropping one narrows the observation by 16.
        #
        # Defaults to all three here, which keeps GoToGoal byte-identical --
        # its goal genuinely moves and comes into lidar range, so its goal ring
        # carries information. RunForward narrows it (see that file): MEASURED
        # over 10,000 real observations, its goal ring was live in 0 of them,
        # because the goal is parked 11 m away (44 m for ant_gym) against
        # LIDAR_MAX_DIST = 2.0.
        #
        # The `object` ring is dead in EVERY task -- `_object_body_ids` is
        # assigned `[]` in _post_init and never written to. It is the slot
        # safety-gym uses for the Push task's box, which this repo does not
        # implement. It USED to be kept in the default purely so the goal task's
        # width did not move; DROPPED 2026-08-22 on the user's call, after an
        # audit measured it at 0/16 live dims (max value 0.0000) over ~7,700
        # observations while obstacle and goal both read 16/16.
        #
        # THIS NARROWS THE GOAL TASK BY 16 (ant 76 -> 60, point 60 -> 44) and so
        # invalidates every pre-2026-08-22 goal-task checkpoint. They remain
        # loadable by passing lidar_groups=("obstacle","goal","object")
        # explicitly -- which is what train_ppo._LEGACY_LIDAR_GROUPS is, and it
        # is already one of the configurations build_env_for_checkpoint tries.
        # run/minefield are UNAFFECTED: they pass ("obstacle",) explicitly and
        # dropped both other rings back on 2026-08-15.
        groups = list(
            lidar.DEFAULT_LIDAR_GROUPS if lidar_groups is None else lidar_groups
        )
        unknown = [g for g in groups if g not in lidar.LIDAR_GROUPS]
        if unknown:
            raise ValueError(
                f"unknown lidar group(s) {unknown}; available: {lidar.LIDAR_GROUPS}"
            )
        # AN EMPTY TUPLE IS LEGAL since 2026-08-22, and used to raise
        # "the robot would be blind". It is a deliberate configuration for the
        # corridor tasks: with hazards on a fixed even lattice, their positions
        # are a function of the robot's own position, which goal bearing + range
        # + the magnetometer's absolute yaw already determine. The task then
        # asks for a GAIT matched to the obstacle pitch rather than for
        # long-range route planning. See RunForward's `lidar_groups` default.
        self._lidar_groups = tuple(groups)
        # HAZARD RADIUS before arena scaling. 0.16 since 2026-08-23 (user's
        # call), i.e. 0.8x safety-gym's 0.2. The radius has moved three times:
        # 0.14 (0.7x) -> 0.18 (0.9x) on 2026-08-22, then 0.16 here. Smaller
        # discs make the corridor a field to be threaded by foot placement
        # rather than a wall to be routed around; 0.14 left more free width
        # than intended, blocking only 56% of the corridor at 20 hazards.
        # Every cost number measured at a different radius is on a different
        # scale; pass hazard_size=0.2 to reproduce safety-gym's.
        self._hazard_size = float(hazard_size)
        # DEFAULT ON since 2026-08-22 (user's call). Hazard cost is charged only
        # while a robot geom is ON THE GROUND inside the hazard disc. The old
        # test was purely 2D, so a foot swung THROUGH THE AIR over a mine cost
        # exactly as much as standing on it -- it charged the ant for its gait
        # rather than for where it put its weight.
        #
        # THIS CHANGES WHAT `cost` MEANS, so every cost number measured before
        # this date is on a different scale. Measured on the 50M unconstrained
        # policy, 128 episodes, both tests evaluated along the SAME trajectory:
        # 465.2 (2D) vs 335.1 (step-on), paired difference -130.1 +- 4.4, i.e.
        # **28% of the old cost was pass-over**. Superseded by this flip:
        # the 452.5 hazard baseline and the --safety_budget 250 derived from it.
        # New reference: 1.127 cost/decision unconstrained, so a 25% target is
        # --safety_budget 176.
        #
        # Pass hazard_step_on=False (CLI: --no-hazard_step_on) to reproduce
        # anything measured before this date.
        self._hazard_step_on = bool(hazard_step_on)
        self._foot_obstacle_obs = bool(foot_obstacle_obs)

        # Keepouts scale with the arena, otherwise a large robot spawns
        # overlapping the obstacles it is supposed to avoid.
        self._arena_scale = float(_ROBOT_CONFIGS[robot]["arena_scale"])
        # How far a geom's lowest point may sit above the floor and still count
        # as "on the ground", for --hazard_step_on. Scales with the robot,
        # because a tolerance that is generous for the 0.06 m-radius ant foot is
        # invisible against ant_gym's 4x geometry. Assigned AFTER _arena_scale,
        # which it depends on.
        self._ground_contact_eps = (
            0.02 * self._arena_scale
            if ground_contact_eps is None
            else float(ground_contact_eps)
        )
        a = self._arena_scale
        # Counts are parameters rather than literals so a subclass can drop a
        # whole obstacle class. That is not cosmetic: vases are the only
        # DYNAMIC obstacles (one free joint each = 7 qpos / 6 qvel), so at the
        # default 10 they are ~82% of nq and ~81% of nv for the ant -- the
        # single largest term in per-step physics cost. Hazards are mocap
        # bodies with contype=conaffinity=0, contributing no DOFs and no
        # contacts at all, so they are nearly free by comparison. See
        # envs/minefield.py, which takes exactly that trade.
        self.spec = {
            "robot": ObjectSpec(0.4 * a, 1),
            "goal": ObjectSpec(0.305 * a, 1),
            # 0.9 * radius, which reproduces the historical 0.18 exactly at the
            # old radius of 0.2 and shrinks with it. Only vase placement reads
            # this now -- hazards themselves are laid out on a lattice by
            # RunForward and do not go through rejection sampling.
            "hazards": ObjectSpec(0.9 * self._hazard_size * a, int(num_hazards)),
            "vases": ObjectSpec(0.15 * a, int(num_vases)),
        }

        # Stored, not just used: build_morphology_model has to replay this
        # exact compile path, and a morphology built under a different
        # integrator would silently step different physics from the nominal env
        # while being stacked into the same batch.
        self._integrator = integrator

        mjSpec: mj.MjSpec = mj.MjSpec.from_file(filename=str(self._xml_path), assets={})
        apply_integrator(mjSpec, integrator)
        self._build_arena(mjSpec)
        self._mj_model = mjSpec.compile()

        # print(mjSpec.to_xml())

        self._mjx_model = mjx.put_model(self._mj_model)

        # Set (not derived from _mjx_model -- see morphology.py's
        # randomization_fn docstring for why genes can't be recovered from a
        # compiled model) once here and, under morphology randomization,
        # overwritten per training-env lane by
        # mjx_safety_gym.algorithms.wrappers.MorphologyDomainRandomizationWrapper,
        # the same way that wrapper overwrites `_mjx_model`. Appended in
        # get_obs unconditionally when morphology_conditioning is set, so obs
        # width is stable from the very first reset() -- this MUST happen
        # inside the env itself, not in an outer wrapper applied after
        # CostEpisodeWrapper: that wrapper's own action_repeat scan carries
        # `state` through repeated internal env.step() calls, so obs width
        # must already be final before it, or the scan's carry-vs-output
        # types mismatch on the very first step (this crashed a real training
        # run before being caught -- see the project plan).
        self._morphology_conditioning = morphology_conditioning
        self._morphology_genes = jp.full((NUM_GENES,), 0.5)

        self._post_init()

        self._vision = vision
        # Built per-instance rather than as an argument default: a
        # ConfigDict default is constructed once at import and then
        # SHARED by every env, so one env mutating it changes all of them.
        self._vision_config = (
            default_vision_config() if vision_config is None else vision_config
        )
        if self._vision: 
            try:
                # pylint: disable=import-outside-toplevel
                from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
            except ImportError:
                warnings.warn("Madrona MJX not installed. Cannot use vision with.")
                return
            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=self._vision_config.gpu_id,
                num_worlds=self._vision_config.render_batch_size,
                batch_render_view_width=self._vision_config.render_width,
                batch_render_view_height=self._vision_config.render_height,
                enabled_geom_groups=np.asarray(
                    self._vision_config.enabled_geom_groups
                ),
                enabled_cameras=np.asarray([
                    0,
                ]),
                add_cam_debug_geo=False,
                use_rasterizer=self._vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def build_morphology_model(self, spec) -> mj.MjModel:
        """Compile THIS task's arena around a morphology-scaled robot.

        Replays __init__'s own compile path with one extra step, so the model
        differs from what this env normally builds in the robot's geometry and
        NOTHING else -- same XML, same integrator, same `_build_arena`, so same
        topology and therefore batchable across morphologies.

        This exists because `morphology.build_mj_model` hardcodes GoToGoal's
        arena (10 hazards + 10 free-jointed vases). Under `--task minefield`
        that is silently the wrong model: Minefield is nq=15/nv=14 against
        GoToGoal's nq=85/nv=74, yet nbody and ngeom happen to MATCH at 38/35
        (20 hazards versus 10 hazards + 10 vases), so the mismatch does not
        reliably raise -- it just steps physics whose bodies the env's cached
        geom ids do not describe. Dispatching through the env instead means any
        task that overrides `_build_arena` gets the right arena for free.

        `spec` is a `morphology.MorphologySpec`; typed loosely to avoid a
        circular import (morphology imports NUM_GENES from this module).
        """
        from mjx_safety_gym import morphology as morphology_lib

        mjSpec: mj.MjSpec = mj.MjSpec.from_file(
            filename=str(self._xml_path), assets={}
        )
        apply_integrator(mjSpec, self._integrator)
        morphology_lib.apply_morphology(mjSpec, spec)
        self._build_arena(mjSpec)
        model = mjSpec.compile()
        morphology_lib.rescale_actuators(model, self._robot)
        return model

    def _build_arena(self, mjSpec: mj.MjSpec) -> None:
        """Add obstacles/goal/lidar rings to the spec, before it is compiled.

        A hook rather than an inline call so subclasses can reshape the arena
        without duplicating __init__'s whole compile sequence -- see
        envs/run_forward.py, which needs a long corridor instead of the default
        square. Overriding this is the ONLY supported way to change arena
        geometry: it runs before compile(), which is the last moment the
        MjSpec is still mutable.
        """
        build_arena(
            mjSpec, objects=self.spec, visualize=True,
            obstacle_scale=self._arena_scale,
            lidar_groups=self._lidar_groups,
            hazard_size=self._hazard_size,
            vase_mass=_ROBOT_CONFIGS[self._robot]["vase_mass"],
        )

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        self._goal_body_id = self._mj_model.body("goal").id

        # For cost function
        robot_config = _ROBOT_CONFIGS[self._robot]
        self._robot_collision_geom_ids = [
            self._mj_model.geom(name).id
            for name in robot_config["collision_geoms"]
        ]
        # Geoms, not bodies
        self._collision_obstacle_geoms_ids = [
            self._mj_model.geom(f"vase_{i}_geom").id
            for i in range(self.spec["vases"].num_objects)
            # + self._mj_model.geom(f'pillar{i}').id for i in range(self._num_pillars)
        ]
        self._hazard_body_ids = [
            self._mj_model.body(f"hazard_{i}").id
            for i in range(self.spec["hazards"].num_objects)
        ]  # Bodies, not geoms

        # Hazard proximity is measured against the robot's *whole body*, not just
        # the torso site: every collision geom is reduced to a swept sphere
        # (segment + radius), which is exact for the spheres and capsules the
        # robots are built from. Using the torso site alone let limbs reach well
        # inside a hazard for free -- the ant's feet extend ~0.30 from the site
        # while the hazard radius is only 0.20, so a foot could sit dead centre
        # in a hazard at zero cost. Deriving the extent from the geoms keeps this
        # honest as morphologies change: a longer leg cannot buy a blind spot.
        #
        # Only the geom *type* (capsule/sphere/other) is cached here. It is
        # structural and identical across every morphology this project
        # generates (verified: ngeom/geom_type never change when limbs are
        # rescaled -- only geom_size does). Caching *sizes* here would be wrong
        # under morphology randomization: DomainRandomizationVmapWrapper swaps
        # self._mjx_model out from under this env before each reset/step (see
        # the `sys` property), so a radius baked in once at __init__ would
        # silently describe the *nominal* body for every non-nominal one --
        # corrupting exactly the hazard-cost signal this comment is about.
        # get_cost re-reads sizes from the live model every call instead, via
        # _robot_geom_extent().
        geom_ids = np.array(self._robot_collision_geom_ids, dtype=int)
        geom_types = self._mj_model.geom_type[geom_ids]
        # Capsule size is (radius, half_length); a sphere is the half_length == 0
        # case. Anything else (e.g. the point robot's box arrow) has no exact
        # swept-sphere form, so fall back to its bounding sphere -- conservative,
        # never a blind spot.
        self._robot_geom_is_capsule = jp.array(geom_types == mj.mjtGeom.mjGEOM_CAPSULE)
        self._robot_geom_is_sphere = jp.array(geom_types == mj.mjtGeom.mjGEOM_SPHERE)
        self._robot_collision_geom_ids_arr = jp.array(geom_ids)
        # COLUMN indices of the foot geoms within `_robot_collision_geom_ids`,
        # so the per-foot observation can slice the same (H, G) matrix the cost
        # is computed from instead of recomputing the geometry.
        foot_names = robot_config.get("foot_geoms", [])
        self._foot_geom_names = list(foot_names)
        self._foot_geom_cols = jp.array(
            [robot_config["collision_geoms"].index(n) for n in foot_names],
            dtype=int,
        ) if foot_names else jp.zeros((0,), dtype=int)
        # Read the hazard radius off the model rather than hardcoding it, so the
        # threshold cannot silently drift from the geometry drawn in the arena.
        # GUARDED because a task may legitimately have NO hazards (envs/lasers.py
        # replaces them with beams); an unguarded read raises KeyError at
        # construction, which is why Minefield still refuses num_hazards < 1.
        self._hazard_radius = (
            float(self._mj_model.geom("hazard_0_geom").size[0])
            if self.spec["hazards"].num_objects
            else 0.0
        )

        # For lidar
        self._robot_body_id = self._mj_model.body("robot").id
        self._vase_body_ids = [
            self._mj_model.body(f"vase_{i}").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._obstacle_body_ids = self._vase_body_ids + self._hazard_body_ids
        self._object_body_ids = []

        # For observations: base IMU sensors plus any robot-specific proprioceptive
        # sensors (e.g. the ant's joint angles/velocities). Total dim is cached so
        # observation_size reflects the actual sensor layout of the chosen robot.
        self._obs_sensor_names = BASE_SENSORS + robot_config["extra_sensors"]
        # `.item()`, not `int(...)` on the summed generator: numpy's strictness
        # around converting non-0-d arrays to scalars varies by version --
        # `sensor(name).dim` is a shape-(1,) array on some numpy/mujoco
        # combinations, and summing several of those keeps it shape-(1,), which
        # `int()` rejects on newer numpy even though it's a single value.
        self._obs_sensor_dim = sum(
            np.asarray(self._mj_model.sensor(name).dim).item()
            for name in self._obs_sensor_names
        )

        # For position updates: either two planar slide joints (point) or a single
        # free joint (ant). `_robot_free_qposadr`, when set, is the start index of
        # the free joint's 7 qpos entries (3 translation + 4 quaternion).
        if robot_config["slide_joints"] is not None:
            self._robot_slide_qposadr = [
                self._mj_model.jnt_qposadr[self._mj_model.joint(name).id]
                for name in robot_config["slide_joints"]
            ]
            self._robot_free_qposadr = None
        else:
            free_joint_id = self._mj_model.joint(robot_config["free_joint"]).id
            self._robot_free_qposadr = int(self._mj_model.jnt_qposadr[free_joint_id])
            self._robot_spawn_height = float(robot_config["spawn_height"])
            self._robot_slide_qposadr = None
        self._goal_mocap_id = self._mj_model.body("goal").mocapid[0]
        self._hazard_mocap_id = [
            self._mj_model.body(f"hazard_{i}").mocapid[0]
            for i in range(self.spec["hazards"].num_objects)
        ]
        self._vase_joint_ids = [
            self._mj_model.joint(f"vase_{i}_joint").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._vase_joint_qposadr = [
            self._mj_model.jnt_qposadr[joint_id] for joint_id in self._vase_joint_ids
        ]

    def get_reward(
        self, data: mjx.Data, last_goal_dist: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        goal_distance = jp.linalg.norm(
            data.xpos[self._goal_body_id][:2] - data.site_xpos[self._robot_site_id][0:2]
        )
        reward = last_goal_dist - goal_distance
        return reward, goal_distance

    def _reset_goal(
        self, data: mjx.Data, rng: jax.Array
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        # Returns the distance to the *new* goal alongside the updated data, so
        # the caller can refresh `last_goal_dist`. Without that, the next step's
        # progress reward (last_goal_dist - goal_distance) subtracts a distance
        # measured against the old goal from one measured against the new one,
        # producing a large negative spike exactly when the goal is reached.

        # new_rng, goal_key = jax.random.split(rng)
        # new_xy = jax.random.uniform(goal_key, (2,), minval=-2.0, maxval=2.0)
        # # new_qpos = data.qpos.at[jp.array([self._goal_x_joint_id, self._goal_y_joint_id])].set(new_xy)
        # # data = data.replace(qpos=new_qpos)
        # data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(new_xy))
        # jax.debug.print("New goal position: {pos}", pos=new_xy)
        # return data, rng

        # TODO: probably could just use xpos with self._obstacle_body_ids instead of mocap_pos as well - it seems to work
        rng, goal_key = jax.random.split(rng)
        # `jp.array([])` is float32, and indexing with a float array raises --
        # so an obstacle class set to 0 (reachable since num_hazards/num_vases
        # became parameters) has to short-circuit rather than fall through.
        def _xy(positions, ids):
            return positions[jp.array(ids)][:, :2] if ids else jp.zeros((0, 2))

        hazard_pos = _xy(data.mocap_pos, self._hazard_mocap_id)
        vases_pos = _xy(data.xpos, self._vase_body_ids)
        other_xy = jp.vstack([hazard_pos, vases_pos])

        hazard_keepout = jp.full((hazard_pos.shape[0],), self.spec["hazards"].keepout)
        vases_keepout = jp.full((vases_pos.shape[0],), self.spec["vases"].keepout)
        other_keepout = jp.hstack([hazard_keepout, vases_keepout])

        xy, _ = draw_until_valid(
            goal_key, self.spec["goal"].keepout, other_xy, other_keepout
        )

        # new_qpos = data.qpos.at[jp.array([self._goal_x_joint_id, self._goal_y_joint_id])].set(new_xy)
        # data = data.replace(qpos=new_qpos)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(xy)
        )
        # Measured from `xy` rather than data.xpos[self._goal_body_id]: replacing
        # mocap_pos does not re-run forward kinematics, so xpos still holds the
        # old goal until the next step() integrates.
        new_goal_dist = jp.linalg.norm(xy - data.site_xpos[self._robot_site_id][0:2])
        # jax.debug.print("New goal position: {pos}", pos=xy)
        return data, rng, new_goal_dist

    def _robot_geom_extent(self) -> tuple[jax.Array, jax.Array]:
        """Robot collision-geom (radius, half_length), read from the *live*
        model.

        Must be recomputed here rather than cached once in _post_init: see the
        comment there. Reads from self._mjx_model (not self._mj_model, the
        host-side nominal model) so this reflects whatever body
        DomainRandomizationVmapWrapper has currently swapped in via the `sys`
        property, if any -- and is simply the nominal ant's own geometry
        otherwise.
        """
        sizes = self._mjx_model.geom_size[self._robot_collision_geom_ids_arr]
        half_len = jp.where(self._robot_geom_is_capsule, sizes[:, 1], 0.0)
        radius = jp.where(
            self._robot_geom_is_capsule | self._robot_geom_is_sphere,
            sizes[:, 0],
            self._mjx_model.geom_rbound[self._robot_collision_geom_ids_arr],
        )
        return radius.astype(jp.float32), half_len.astype(jp.float32)

    def get_cost(self, data: mjx.Data) -> jax.Array:
        # Check if any robot geom collides with any vase or pillar.
        # Skipped entirely when there are no collidable obstacles (see
        # envs/minefield.py): the comprehension would otherwise emit
        # `jp.array([])`, which sums to 0.0 correctly but only by accident of
        # float32 being the default empty dtype. A python-level branch on a
        # count fixed at __init__ is trace-time, so this costs nothing.
        if self._collision_obstacle_geoms_ids:
            collision_cost = jp.sum(
                jp.array([
                    jp.any(
                        jp.array([
                            geoms_colliding(data, geom, robot_geom)
                            for robot_geom in self._robot_collision_geom_ids
                        ])
                    )
                    for geom in self._collision_obstacle_geoms_ids
                ])
            )
        else:
            collision_cost = jp.zeros(())

        return (collision_cost + jp.sum(self.hazard_contacts(data))).astype(
            jp.float32
        )

    def _hazard_surface_matrix(self, data: mjx.Data, *, step_on: bool) -> jax.Array:
        """(H, G) distance from each hazard CENTRE to each robot geom's surface.

        Shared by the COST (`hazard_distances`, min over geoms) and the
        OBSERVATION (`foot_obstacle_observations`, min over hazards for the
        foot columns only), so the two can never describe different geometry.

        `step_on` is a parameter rather than read from the flag because the two
        callers genuinely want different things: the cost charges only geoms on
        the ground, while the observation must report clearance for a foot that
        is still IN THE AIR -- that is the whole point of giving it, and gating
        it would only ever say "you are already standing in one".
        """
        radius, half_len = self._robot_geom_extent()
        geom_pos = data.geom_xpos[self._robot_collision_geom_ids_arr]  # (G, 3)
        geom_axis = data.geom_xmat[self._robot_collision_geom_ids_arr].reshape(
            -1, 3, 3
        )[:, :, 2]  # capsule axis is local z
        offset = geom_axis[:, :2] * half_len[:, None]
        seg_a = geom_pos[:, :2] + offset
        seg_b = geom_pos[:, :2] - offset
        hazard_pos = data.xpos[jp.array(self._hazard_body_ids)][:, :2]  # (H, 2)
        surface = _segment_point_distance_2d(hazard_pos, seg_a, seg_b)
        surface -= radius[None, :]
        if step_on:
            # STEP-ON SEMANTICS: a geom only triggers a hazard while it is on
            # the ground. Without this the test is purely 2D, so a foot swung
            # THROUGH THE AIR over a mine is charged exactly as much as standing
            # on it -- which is not what a minefield means, and it charges the
            # ant for a gait rather than for where it puts its weight.
            #
            # Gate on the geom's LOWEST POINT rather than on mjx contacts. A
            # capsule's lowest point is centre_z - |axis_z| * half_len - radius
            # (spheres are the half_len == 0 case). Contacts would be the more
            # literal test but go through the broad phase, which
            # `max_geom_pairs=16` truncates -- so a foot genuinely on the ground
            # could be missing from `data.contact` and silently escape its cost.
            # Geometry cannot be truncated.
            lowest_z = geom_pos[:, 2] - jp.abs(geom_axis[:, 2]) * half_len - radius
            grounded = lowest_z <= self._ground_contact_eps  # (G,)
            # A large finite sentinel, not jp.inf: an all-airborne robot would
            # otherwise reduce to inf and any later arithmetic produces NaN
            # rather than a clean "no hazard".
            surface = jp.where(grounded[None, :], surface, 1e6)
        return surface

    def hazard_distances(self, data: mjx.Data) -> jax.Array:
        """Distance from each hazard centre to the robot's nearest surface, (H,).

        Split out of `get_cost` so the viewer can highlight exactly the hazards
        being CHARGED rather than reimplementing the geometry and letting the
        two drift apart. Traced inline, so `get_cost` is unchanged.
        """
        if not self._hazard_body_ids:
            # NO HAZARDS AT ALL (envs/lasers.py). Short-circuit at trace time:
            # `jp.array([])` is float32, and indexing `data.xpos` with a float
            # array raises -- the same trap `_reset_goal` guards against.
            return jp.zeros((0,), dtype=jp.float32)
        return jp.min(
            self._hazard_surface_matrix(data, step_on=self._hazard_step_on), axis=1
        )

    def hazard_contacts(self, data: mjx.Data) -> jax.Array:
        """Boolean (H,): which hazards are currently charging cost."""
        return self.hazard_distances(data) <= self._hazard_radius

    def lidar_observations(self, data: mjx.Data) -> jax.Array:
        """Compute Lidar observations."""
        robot_body_pos = data.xpos[self._robot_body_id]
        robot_body_mat = data.xmat[self._robot_body_id].reshape(3, 3)

        # Vectorized obstacle position retrieval -- note we can use xpos even for mocap positions after they have been updated
        # These values seem to be equal; TODO: using mocap_pos is maybe more correct
        targets = {
            # Guarded exactly like "object" below: with no hazards AND no vases
            # (envs/lasers.py) this list is empty, and `jp.array([])` is
            # float32 -- indexing `data.xpos` with it raises
            # "Indexer must have integer or boolean type". The ring is then a
            # constant zero vector, which KEEPS observation_size at 47 so
            # checkpoints still transfer between the corridor tasks.
            "obstacle": lambda: (
                data.xpos[jp.array(self._obstacle_body_ids)]
                if self._obstacle_body_ids
                else jp.zeros((0, 3))
            ),
            "goal": lambda: data.mocap_pos[jp.array([self._goal_mocap_id])],
            "object": lambda: (
                data.xpos[jp.array(self._object_body_ids)]
                if self._object_body_ids
                else jp.zeros((0, 3))
            ),
        }
        # Only the configured rings are computed, so a dropped ring costs
        # nothing at trace time either -- it is a python-level loop over a tuple
        # fixed at __init__.
        return jp.array([
            lidar.compute_lidar(robot_body_pos, robot_body_mat, targets[group]())
            for group in self._lidar_groups
        ])

    def sensor_observations(self, data: mjx.Data) -> jax.Array:
        vals = []
        for sensor in self._obs_sensor_names:
            vals.append(get_sensor_data(self.mj_model, data, sensor))
        return jp.hstack(vals)

    def task_observations(self, data: mjx.Data) -> Optional[jax.Array]:
        """Extra task-specific observation entries, or None.

        A hook so a subclass can widen the observation WITHOUT displacing the
        morphology genes, which must stay the last `NUM_GENES` entries:
        MorphologyDomainRandomizationWrapper writes them per-lane and
        tests/test_morphology.py asserts they land in the obs tail. Anything
        appended here goes before them.
        """
        return None

    def task_observation_size(self) -> int:
        """Width of `task_observations`. Must agree with it or the policy's
        input shape and the env's reported shape silently disagree."""
        return 0

    def _foot_ground_gap(self, data: mjx.Data) -> jax.Array:
        """(F,) signed height of each foot above the GROUNDED threshold.

        Negative means the foot is down far enough to charge cost; positive
        means it is in the air. Same zero-reference convention as the clearance
        entry, so the whole cost condition reads as

            cost fires  iff  clearance <= 0  AND  ground_gap <= 0

        A capsule's lowest point is `centre_z - |axis_z| * half_len - radius`,
        which is exactly what `_hazard_surface_matrix` gates on -- computed
        here rather than returned from there because the observation wants it
        for the FOOT columns only and ungated.
        """
        radius, half_len = self._robot_geom_extent()
        cols = self._foot_geom_cols
        ids = self._robot_collision_geom_ids_arr[cols]
        gp = data.geom_xpos[ids]
        ga = data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        lowest_z = gp[:, 2] - jp.abs(ga[:, 2]) * half_len[cols] - radius[cols]
        return lowest_z - self._ground_contact_eps

    def _scaled_ground_gap(self, data: mjx.Data) -> jax.Array:
        """`_foot_ground_gap` normalised to O(1).

        Scaled by the robot's SPAWN HEIGHT rather than the 2 m used for
        clearance: feet live within a body-height of the floor, so the
        clearance scale would compress the entire useful range into a few
        hundredths and waste the signal. Spawn height is the natural body unit
        and is already per-robot (0.18 for ant, 0.75 for ant_gym).
        """
        scale = float(getattr(self, "_robot_spawn_height", 0.0)) or (
            0.2 * self._arena_scale
        )
        return jp.clip(self._foot_ground_gap(data) / scale, -1.0, 1.0)

    def foot_obstacle_observations(self, data: mjx.Data) -> Optional[jax.Array]:
        """Per-foot clearance to the nearest hazard: 3 entries per foot.

        WHY THIS EXISTS. The only obstacle input the policy ever had was the
        lidar ring, and that is computed from `data.xpos[self._robot_body_id]`
        -- a SINGLE TORSO POINT. Cost, meanwhile, is charged on the minimum
        over all 13 collision geoms and, since `--hazard_step_on`, only for the
        ones ON THE GROUND. So the policy was punished for where its FEET
        landed while being shown only where its TORSO was, and had to recover
        foot placement by composing the ring with 8 joint angles. That is a
        plausible reason a 32x4 net was 2.7x worse than 256x4, and why the ring
        helped ONLY at 256x4.

        Per foot, in the torso's YAW frame:
            [0] clearance to the nearest hazard's EDGE, negative when inside
            [1] cos of the bearing to that hazard
            [2] sin of the bearing to that hazard
            [3] height above the GROUNDED threshold, negative when down

        [3] EXISTS BECAUSE [0] IS PURELY HORIZONTAL. A foot 40 cm in the air
        directly over a disc reports the same -0.031 as a foot planted in it --
        measured. Cost, though, fires only when BOTH the horizontal clearance
        and the height are non-positive, so without [3] the policy sees one
        number for two situations that differ entirely in what they cost, and
        would have to recover height from 8 joint angles plus a torso height
        that IS NOT IN THE OBSERVATION AT ALL (no sensor reports absolute z --
        accelerometer, velocimeter, gyro and magnetometer are all body-frame).
        With [3] the cost condition is fully observable: `[0] <= 0 and [3] <= 0`.

        Clearance is the quantity `get_cost` thresholds (surface distance minus
        `_hazard_radius`), so 0 is exactly the point at which a grounded foot
        starts charging. It is scaled by 2 x arena_scale and clipped to
        [-1, 1]: observations are NOT normalised anywhere in this stack
        (`normalize = lambda x, y: x`, ppo/train.py), so raw metres would enter
        the first layer an order of magnitude above every other input -- the
        same reason `task_observations` divides its range by corridor_length.

        BEARING IS YAW-ONLY, not the full orientation matrix, for the reason
        documented on `RunForward.task_observations`: `lidar.ego_xy` leaves the
        robot's own height in the vector it rotates, so a tilted torso distorts
        every reading -- at 60 degrees of pitch a target 2.00 m away registers
        as 1.58 m. A foot-placement signal that degrades as the ant falls over
        would fail exactly where it is needed.

        NOT step-on gated, deliberately -- see `_hazard_surface_matrix`.
        """
        if not self._foot_obstacle_obs or self._foot_geom_cols.size == 0:
            return None
        n_feet = int(self._foot_geom_cols.size)
        if not self._hazard_body_ids:
            # No hazards (envs/lasers.py overrides this method with its own
            # beam version). Report maximum clearance rather than a shape
            # mismatch: constant, but honest -- there is nothing to avoid.
            return jp.stack(
                [
                    jp.ones((n_feet,)),        # maximum clearance: nothing to avoid
                    jp.ones((n_feet,)),        # bearing points nowhere in particular
                    jp.zeros((n_feet,)),
                    self._scaled_ground_gap(data),  # height is still real and useful
                ],
                axis=-1,
            ).flatten()

        # (H, G) UNGATED -- a foot in the air must still see what it is about
        # to land on -- then restrict to the foot columns.
        surface = self._hazard_surface_matrix(data, step_on=False)
        surface = surface[:, self._foot_geom_cols]                    # (H, F)
        nearest = jp.argmin(surface, axis=0)                          # (F,)
        clearance = jp.min(surface, axis=0) - self._hazard_radius     # (F,)
        scale = 2.0 * self._arena_scale
        clearance = jp.clip(clearance / scale, -1.0, 1.0)

        foot_xy = data.geom_xpos[
            self._robot_collision_geom_ids_arr[self._foot_geom_cols]
        ][:, :2]                                                      # (F, 2)
        hazard_xy = data.xpos[jp.array(self._hazard_body_ids)][:, :2]  # (H, 2)
        delta = hazard_xy[nearest] - foot_xy                           # (F, 2)
        mat = data.xmat[self._robot_body_id].reshape(3, 3)
        yaw = jp.arctan2(mat[1, 0], mat[0, 0])
        rel = jp.arctan2(delta[:, 1], delta[:, 0]) - yaw
        return jp.stack(
            [clearance, jp.cos(rel), jp.sin(rel), self._scaled_ground_gap(data)],
            axis=-1,
        ).flatten()

    def foot_obstacle_observation_size(self) -> int:
        """Width of `foot_obstacle_observations`. Must agree with it or the
        policy's input shape and the env's reported shape silently disagree."""
        if not self._foot_obstacle_obs:
            return 0
        return 4 * int(self._foot_geom_cols.size)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        lidar = self.lidar_observations(data)
        other_sensors = self.sensor_observations(data)
        parts = [lidar.flatten(), other_sensors]
        task = self.task_observations(data)
        if task is not None:
            parts.append(task)
        feet = self.foot_obstacle_observations(data)
        if feet is not None:
            parts.append(feet)
        if self._morphology_conditioning:
            parts.append(self._morphology_genes)
        return jp.hstack(parts)

    def update_positions(
        self,
        data: mjx.Data,
        layout: dict[str, list[tuple[int, jax.Array]]],
        rng: jax.Array,
    ) -> tuple[mjx.Data, jax.Array]:
        mocap_pos = data.mocap_pos
        qpos = data.qpos

        # Set robot position. Planar robots use two slide joints (xy only); free-joint
        # robots (e.g. ant) need all 7 qpos: xy + spawn height + an identity quaternion.
        robot_xy = layout["robot"][0][1]
        if self._robot_free_qposadr is not None:
            adr = self._robot_free_qposadr
            identity_quat = jp.array([1.0, 0.0, 0.0, 0.0])
            qpos = qpos.at[adr : adr + 7].set(
                jp.hstack([robot_xy, self._robot_spawn_height, identity_quat])
            )
        else:
            qpos = qpos.at[jp.array(self._robot_slide_qposadr)].set(robot_xy)

        # N.B. could not figure out how to do it with get_qpos_ids, it seems to repeat some indices and hence does not set stuff correctly
        for i, (_, xy) in enumerate(layout["vases"]):
            rng, rng_ = jax.random.split(rng)
            adr = self._vase_joint_qposadr[i]
            rotation = jax.random.uniform(rng_, minval=0.0, maxval=2 * jp.pi)
            quat = _rot2quat(rotation)
            qpos = qpos.at[adr : adr + 7].set(
                jp.hstack([xy, 0.1 * self._arena_scale, quat])
            )

        # Set hazard positions
        for i, (_, xy) in enumerate(layout["hazards"]):
            mocap_pos = mocap_pos.at[self._hazard_mocap_id[i]].set(
                jp.hstack([xy, 0.02 * self._arena_scale])
            )

        # Set goal position
        mocap_pos = mocap_pos.at[self._goal_mocap_id].set(
            jp.hstack([layout["goal"][0][1], (0.3 / 2.0 + 1e-2) * self._arena_scale])
        )

        data = data.replace(qpos=qpos, mocap_pos=mocap_pos)

        return data, rng

    def reset(self, rng) -> State:
        data = mjx.make_data(self._mjx_model)

        # Set initial object positions
        layout = _sample_layout(rng, self.spec)
        data, rng = self.update_positions(data, layout, rng)
        data = mjx.forward(
            self._mjx_model, data
        )  # Make sure updated positions are reflected in data

        # Check updated positiosn are correct
        # print("Hazards:")
        # print(layout["hazards"])
        # print(data.mocap_pos[jp.array(self._hazard_mocap_id)])

        # print("Vases:")
        # print(layout["vases"])
        # print(data.xpos[jp.array(self._vase_body_ids)])

        # print("Goal:")
        # print(layout["goal"])
        # print(data.mocap_pos[self._goal_mocap_id])

        # print("Robot:")
        # print(layout["robot"])
        # print(data.xpos[jp.array(self._robot_body_id)])

        initial_goal_dist = jp.linalg.norm(
            data.mocap_pos[self._goal_mocap_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        # "goal_reached" must exist here as well as in step(): the auto-reset
        # wrapper tree_maps between reset and stepped states, so a key present
        # in only one of them is a pytree structure mismatch.
        info = {
            "rng": rng,
            "last_goal_dist": initial_goal_dist,
            "cost": jp.zeros(()),
            "goal_reached": jp.zeros(()),
        }

        obs = self.get_obs(data)

            # Vision observation instead 
        if self._vision:
            # Assume CNN takes grayscale images of dimensions (history, height, width)
            render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
            info.update({"render_token": render_token})
            obs = _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0 
            obs_history = jp.tile(obs, (self._vision_config.history, 1, 1))
            info.update({"obs_history": obs_history})
            obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

        return State(data, obs, jp.zeros(()), jp.zeros(()), {}, info)  # type: ignore

    def step(self, state: State, action: jax.Array) -> State:
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        action = (action + 1.0) / 2.0 * (upper - lower) + lower

        data = step(self._mjx_model, state.data, action, n_substeps=2)
        # The previous distance is RECOMPUTED from state.data rather than read
        # from state.info. BraxAutoResetWrapper restores `data` and `obs` from
        # the reset state when an episode ends but leaves every other info key
        # untouched, so a carried scalar makes the first step of each episode
        # difference against the PREVIOUS episode's final goal distance. That
        # produced a ~1 m spurious reward once per episode against a typical
        # per-step ~0.01 m -- a 100x outlier landing straight in PPO's
        # advantage normalisation. Deriving it from `data` is correct across
        # episode boundaries by construction.
        #
        # mocap_pos, NOT xpos: _reset_goal writes mocap_pos without re-running
        # forward kinematics (see its comment), so xpos still holds the OLD
        # goal for one step after a mid-episode respawn. reset() measures
        # initial_goal_dist the same way.
        prev_goal_dist = jp.linalg.norm(
            state.data.mocap_pos[self._goal_mocap_id][:2]
            - state.data.site_xpos[self._robot_site_id][0:2]
        )
        reward, goal_dist = self.get_reward(data, prev_goal_dist)

        # Reset goal if robot inside goal
        condition = goal_dist < 0.3
        data, rng, goal_dist = jax.lax.cond(
            condition,
            self._reset_goal,
            lambda d, r: (d, r, goal_dist),
            data,
            state.info["rng"],
        )
        # Explicit bonus for reaching the goal. The shaped term alone telescopes
        # to (initial_dist - final_dist) over an episode, so without this there
        # is no standing incentive to reach goals at all.
        reward = jp.where(condition, reward + 1.0, reward)

        cost = self.get_cost(data)

        observations = self.get_obs(data)

        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(jp.float32)

        # Update in place (rather than replacing) so that keys injected by
        # training wrappers (e.g. "steps", "truncation") survive the step.
        state.info["rng"] = rng
        state.info["cost"] = cost
        # DIAGNOSTIC ONLY (scripts/interactive.py reads it). Do not feed this
        # back into the reward: it survives the auto-reset boundary. See the
        # prev_goal_dist comment above.
        state.info["last_goal_dist"] = goal_dist
        state.info["goal_reached"] = condition.astype(jp.float32)
        info = state.info

        if self._vision:
            _, rgb, _ = self.renderer.render(state.info["render_token"], data)
            # Update observation buffer
            obs_history = state.info["obs_history"]
            obs_history = jp.roll(obs_history, 1, axis=0)
            obs_history = obs_history.at[0].set(
                _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
            )
            state.info["obs_history"] = obs_history
            obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

            return State(data, obs, reward, done, state.metrics, state.info)

        return State(
            data=data,
            obs=observations,
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=info,
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def lidar_groups(self) -> tuple[str, ...]:
        """Which lidar rings this env puts in the observation, in order.

        Public because the viewer has to slice the observation by it --
        main.py and scripts/interactive.py both read the leading
        `len(lidar_groups) * NUM_LIDAR_BINS` entries back out to light up the
        rings, and hardcoding 3 there silently mis-slices a narrowed env.
        """
        return self._lidar_groups

    @property
    def observation_size(self) -> int:
        size = len(self._lidar_groups) * lidar.NUM_LIDAR_BINS + self._obs_sensor_dim
        size += self.task_observation_size()
        size += self.foot_obstacle_observation_size()
        if self._morphology_conditioning:
            size += NUM_GENES
        return size


def get_sensor_data(model: mj.MjModel, data: mjx.Data, sensor_name: str) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def _rot2quat(theta):
    return jp.array([jp.cos(theta / 2), 0, 0, jp.sin(theta / 2)])

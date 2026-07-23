from typing import Mapping, Union
import warnings

import jax
from mujoco import mjx
import mujoco as mj
import jax.numpy as jp
from importlib.resources import files
import numpy as np

from ml_collections import config_dict

from mjx_safety_gym.collision import geoms_colliding
from mjx_safety_gym.mjx_env import State, step
import mjx_safety_gym.lidar as lidar
from mjx_safety_gym.world import (
    _EXTENTS,
    ObjectSpec,
    _sample_layout,
    build_arena,
    draw_until_valid,
)

_XML_PATH = files("mjx_safety_gym.envs.xmls") / "point.xml"

Observation = Union[jax.Array, Mapping[str, jax.Array]]
BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        return

    in_axes = jax.tree_map(lambda x: None, sys)
    return sys, in_axes, jp.zeros(())

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

def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
  """
  Intensity-weigh the colors.
  This expects the input to have the channels in the last dim.
  Values from ITU-R BT.60 standard for RGB to grayscale conversion.
  """
  r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

class GoToGoal:
    def __init__(self, vision: bool=False, vision_config=default_vision_config()):
        self.spec = {
            "robot": ObjectSpec(0.4, 1),
            "goal": ObjectSpec(0.305, 1),
            "hazards": ObjectSpec(0.18, 10),
            "vases": ObjectSpec(0.15, 10),
        }

        mjSpec: mj.MjSpec = mj.MjSpec.from_file(filename=str(_XML_PATH), assets={})
        build_arena(mjSpec, objects=self.spec, visualize=True)
        self._mj_model = mjSpec.compile()

        # print(mjSpec.to_xml())

        self._mjx_model = mjx.put_model(self._mj_model)

        
    
        self._post_init()

        self._vision = vision
        self._vision_config = vision_config 
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

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        self._goal_body_id = self._mj_model.body("goal").id

        # For cost function
        self._robot_geom_id = self._mj_model.geom("robot").id
        self._pointarrow_geom_id = self._mj_model.geom("pointarrow").id
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

        # For lidar
        self._robot_body_id = self._mj_model.body("robot").id
        self._vase_body_ids = [
            self._mj_model.body(f"vase_{i}").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._obstacle_body_ids = self._vase_body_ids + self._hazard_body_ids
        self._object_body_ids = []

        # For position updates
        self._robot_x_id = self._mj_model.joint("x").id
        self._robot_y_id = self._mj_model.joint("y").id
        self._robot_joint_qposadr = [
            self._mj_model.jnt_qposadr[joint_id]
            for joint_id in [self._robot_x_id, self._robot_y_id]
        ]
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

    def _reset_goal(self, data: mjx.Data, rng: jax.Array) -> tuple[mjx.Data, jax.Array]:
        # Initial state

        # new_rng, goal_key = jax.random.split(rng)
        # new_xy = jax.random.uniform(goal_key, (2,), minval=-2.0, maxval=2.0)
        # # new_qpos = data.qpos.at[jp.array([self._goal_x_joint_id, self._goal_y_joint_id])].set(new_xy)
        # # data = data.replace(qpos=new_qpos)
        # data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(new_xy))
        # jax.debug.print("New goal position: {pos}", pos=new_xy)
        # return data, rng

        # TODO: probably could just use xpos with self._obstacle_body_ids instead of mocap_pos as well - it seems to work
        rng, goal_key = jax.random.split(rng)
        hazard_pos = data.mocap_pos[jp.array(self._hazard_mocap_id)][:, :2]
        vases_pos = data.xpos[jp.array(self._vase_body_ids)][:, :2]
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
        # jax.debug.print("New goal position: {pos}", pos=xy)
        return data, rng

    def get_cost(self, data: mjx.Data) -> jax.Array:
        # Check if robot or pointarrow geom collide with any vase or pillar
        colliding_obstacles = jp.array(
            [
                jp.logical_or(
                    geoms_colliding(data, geom, self._robot_geom_id),
                    geoms_colliding(data, geom, self._pointarrow_geom_id),
                )
                for geom in self._collision_obstacle_geoms_ids
            ]
        )

        # FOR DEBUG PURPOSES, UNCOMMENT THIS
        colliding_obstacles = jax.lax.cond(
            jp.any(colliding_obstacles),  # If there's any collision
            lambda x: jax.debug.print(
                "Collision detected with obstacles: {collisions}", collisions=x
            )
            or x,  # Print and return the collisions
            lambda x: x,  # Otherwise, return the input unchanged
            colliding_obstacles,  # The value to pass into the lambda
        )

        # Hazard distance calculation (vectorized for all hazards)
        robot_pos = data.site_xpos[self._robot_site_id][:2]
        hazard_distances = jp.linalg.norm(
            data.xpos[jp.array(self._hazard_body_ids)][:, :2] - robot_pos, axis=1
        )
        # jax.debug.print("Hazard distances: {dist}", dist=hazard_distances)

        # Compute cost: Add cost for collisions and proximity to hazards
        cost = jp.sum(colliding_obstacles) + jp.sum(hazard_distances <= 0.2)

        return cost.astype(jp.float32)

    def lidar_observations(self, data: mjx.Data) -> jax.Array:
        """Compute Lidar observations."""
        robot_body_pos = data.xpos[self._robot_body_id]
        robot_body_mat = data.xmat[self._robot_body_id].reshape(3, 3)

        # Vectorized obstacle position retrieval -- note we can use xpos even for mocap positions after they have been updated
        # These values seem to be equal; TODO: using mocap_pos is maybe more correct
        obstacle_positions = data.xpos[jp.array(self._obstacle_body_ids)]
        goal_positions = data.mocap_pos[jp.array([self._goal_mocap_id])]
        object_positions = (
            data.xpos[jp.array(self._object_body_ids)]
            if self._object_body_ids
            else jp.zeros((0, 3))
        )

        lidar_readings = jp.array(
            [
                lidar.compute_lidar(robot_body_pos, robot_body_mat, obstacle_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, goal_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, object_positions),
            ]
        )

        return lidar_readings

    def sensor_observations(self, data: mjx.Data) -> jax.Array:
        vals = []
        for sensor in BASE_SENSORS:
            vals.append(get_sensor_data(self.mj_model, data, sensor))
        return jp.hstack(vals)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        lidar = self.lidar_observations(data)
        other_sensors = self.sensor_observations(data)
        return jp.hstack([lidar.flatten(), other_sensors])

    def update_positions(
        self,
        data: mjx.Data,
        layout: dict[str, list[tuple[int, jax.Array]]],
        rng: jax.Array,
    ) -> tuple[mjx.Data, jax.Array]:
        mocap_pos = data.mocap_pos
        qpos = data.qpos

        # Set robot position
        qpos = data.qpos.at[jp.array(self._robot_joint_qposadr)].set(
            layout["robot"][0][1]
        )

        # N.B. could not figure out how to do it with get_qpos_ids, it seems to repeat some indices and hence does not set stuff correctly
        for i, (_, xy) in enumerate(layout["vases"]):
            rng, rng_ = jax.random.split(rng)
            adr = self._vase_joint_qposadr[i]
            rotation = jax.random.uniform(rng_, minval=0.0, maxval=2 * jp.pi)
            quat = _rot2quat(rotation)
            qpos = qpos.at[adr : adr + 7].set(jp.hstack([xy, 0.1, quat]))

        # Set hazard positions
        for i, (_, xy) in enumerate(layout["hazards"]):
            mocap_pos = mocap_pos.at[self._hazard_mocap_id[i]].set(
                jp.hstack([xy, 0.02])
            )

        # Set goal position
        mocap_pos = mocap_pos.at[self._goal_mocap_id].set(
            jp.hstack([layout["goal"][0][1], 0.3 / 2.0 + 1e-2])
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
        info = {"rng": rng, "last_goal_dist": initial_goal_dist, "cost": jp.zeros(())}

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
        reward, goal_dist = self.get_reward(data, state.info["last_goal_dist"])

        # Reset goal if robot inside goal
        condition = goal_dist < 0.3
        data, rng = jax.lax.cond(
            condition, self._reset_goal, lambda d, r: (d, r), data, state.info["rng"]
        )

        cost = self.get_cost(data)

        observations = self.get_obs(data)

        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(jp.float32)

        info = {"rng": rng, "cost": cost, "last_goal_dist": goal_dist}

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
        return _XML_PATH

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
    def observation_size(self) -> int:
        return 3 * lidar.NUM_LIDAR_BINS + len(BASE_SENSORS) * 3


def get_sensor_data(model: mj.MjModel, data: mjx.Data, sensor_name: str) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def _rot2quat(theta):
    return jp.array([jp.cos(theta / 2), 0, 0, jp.sin(theta / 2)])

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
  * No "healthy"/alive bonus and no fall termination. Gym's Ant terminates
    outside a torso-z range of (0.2, 1.0), but that threshold is tuned to a
    robot whose torso radius is 0.25 m; safety-gymnasium's ant is 4x smaller and
    settles at z ~= 0.13-0.16, so the same numbers would terminate it on the
    first step. A flipped ant simply stops making progress, which this reward
    already handles. Termination is left out rather than guessed at.
  * The goal body is NOT removed, even though nothing respawns it. It is parked
    once at the far end of the corridor as a fixed beacon. Keeping it holds the
    observation width identical to GoToGoal (76 for the ant), so networks,
    checkpoints, morphology-gene conditioning and the eval plumbing all work
    unchanged -- and its lidar reading degenerates into a useful "how far is
    left to run" signal rather than a navigation problem, since the direction
    never changes.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx

from mjx_safety_gym.envs.go_to_goal import _ROBOT_CONFIGS, GoToGoal
from mjx_safety_gym.mjx_env import State, step
from mjx_safety_gym.world import build_arena, placement_not_valid


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

        # Robots start at -L/2 + margin and run toward +L/2. Obstacles fill the
        # span between the start line and the far end.
        self._start_x = -0.5 * self._corridor_length + self._start_margin
        self._finish_x = 0.5 * self._corridor_length
        self._obstacle_x_lo = self._start_x + 0.75 * a  # clear runway to get moving
        self._obstacle_x_hi = self._finish_x - 0.25 * a

        super().__init__(robot=robot, **kwargs)

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
            vase_mass=_ROBOT_CONFIGS[self._robot]["vase_mass"],
        )

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
        for name, count in (("hazards", n_haz), ("vases", n_vase)):
            keepout = self.spec[name].keepout
            entries = []
            rng, sub = jax.random.split(rng)
            for key in jax.random.split(sub, count):
                xy = draw_one(key, keepout, placed, keepouts)
                placed = placed.at[idx].set(xy)
                keepouts = keepouts.at[idx].set(keepout)
                entries.append((idx, xy))
                idx += 1
            layout[name] = entries

        # Robot on the start line, jittered in y so it does not memorise one lane.
        rng, rk = jax.random.split(rng)
        robot_xy = jp.array([
            self._start_x,
            jax.random.uniform(
                rk,
                (),
                minval=-0.5 * self._corridor_half_width,
                maxval=0.5 * self._corridor_half_width,
            ),
        ])
        layout["robot"] = [(0, robot_xy)]
        # Fixed beacon at the far end -- never moves, never respawns.
        layout["goal"] = [(0, jp.array([self._finish_x, 0.0]))]
        return layout, rng

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

        NOTE: GoToGoal.get_reward has the identical latent bug via its carried
        `last_goal_dist`. It is NOT fixed here because doing so changes the
        point robot's training dynamics and invalidates the only converged
        baselines this project has. See the project plan.
        """
        x_prev = prev_data.site_xpos[self._robot_site_id][0]
        x_new = data.site_xpos[self._robot_site_id][0]
        return (x_new - x_prev) * self._forward_reward_weight

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
            # Recomputed fresh every step, never carried.
            "out_of_bounds": jp.zeros(()),
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

        state.info["cost"] = cost
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

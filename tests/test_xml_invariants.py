"""Invariants of the robot XMLs that are silent when broken.

Every check here corresponds to something that either cost this project real
time or was adopted only after a measurement, and none of them raise an error
if violated -- the model still compiles and trains, it just trains against
different physics or a dead cost signal. That is precisely why they need tests
rather than comments.

Deliberately parses the XML with `mujoco` and inspects the COMPILED model
rather than reading the file as text, because most of these are properties of
what MuJoCo resolves (default-class inheritance, contact-parameter combination
rules), not of what the file literally says.
"""

from __future__ import annotations

from importlib.resources import files

import mujoco as mj
import numpy as np
import pytest

_XML_DIR = files("mjx_safety_gym.envs.xmls")
ROBOT_XMLS = {"point": "point.xml", "ant": "ant.xml", "ant_gym": "ant_gym.xml"}
ANTS = ("ant", "ant_gym")


def _xml_text(name: str) -> str:
    return (_XML_DIR / ROBOT_XMLS[name]).read_text()


def _compiled(name: str) -> mj.MjModel:
    """Compile the bare robot XML, without the arena built on top of it."""
    return mj.MjModel.from_xml_string(_xml_text(name))


@pytest.mark.parametrize("robot", sorted(ROBOT_XMLS))
def test_max_geom_pairs_is_the_first_numeric(robot):
    """MJX indexes `numeric_data` by numeric ID, not by address.

    `collision_driver._numeric()` reads `numeric_data[id]` instead of
    `numeric_data[adr]`, so if `init_qpos` is declared first this reads
    init_qpos's SECOND element (0.0), the cap becomes 0, every collision group
    empties, and model construction dies with the deeply unhelpful "Need at
    least one array to concatenate". Upstream bug; declaration order is the
    workaround, so the order is load-bearing and must not be tidied.

    Checked on the compiled model rather than the file text, both because
    numeric IDs are what MJX actually indexes and because these XMLs are not
    parseable by strict XML tools -- their comments use `--` as an em-dash,
    which is illegal inside an XML comment. MuJoCo's parser tolerates it.
    """
    model = _compiled(robot)
    numerics = [model.numeric(i).name for i in range(model.nnumeric)]
    assert numerics, f"{robot}: no <numeric> block at all"
    assert numerics[0] == "max_geom_pairs", (
        f"{robot}: max_geom_pairs must be the FIRST <numeric>, got {numerics}"
    )


@pytest.mark.parametrize("robot", ANTS)
def test_floor_condim_matches_the_default_class(robot):
    """A contact's condim is the MAX of its two geoms'.

    So a condim-6 floor silently keeps every robot-vs-floor contact at 6 no
    matter what the robot's geoms declare -- which would have made the
    2026-08-13 condim-3 change (measured 1.82x) buy exactly nothing, with no
    error and no visible symptom. Guard the whole model rather than just the
    floor: all geoms must agree.
    """
    model = _compiled(robot)
    condims = set(model.geom_condim.tolist())
    assert len(condims) == 1, (
        f"{robot}: mixed condim {condims}. A contact takes the MAX of its two "
        "geoms, so the highest value here wins for every pair that touches it."
    )


@pytest.mark.parametrize("robot", ANTS)
def test_ant_friction_coefficients_are_used_not_decorative(robot):
    """Don't pay for contact dimensions whose coefficients are left at default.

    The pre-2026-08-13 configuration was condim 6 with torsional/rolling
    friction at MuJoCo's defaults (0.005 / 0.0001), i.e. 100x and 5000x below
    the values the Gym ant declares: the solver paid for six dimensions while
    two of them did almost nothing. Either enable a dimension AND give it a
    real coefficient, or don't enable it. This test fails on the incoherent
    combination in both directions.
    """
    model = _compiled(robot)
    condim = int(model.geom_condim[0])
    torsion, roll = model.geom_friction[0][1], model.geom_friction[0][2]
    mujoco_default_torsion, mujoco_default_roll = 0.005, 0.0001

    if condim >= 4:
        assert torsion > mujoco_default_torsion, (
            f"{robot}: condim {condim} solves torsional friction but leaves the "
            f"coefficient at MuJoCo's default {torsion} -- full solver cost, no "
            "effect. Either raise it or drop to condim 3."
        )
    if condim >= 6:
        assert roll > mujoco_default_roll, (
            f"{robot}: condim {condim} solves rolling friction but leaves the "
            f"coefficient at MuJoCo's default {roll}."
        )


@pytest.mark.parametrize("robot", ANTS)
def test_ants_use_rk4(robot):
    """Euler and implicitfast blow up at exactly the gaits a policy produces.

    Measured 2026-08-12 under a scripted trot: Euler 16.7% NaN, implicitfast
    16.7%, RK4 0.0%. Random actions were safe at 0% for ALL THREE, which is how
    the Euler switch got made in the first place -- so this test exists because
    the cheap version of the check gave the wrong answer. A NaN reward lands
    straight in PPO's batch.
    """
    model = _compiled(robot)
    assert model.opt.integrator == mj.mjtIntegrator.mjINT_RK4, (
        f"{robot}: integrator is {mj.mjtIntegrator(model.opt.integrator).name}, "
        "not RK4. See the header of ant_gym.xml for the NaN measurements."
    )


@pytest.mark.parametrize("robot", ANTS)
def test_limb_geoms_still_collide(robot):
    """Gym's ant sets contype=0 on everything but the feet. We must not.

    `get_cost` detects vase contact via `geoms_colliding()` on the limb geoms,
    so masking limb collisions would zero the safety signal -- the quantity
    this entire project studies -- while everything still ran and trained. It
    is also a tempting speedup, which is exactly why it needs a test and not a
    comment.
    """
    model = _compiled(robot)
    limbs = [
        i
        for i in range(model.ngeom)
        if any(k in (model.geom(i).name or "") for k in ("leg", "ankle", "aux", "torso"))
    ]
    assert limbs, f"{robot}: found no limb geoms to check -- naming changed?"
    dead = [model.geom(i).name for i in limbs if model.geom_contype[i] == 0]
    assert not dead, (
        f"{robot}: limb geoms with contype=0 {dead}. These can never appear in "
        "state.contact, so geoms_colliding() returns False for them and the "
        "vase cost silently reads zero."
    )


def test_point_physics_is_untouched_by_ant_changes():
    """Point baselines stay comparable only if point's physics never moves.

    The ant XMLs have been changed repeatedly (integrator, condim, friction).
    Point is the only robot with converged baselines worth comparing against,
    so pin the values that would invalidate them.
    """
    model = _compiled("point")
    assert set(model.geom_condim.tolist()) == {6}
    np.testing.assert_allclose(model.geom_friction[0], [1.0, 0.005, 0.0001])

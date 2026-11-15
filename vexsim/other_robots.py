"""The other robots on the field: the partner's and the opponents'.

Each is a drivetrain like the robot's own (`mujoco_drive`), on its own
battery, with a box the size of its frame that the robot, the pieces and
the field hit.  It holds still with its motors braking, or drives a path:
`start_s` after the Match starts it chases a point LOOKAHEAD ahead along the path from
where it really is (pure pursuit), at `volts`, slowing over the last
SLOW_DOWN, and brakes at the end.  It turns toward that point at TURN_GAIN
and drives forward only as far as it faces it, so it turns in place
toward a point beside or behind it.  It has no lift or claw.

It stops once it is within FINISH of the end or has crossed the line
through the end square to the path's last leg.

A robot holding still on omni wheels rolls sideways under a push and
resists one from the front with its braking motors, as a real one does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np

from . import presets
from .chassis import Chassis, with_traction_wheels
from .control import PurePursuit, angle_error, arcade
from .motor import BrakeMode
from .mujoco_drive import Lump, MujocoDrive, add_drivetrain, settle
from .pieces import (ELEMENT_PREFIX, FLAT_BIT, PIECE_BIT, ROBOT_BIT, ROBOT_MU, STATIC_BIT, Pieces, RobotBuild,
                     contact_settings)
from .sim import SimConfig, Simulator
from .units import INCH, LB

#: How far ahead along its path a driving robot aims
LOOKAHEAD = 12.0 * INCH
#: A driving robot stops this close to the end of its path
FINISH = 2.0 * INCH
#: Over this last stretch of its path it slows down, to SLOWEST_VOLTS at the end
SLOW_DOWN = 24.0 * INCH
SLOWEST_VOLTS = 3.0
#: Volts of turn per radian it faces off the point it chases
TURN_GAIN = 6.0
#: How often its controller runs, as a robot's code does
CONTROL_PERIOD = 0.010
#: The frame's bottom, off the floor
FRAME_FLOOR_GAP = 0.4 * INCH
#: How long a robot settles alone to find how it rests on its wheels
REST_SETTLE_SECONDS = 0.5
ROLES = ("partner", "opponent 1", "opponent 2")

GUESSES: Dict[str, str] = {
    "Braking": "A robot holding still has its drive motors on brake (shorted), as teams set them in "
               "autonomous; hold mode would push back harder.",
    "Frame": "A box the robot's length and width from 0.4 in off the floor to its height: no bumpers, lift "
             "or intake reaching out. Its mass is a box centred at the drive's CG height.",
    "Driving": "Pure pursuit on where it really is, 12 in ahead, at a set voltage, slowing over the last "
               "24 in to 3 V and stopping 2 in from the end or once past it; 6 V of turn per radian off the "
               "point it chases. A team's own code would follow its path on its "
               "odometry, with its own errors.",
}


@dataclass(frozen=True)
class OtherRobot:
    """One other robot: its role in ROLES, where it starts (field frame,
    metres and radians), its size and mass, the vexsim preset of its
    drivetrain and which of its wheels are traction wheels ("none",
    "middle" or "all"), and the path it drives (field metres; empty, it
    holds still)."""
    role: str
    x: float
    y: float
    theta: float
    length: float = 18.0 * INCH
    width: float = 18.0 * INCH
    height: float = 18.0 * INCH
    mass_lb: float = 15.0
    drive: str = "four_motor_450"
    traction: str = "none"
    path: Tuple[Tuple[float, float], ...] = ()
    volts: float = 8.0
    start_s: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"an other robot's role is one of {ROLES}, not {self.role!r}")
        if not (6.0 * INCH <= min(self.length, self.width) and max(self.length, self.width) <= 24.0 * INCH):
            raise ValueError(f"the {self.label} is {self.length / INCH:.1f} x {self.width / INCH:.1f} in; "
                             "a robot is 6 to 24 in each way (SG2)")
        if not 6.0 * INCH <= self.height <= 50.0 * INCH:
            raise ValueError(f"the {self.label} is {self.height / INCH:.1f} in tall; 6 to 50 in (SG3)")
        if not 2.0 <= self.mass_lb <= 40.0:
            raise ValueError(f"the {self.label} weighs {self.mass_lb} lb; 2 to 40 lb")
        if not 0.0 < self.volts <= 12.0 or self.start_s < 0.0:
            raise ValueError(f"the {self.label} drives at 0 to 12 V, starting at 0 s or later")

    @property
    def label(self) -> str:
        return f"{self.role} robot"

    @property
    def body(self) -> str:
        return f"other {self.role}"

    @property
    def element(self) -> str:
        """Its name in contact reports, as a field element's."""
        return f"the {self.label}"

    def chassis(self) -> Chassis:
        chassis = presets.get(self.drive)
        if self.traction != "none":
            with_traction_wheels(chassis, self.traction)
        chassis.mass = self.mass_lb * LB
        chassis.length, chassis.width = self.length, self.width
        chassis.inertia_z = chassis.default_inertia()
        chassis.name = f"{self.label}: {chassis.name}"
        return chassis

    def build_key(self) -> tuple:
        return (self.role, self.length, self.width, self.height, self.mass_lb, self.drive, self.traction)


def check_roles(robots: Sequence[OtherRobot]) -> None:
    roles = [robot.role for robot in robots]
    if len(set(roles)) != len(roles):
        raise ValueError(f"each other robot needs its own role; got {roles}")


def robot_build(robot: OtherRobot) -> RobotBuild:
    return RobotBuild(robot.body, robot.build_key(), lambda spec: _build(spec, robot), repr(robot.build_key()))


def _build(spec: mujoco.MjSpec, robot: OtherRobot) -> None:
    chassis = robot.chassis()
    wheels = sum(unit.wheel.mass for unit in chassis.wheel_units)
    half_height = chassis.cg_height - FRAME_FLOOR_GAP
    mass = Lump("mass", chassis.mass - wheels, (chassis.cg_x, chassis.cg_y, chassis.cg_height),
                (robot.length / 2.0, robot.width / 2.0, half_height))
    body = add_drivetrain(spec, chassis, [mass], robot.body)
    frame_height = robot.height - FRAME_FLOOR_GAP
    geom = body.add_geom(name=f"{ELEMENT_PREFIX}{robot.element}|frame", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[robot.length / 2.0, robot.width / 2.0, frame_height / 2.0],
                         pos=[0.0, 0.0, FRAME_FLOOR_GAP + frame_height / 2.0], mass=0.0,
                         contype=STATIC_BIT, conaffinity=STATIC_BIT | FLAT_BIT | PIECE_BIT | ROBOT_BIT)
    contact_settings(geom, ROBOT_MU)


_AT_REST: Dict[tuple, np.ndarray] = {}


def at_rest(robot: OtherRobot) -> np.ndarray:
    """Its joint positions standing still on the floor at the field's middle
    heading +x (`MujocoDrive.robot_qpos`), worked out once per build on a
    field with nothing else on it."""
    key = robot.build_key()
    if key not in _AT_REST:
        chassis = robot.chassis()
        field = Pieces(goals=(), loaders=(), pins=0, cups=0, robot=robot_build(robot))
        drive = MujocoDrive(chassis, field.model, field.data, robot.body, -float(field.model.opt.gravity[2]))
        mujoco.mj_forward(field.model, field.data)
        settle(drive, REST_SETTLE_SECONDS)
        rest = drive.robot_qpos()
        heading = drive.state.theta
        unturned = np.empty(4)
        mujoco.mju_mulQuat(unturned, np.array([math.cos(heading / 2.0), 0.0, 0.0, -math.sin(heading / 2.0)]),
                           rest[3:7].copy())
        rest[0:2] = 0.0
        rest[3:7] = unturned
        _AT_REST[key] = rest
    return _AT_REST[key].copy()


class OtherDrive:
    """One other robot in a world: its motors on its own battery, its
    drivetrain's forces each step, and its driver."""

    def __init__(self, robot: OtherRobot, model: mujoco.MjModel, data: mujoco.MjData, gravity: float):
        self.robot = robot
        self.chassis = robot.chassis()
        self.power = Simulator(self.chassis, config=SimConfig(log_every=0, gravity=gravity))
        self.drive = MujocoDrive(self.chassis, model, data, robot.body, gravity)
        self.frame_geoms = [g for g in range(model.ngeom) if model.geom(g).name.startswith(
            f"{ELEMENT_PREFIX}{robot.element}|")]
        self.drive.place(at_rest(robot), robot.x, robot.y, robot.theta)
        self.pursuit = PurePursuit(list(robot.path), lookahead=LOOKAHEAD, finish_tolerance=FINISH)
        self.left = [m for unit in self.chassis.left_units() for m in unit.motors]
        self.right = [m for unit in self.chassis.right_units() for m in unit.motors]
        self.match_started = False
        self.t = self.next_control = 0.0
        self.finished = not robot.path
        self._brake()

    def start(self) -> None:
        """The Match starts now: start its clock and let it drive."""
        self.match_started = True
        self.t = self.next_control = 0.0
        self.finished = not self.robot.path
        self.pursuit.reset()
        self._brake()

    def _brake(self) -> None:
        for motor in self.left + self.right:
            motor.stop(BrakeMode.BRAKE)

    def push(self, dt: float) -> None:
        if self.t >= self.next_control - 1e-12:
            self.next_control += CONTROL_PERIOD
            self._control()
        self.power.run_motors(dt)
        self.drive.push(dt)

    def moved(self, dt: float) -> None:
        self.drive.moved(dt)
        self.t += dt

    def _control(self) -> None:
        robot, st = self.robot, self.drive.state
        if not self.match_started or self.finished or self.t < robot.start_s:
            self._brake()
            return
        end_x, end_y = robot.path[-1]
        left_to_go = math.hypot(end_x - st.x, end_y - st.y)
        on_last_leg = self.pursuit.target(st.x, st.y) == robot.path[-1]
        last_x, last_y = robot.path[-2] if len(robot.path) > 1 else (robot.x, robot.y)
        past_end = (st.x - end_x) * (end_x - last_x) + (st.y - end_y) * (end_y - last_y) > 0.0
        if on_last_leg and (left_to_go < FINISH or past_end):
            self.finished = True
            self._brake()
            return
        volts = robot.volts if not on_last_leg else max(SLOWEST_VOLTS, robot.volts * min(1.0, left_to_go / SLOW_DOWN))
        target_x, target_y = self.pursuit.target(st.x, st.y)
        off = angle_error(math.atan2(target_y - st.y, target_x - st.x), st.theta)
        left, right = arcade(volts * max(0.0, math.cos(off)), TURN_GAIN * off)
        for motor in self.left:
            motor.set_voltage(left)
        for motor in self.right:
            motor.set_voltage(right)

    def report(self) -> dict:
        st = self.drive.state
        robot = self.robot
        end = robot.path[-1] if robot.path else (robot.x, robot.y)
        return dict(role=robot.role, x=st.x, y=st.y, theta=st.theta, finished=self.finished,
                    moved=math.hypot(st.x - robot.x, st.y - robot.y),
                    from_end=math.hypot(st.x - end[0], st.y - end[1]), fallen=self.drive.tilt.fallen)


class OtherRobots:
    """The other robots in a world, stepped with it: `drive` is the robot's
    own drivetrain, whose step steps the world."""

    def __init__(self, robots: Sequence[OtherRobot], model: mujoco.MjModel, data: mujoco.MjData,
                 drive: MujocoDrive):
        check_roles(robots)
        self.drives: List[OtherDrive] = [OtherDrive(robot, model, data, drive.gravity) for robot in robots]
        drive.forces.append(self._push)
        drive.after.append(self._moved)

    def _push(self, dt: float) -> None:
        for other in self.drives:
            other.push(dt)

    def _moved(self, dt: float) -> None:
        for other in self.drives:
            other.moved(dt)

    def start(self) -> None:
        for other in self.drives:
            other.start()

    def reports(self) -> List[dict]:
        return [other.report() for other in self.drives]

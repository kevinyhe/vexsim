"""A drivetrain as MuJoCo bodies.

The chassis is a body on a free joint, so it rolls, pitches and tips over as
the forces on it say.  Each wheel turns on a hinge on it, driven by its
motors through its gearing.  A tracking wheel also slides up and down on a
spring that presses it to the floor.

The wheels do not touch the floor through MuJoCo's contacts.  The floor is
foam, and a wheel sinks into it about a millimetre: a wheel's load is what
the foam pushes back with for how deep the chassis presses that wheel in
(`foam.sink`, turned round), so load moves from wheel to wheel as the chassis
rocks, brakes and turns.  Pressing the foam in takes more force than it
gives back coming out, by the foam's energy loss.  On that load vexsim's
tire model (`wheels.tire_forces`) gives each wheel's grip from its slip, the
foam's rolling resistance slows it, and its patch resists turning in place
(`foam.scrub_arm`).  The wheel's outline is a circle in its own plane: the
point of it lowest in the tile is where all of this acts.  These forces are
worked out from where every body is and how it moves at the start of each
MuJoCo step, and act through that step.

`MujocoDrive` takes the place of `dynamics.Dynamics` in a `Simulator`.  It
steps the whole MuJoCo world, so whatever else is in the model (a lift,
Pins, a field) moves with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import mujoco
import numpy as np

from . import foam
from .chassis import Chassis, WheelUnit
from .mujoco_checks import raise_on_warnings
from .dynamics import (BodyState, CREEP_SPEED, KAPPA_CLAMP, OMEGA_EPS, RELAX_TAU_MAX, V_FLOOR, FootprintEdge,
                       Tilt, footprint_edges)
from .units import INCH, clamp
from .wheels import tire_forces

#: Pressing foam in takes (1 + FOAM_HYSTERESIS) of the force it gives back at
#: the same depth coming out, (1 - FOAM_HYSTERESIS): a press and release then
#: gives back foam.FOAM_ENERGY_LOSS less than it took.
FOAM_HYSTERESIS = foam.FOAM_ENERGY_LOSS / (2.0 - foam.FOAM_ENERGY_LOSS)
#: m/s of pressing in or coming out over which the foam goes from one to the other
FOAM_PRESS_SPEED = 0.005
#: N/m, the spring that presses a tracking wheel down, and how far it can travel up or down
TRACKING_SPRING = 630.0
TRACKING_TRAVEL = 0.25 * INCH
#: Leaning less than this, the robot is on its wheels and the angle it falls over at is not worked out again
FALL_ANGLE_RECHECK = math.radians(1.0)
WHEEL_RIM_POINTS = 24
#: With this many bodies moving on their own (the robot and a stack of two
#: pieces in its claw, say), MuJoCo checks hundreds of pairs of convex parts
#: a step, and COLLISION_THREADS threads check them about 30% faster.  With
#: fewer, one thread is faster.  Either way the step comes out the same.
BUSY_TREES = 3
COLLISION_THREADS = 4
_collision_workers: List[mujoco.MjData] = []


def collision_workers() -> int:
    """The address of MuJoCo's worker threads, made once for the process,
    for a world's `MjData.threadpool` while it is busy."""
    if not _collision_workers:
        holder = mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>"))
        mujoco.mju_threadpool(holder, COLLISION_THREADS)
        _collision_workers.append(holder)
    return _collision_workers[0].threadpool

GUESSES: Dict[str, str] = {
    "FOAM_PRESS_SPEED": "5 mm/s: the foam's push goes from its coming-out value to its pressing-in value "
                        "over this speed, so a wheel at rest is not pushed by one or the other at random.",
    "TRACKING_SPRING": "630 N/m: a tracking wheel's spring presses with its preload (4 N for "
                       "chassis.add_tracking_wheels) after 0.25 in of travel, and it moves 0.25 in either way.",
    "Wheel outline": "A circle the wheel's diameter. An omni's rollers make its outline a polygon with "
                     "gaps; the foam and the tire model average that out, as they did on the plain drive.",
    "Tread": "A wheel's tread bends over the tire's relaxation length (30 mm on the omnis) to give its "
             "slip, so standing still it is a spring: 10 N on a 2.75 in omni carrying 13 N bends it about 2 mm. "
             "Nothing damps it but the motors, and the robot rocks on it at about 12 Hz, a few thousandths "
             "of an inch, after it stops.",
}


@dataclass(frozen=True)
class Lump:
    """Part of the robot's mass: a solid box `half` metres either way of
    `center`, in the robot frame (x forward, y left, z up from the floor)."""
    name: str
    mass: float
    center: Tuple[float, float, float]
    half: Tuple[float, float, float]


def foam_load(sink: float, radius: float, width: float) -> float:
    """What the foam pushes back on a wheel pressed `sink` metres into it with, at rest (`foam.sink` turned round)."""
    if sink <= 0.0:
        return 0.0
    return 4.0 / 3.0 * foam.FOAM_STIFFNESS * width * math.sqrt(2.0 * radius) * sink ** 1.5


def tread_slip(slip: float, slip_speed: float, rolling_speed: float, relaxation_length: float, dt: float) -> float:
    """A wheel's longitudinal slip `dt` on: how far its tread is bent, over
    the relaxation length.  The tread bends as fast as the wheel's surface
    runs ahead of the floor (`slip_speed`) and straightens as fast as the
    wheel rolls or spins (`rolling_speed`), so rolling it settles to the
    slip ratio, and a stalled wheel pushing on something holds its push
    without turning."""
    straightening = rolling_speed / relaxation_length
    if straightening * dt < 1e-12:
        bent = slip + slip_speed / relaxation_length * dt
    else:
        kept = math.exp(-straightening * dt)
        bent = slip * kept + slip_speed / rolling_speed * (1.0 - kept)
    return clamp(bent, -KAPPA_CLAMP, KAPPA_CLAMP)


def wheel_body_name(robot: str, unit: WheelUnit) -> str:
    return f"{robot} wheel {unit.name}"


def axle_direction(unit: WheelUnit) -> List[float]:
    return [-math.sin(unit.steer_angle), math.cos(unit.steer_angle), 0.0]


def add_drivetrain(spec: mujoco.MjSpec, chassis: Chassis, lumps: Sequence[Lump], robot: str):
    """The chassis body, its `lumps` of mass, and a body for each wheel.
    Returns the chassis body, for the robot's other parts to go on."""
    body = spec.worldbody.add_body(name=robot)
    body.add_freejoint()
    body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
    for lump in lumps:
        body.add_geom(name=f"{robot} {lump.name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(lump.half),
                      pos=list(lump.center), mass=lump.mass, contype=0, conaffinity=0, group=3)
    for unit in chassis.wheel_units:
        wheel = body.add_body(name=wheel_body_name(robot, unit), pos=[unit.x, unit.y, unit.radius])
        if unit.preload > 0.0:
            spring = wheel.add_joint(name=f"{wheel_body_name(robot, unit)} spring", type=mujoco.mjtJoint.mjJNT_SLIDE,
                                     axis=[0.0, 0.0, 1.0], range=[-TRACKING_TRAVEL, TRACKING_TRAVEL])
            spring.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            spring.stiffness = [TRACKING_SPRING, 0.0, 0.0]
            spring.springref = -unit.preload / TRACKING_SPRING
        hinge = wheel.add_joint(name=wheel_body_name(robot, unit), type=mujoco.mjtJoint.mjJNT_HINGE,
                                axis=axle_direction(unit))
        hinge.armature = sum(m.output_inertia * unit.gear_ratio ** 2 for m in unit.motors)
        wheel.explicitinertial = True
        wheel.mass = unit.wheel.mass
        wheel.ipos = [0.0, 0.0, 0.0]
        axle = np.abs(axle_direction(unit))
        across = unit.wheel.inertia / 2.0
        wheel.inertia = (across + (unit.wheel.inertia - across) * axle).tolist()
    return body


@dataclass
class _Wheel:
    unit: WheelUnit
    body: int
    angle_at: int
    speed_at: int
    axle_local: Tuple[float, float, float]


class MujocoDrive:
    """The drivetrain's dynamics on the bodies `add_drivetrain` made, for a
    `Simulator`.  `forces` are called each step after the wheels' forces are
    set and before MuJoCo moves anything, to add the robot's other forces;
    `after` are called once it has."""

    def __init__(self, chassis: Chassis, model: mujoco.MjModel, data: mujoco.MjData, robot: str, gravity: float):
        if abs(gravity - float(-model.opt.gravity[2])) > 1e-9:
            raise ValueError(f"the model's gravity is {-model.opt.gravity[2]} m/s^2, the simulator's {gravity}")
        self.chassis = chassis
        self.model, self.data = model, data
        self.gravity = gravity
        self.robot = robot
        self.body = model.body(robot).id
        joint = model.body_jntadr[self.body]
        self.free_q, self.free_v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        self.wheels: List[_Wheel] = []
        for unit in chassis.wheel_units:
            if 1e-9 < abs(unit.wheel.roller_angle) < math.pi / 2.0 - 1e-9:
                raise ValueError(f"{unit.name} is a mecanum wheel; the MuJoCo drivetrain has omni and traction wheels")
            hinge = model.joint(wheel_body_name(robot, unit)).id
            self.wheels.append(_Wheel(unit, model.body(wheel_body_name(robot, unit)).id, model.jnt_qposadr[hinge],
                                      model.jnt_dofadr[hinge], tuple(model.jnt_axis[hinge].tolist())))
        self._velocity = np.empty(6)
        self._footprint = footprint_edges([(wheel.unit.x, wheel.unit.y) for wheel in self.wheels],
                                          [i for i, wheel in enumerate(self.wheels) if wheel.unit.preload <= 0.0])
        self.tree_bodies = np.nonzero(model.body_rootid == self.body)[0]
        self._shapes = self._shape_points()
        self._rims = self._wheel_rims()
        self.robot_mass = float(model.body_subtreemass[self.body])
        self.before: List[Callable[[], None]] = []
        self.forces: List[Callable[[float], None]] = []
        self.after: List[Callable[[float], None]] = []
        self.locked = False
        self.constrained = False
        self.state = BodyState()
        self.tilt = Tilt()
        self._leaning_toward = None
        self._clear_reports()

    def _clear_reports(self) -> None:
        self.total_fx = self.total_fy = self.total_mz = self.total_normal = 0.0
        self.wheel_slip_power = self.rolling_loss_power = self.drag_power = self.scrub_loss_power = 0.0
        self.normal_moment_residual = (0.0, 0.0)
        self.normal_balance_feasible = True
        self.kinetic_energy = 0.0
        self.any_slipping = False
        self.tipping_margin = 1.0
        self.lateral_transfer_ratio = self.pitch_transfer_ratio = 0.0

    # ------------------------------------------------------------- placing

    def robot_qpos(self) -> np.ndarray:
        """Every position the robot's joints have, the free joint's first."""
        return np.concatenate([self.data.qpos[self.model.jnt_qposadr[j]:][:self._joint_size(j)]
                               for j in self._robot_joints()])

    def _robot_joints(self) -> List[int]:
        return [j for j in range(self.model.njnt) if self.model.body_rootid[self.model.jnt_bodyid[j]] == self.body]

    def _joint_size(self, joint: int) -> int:
        return {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(
            mujoco.mjtJoint(self.model.jnt_type[joint]), 1)

    def place(self, rest: np.ndarray, x: float, y: float, theta: float) -> None:
        """Put the robot at (x, y) heading `theta`, in the pose `rest` has it
        in (from `robot_qpos` of a robot at rest at the origin heading +x),
        standing still."""
        d = self.data
        offset = 0
        for joint in self._robot_joints():
            size = self._joint_size(joint)
            d.qpos[self.model.jnt_qposadr[joint]:][:size] = rest[offset:offset + size]
            offset += size
        q = self.free_q
        heading = np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])
        turned = np.empty(4)
        mujoco.mju_mulQuat(turned, heading, d.qpos[q + 3:q + 7].copy())
        c, s = math.cos(theta), math.sin(theta)
        rx, ry = d.qpos[q], d.qpos[q + 1]
        d.qpos[q:q + 2] = (x + c * rx - s * ry, y + s * rx + c * ry)
        d.qpos[q + 3:q + 7] = turned
        dofs = [self.model.jnt_dofadr[j] for j in self._robot_joints()]
        d.qvel[dofs[0]:dofs[0] + 6] = 0.0
        d.qvel[dofs[1:]] = 0.0
        mujoco.mj_forward(self.model, d)
        self._clear_reports()
        self.tilt = Tilt()
        self._leaning_toward = None
        for wheel in self.wheels:
            unit = wheel.unit
            unit.omega = unit.kappa = unit.tan_alpha = 0.0
            unit.angle = float(d.qpos[wheel.angle_at])
            unit.sync_motors()
        self._read_state(0.0, first=True)

    def _shape_points(self) -> List[Tuple[int, np.ndarray]]:
        """Every corner of the robot's solid parts that can rest on the
        floor, in the frame of the geom it is on."""
        m = self.model
        out: List[Tuple[int, np.ndarray]] = []
        for geom in np.nonzero(np.isin(m.geom_bodyid, self.tree_bodies) & (m.geom_contype != 0))[0]:
            if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[geom]
                points = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            elif m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                points = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[geom]
            else:
                raise ValueError(f"the robot's {m.geom(geom).name} is a {mujoco.mjtGeom(m.geom_type[geom]).name}; "
                                 "its shapes are meshes and boxes")
            out.append((int(geom), np.array(points, dtype=float)))
        return out

    def _wheel_rims(self) -> List[Tuple[_Wheel, np.ndarray]]:
        """Points round a circle of radius 1 in the plane of each wheel that
        carries the robot, in the wheel body's frame.  A tracking wheel's
        spring gives way under the robot, so its wheel is left out, as it is
        from the footprint."""
        turns = np.linspace(0.0, 2.0 * math.pi, WHEEL_RIM_POINTS, endpoint=False)
        up = np.array([0.0, 0.0, 1.0])
        rims = []
        for wheel in (wheel for wheel in self.wheels if wheel.unit.preload <= 0.0):
            along = np.cross(np.array(wheel.axle_local), up)
            rims.append((wheel, np.outer(np.cos(turns), along) + np.outer(np.sin(turns), up)))
        return rims

    def fall_angle(self, toward: Tuple[float, float]) -> float:
        """How far the robot, shaped as it is now, leans toward `toward` (a
        level direction in the chassis' frame) before its centre of mass is
        over the point of it that touches the floor: tipped by an angle, it
        rests on its lowest point, so the angle goes up to where the
        centre of mass is over that point until the lowest point stays the
        same.  It starts from FALL_ANGLE_RECHECK, where the points level
        with each other at rest are already apart.  Its wheels rest sunk as
        far into the foam as they sink when the wheels on that side carry
        the whole robot, so a solid part a little above their bottoms can
        reach the floor first."""
        d = self.data
        turn = d.xmat[self.body].reshape(3, 3)
        origin = d.xpos[self.body]
        com = turn.T @ (d.subtree_com[self.body] - origin)
        world = []
        for geom, points in self._shapes:
            world.append(d.geom_xpos[geom] + points @ d.geom_xmat[geom].reshape(3, 3).T)
        edge = max(self._footprint, key=lambda e: e.nx * toward[0] + e.ny * toward[1])
        load = self.robot_mass * self.gravity / len(edge.wheels)
        for wheel, circle in self._rims:
            unit = wheel.unit
            radius = unit.radius - foam.sink(load, unit.radius, unit.wheel.tire.contact_width)
            world.append(d.xpos[wheel.body] + radius * circle @ d.xmat[wheel.body].reshape(3, 3).T)
        body = (np.concatenate(world) - origin) @ turn - com
        ahead = body[:, 0] * toward[0] + body[:, 1] * toward[1]
        up = body[:, 2]
        angle = FALL_ANGLE_RECHECK
        while True:
            lowest = int(np.argmax(ahead * math.sin(angle) - up * math.cos(angle)))
            over = math.atan2(ahead[lowest], -up[lowest])
            if over <= angle + 1e-9:
                return angle
            angle = over

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        raise ValueError("the robot's pose is set where the world is made: set it with Simulator.set_pose "
                         "before making the world, or use MujocoDrive.place")

    # ------------------------------------------------------------- stepping

    def step(self, dt: float) -> None:
        m, d = self.model, self.data
        if abs(dt - m.opt.timestep) > 1e-12:
            raise ValueError(f"the world steps every {m.opt.timestep} s, not {dt}")
        if self.locked or self.constrained:
            raise ValueError("a robot in a MuJoCo world cannot be locked or constrained in place")
        for prepare in self.before:
            prepare()
        if d.ntree_awake >= BUSY_TREES:
            d.threadpool = collision_workers()
        try:
            mujoco.mj_step1(m, d)
            d.qfrc_applied[:] = 0.0
            self.push(dt)
            mujoco.mj_step2(m, d)
        finally:
            d.threadpool = 0
        raise_on_warnings(m, d)
        self.moved(dt)

    def push(self, dt: float) -> None:
        """Set this robot's forces for the step MuJoCo is about to take: the
        wheels', the drag, and those `forces` add.  `step` calls it; a
        robot in a world another drive steps calls it from that drive's
        `forces`."""
        self.data.xfrc_applied[self.tree_bodies] = 0.0
        self._wheel_forces(dt)
        self._drag()
        for add in self.forces:
            add(dt)

    def moved(self, dt: float) -> None:
        """Read back what MuJoCo's step did to the robot: the wheels' spin, its
        state and how far it leans, then run `after`."""
        d = self.data
        for wheel in self.wheels:
            wheel.unit.omega = float(d.qvel[wheel.speed_at])
            wheel.unit.angle = float(d.qpos[wheel.angle_at])
            wheel.unit.sync_motors()
        self._read_state(dt)
        self._watch_tilt()
        for then in self.after:
            then(dt)

    def point_velocity(self, body: int, point) -> Tuple[float, float, float]:
        """World velocity of `point` (world frame) moving with `body`."""
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, body, self._velocity, 0)
        w0, w1, w2, v0, v1, v2 = self._velocity.tolist()
        origin = self.data.xpos[body].tolist()
        return _plus(v0, v1, v2, _cross(w0, w1, w2, point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]))

    def _apply(self, body: int, force, torque, point) -> None:
        """`force` at `point` and `torque` on `body`, world frame."""
        com = self.data.xipos[body].tolist()
        moment = _cross(point[0] - com[0], point[1] - com[1], point[2] - com[2], *force)
        self.data.xfrc_applied[body] += (force[0], force[1], force[2], torque[0] + moment[0],
                                         torque[1] + moment[1], torque[2] + moment[2])

    def _wheel_forces(self, dt: float) -> None:
        d, ch = self.data, self.chassis
        chassis_turn = d.xmat[self.body].tolist()
        chassis_origin = d.xpos[self.body].tolist()
        loads = []
        fx_sum = fy_sum = mz_sum = slip_power = roll_power = scrub_power = 0.0
        self.any_slipping = False
        share = self.robot_mass * self.gravity / max(len(self.wheels), 1)
        yaw_rate = float(d.qvel[self.free_v + 5])
        sqrt, tanh = math.sqrt, math.tanh
        for wheel in self.wheels:
            unit, tire = wheel.unit, wheel.unit.wheel.tire
            r = unit.radius
            turn = d.xmat[wheel.body].tolist()
            a0, a1, a2 = wheel.axle_local
            axle = (turn[0] * a0 + turn[1] * a1 + turn[2] * a2, turn[3] * a0 + turn[4] * a1 + turn[5] * a2,
                    turn[6] * a0 + turn[7] * a1 + turn[8] * a2)
            down = (axle[0] * axle[2], axle[1] * axle[2], axle[2] * axle[2] - 1.0)
            length = sqrt(down[0] ** 2 + down[1] ** 2 + down[2] ** 2)
            center = d.xpos[wheel.body].tolist()
            arm = (r * down[0] / length, r * down[1] / length, r * down[2] / length)
            patch = (center[0] + arm[0], center[1] + arm[1], center[2] + arm[2])
            level = sqrt(axle[0] ** 2 + axle[1] ** 2)
            roll = (axle[1] / level, -axle[0] / level, 0.0)
            lateral = (-roll[1], roll[0], 0.0)
            spin = float(d.qvel[wheel.speed_at])
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, wheel.body, self._velocity, 0)
            w0, w1, w2, v0, v1, v2 = self._velocity.tolist()
            ground = _plus(v0, v1, v2, _cross(w0 - spin * axle[0], w1 - spin * axle[1], w2 - spin * axle[2], *arm))
            surface = spin * r
            vx = ground[0] * roll[0] + ground[1] * roll[1]
            vy = ground[0] * lateral[0] + ground[1] * lateral[1]
            fz = foam_load(-patch[2], r, tire.contact_width) * (
                1.0 + FOAM_HYSTERESIS * tanh(-ground[2] / FOAM_PRESS_SPEED))
            unit.fz = fz
            unit.vx, unit.vy = vx, vy
            denom = max(abs(vx), abs(surface), V_FLOOR)
            kappa_ss = clamp((surface - vx) / denom, -KAPPA_CLAMP, KAPPA_CLAMP)
            tan_ss = clamp(vy / denom, -KAPPA_CLAMP, KAPPA_CLAMP)
            unit.kappa_ss, unit.tan_alpha_ss = kappa_ss, tan_ss
            rate = max(denom / max(tire.relaxation_length, 1e-4), 1.0 / RELAX_TAU_MAX)
            decay = 1.0 - math.exp(-rate * dt)
            unit.tan_alpha += (tan_ss - unit.tan_alpha) * decay
            unit.kappa = tread_slip(unit.kappa, surface - vx, max(abs(vx), abs(surface)), tire.relaxation_length, dt)
            fx, fy = tire_forces(tire, fz, unit.kappa, unit.tan_alpha, ch.surface_mu)
            fy *= tanh(math.hypot(surface - vx, vy) / CREEP_SPEED)
            unit.fx, unit.fy = fx, fy
            measure = abs(unit.kappa) / tire.slip_ratio_peak
            if abs(unit.wheel.roller_angle) < 0.1:
                measure = max(measure, abs(unit.tan_alpha) / tire.slip_angle_peak)
            unit.slip_measure = measure
            unit.slipping = measure > 1.5 and fz > 0.2 * share
            self.any_slipping = self.any_slipping or unit.slipping
            scrub = tire.scrub_arm(fz, r) * fz * tanh(yaw_rate / 0.2)
            rolling = tire.rolling_resistance(fz, r) * fz * r * tanh(spin / OMEGA_EPS)
            unit.roll_resist_torque = rolling
            grip = (fx * roll[0] + fy * lateral[0], fx * roll[1] + fy * lateral[1], 0.0)
            self._apply(wheel.body, (grip[0], grip[1], fz),
                        (-rolling * axle[0], -rolling * axle[1], -rolling * axle[2] - scrub), patch)
            if unit.driven:
                d.qfrc_applied[wheel.speed_at] += unit.motor_torque()
            forward = grip[0] * chassis_turn[0] + grip[1] * chassis_turn[3]
            leftward = grip[0] * chassis_turn[1] + grip[1] * chassis_turn[4]
            offset = (patch[0] - chassis_origin[0], patch[1] - chassis_origin[1], patch[2] - chassis_origin[2])
            ahead = offset[0] * chassis_turn[0] + offset[1] * chassis_turn[3] + offset[2] * chassis_turn[6]
            beside = offset[0] * chassis_turn[1] + offset[1] * chassis_turn[4] + offset[2] * chassis_turn[7]
            fx_sum += forward
            fy_sum += leftward
            mz_sum += ahead * leftward - beside * forward - scrub
            slip_power += abs(fx * (surface - vx)) + abs(fy * vy)
            roll_power += abs(rolling * spin)
            scrub_power += scrub * yaw_rate
            loads.append(fz)
        self.total_fx, self.total_fy, self.total_mz = fx_sum, fy_sum, mz_sum
        self.total_normal = sum(loads)
        self.wheel_slip_power = slip_power
        self.rolling_loss_power = roll_power
        self.scrub_loss_power = scrub_power
        self._load_transfer(loads)

    def _drag(self) -> None:
        """The chassis's bearing, air and yaw drag (`chassis.Chassis`)."""
        st, ch = self.state, self.chassis
        along = ch.bearing_drag * st.u + ch.aero_drag * st.u * abs(st.u)
        across = ch.bearing_drag * st.v
        yaw = ch.yaw_drag * st.r
        c, s = math.cos(st.theta), math.sin(st.theta)
        self._apply(self.body, (-(along * c - across * s), -(along * s + across * c), 0.0), (0.0, 0.0, -yaw),
                    self.data.xipos[self.body].tolist())
        self.drag_power = abs(along * st.u) + abs(across * st.v) + abs(yaw * st.r) + self.scrub_loss_power

    def _load_transfer(self, loads: List[float]) -> None:
        total = sum(loads) or 1.0
        units = [wheel.unit for wheel in self.wheels]
        left = sum(fz for unit, fz in zip(units, loads) if unit.y > 0.0)
        right = sum(fz for unit, fz in zip(units, loads) if unit.y < 0.0)
        front = sum(fz for unit, fz in zip(units, loads) if unit.x > 0.0)
        rear = sum(fz for unit, fz in zip(units, loads) if unit.x < 0.0)
        self.lateral_transfer_ratio = abs(left - right) / total
        self.pitch_transfer_ratio = abs(front - rear) / total
        self.tipping_margin = clamp(1.0 - max(self.lateral_transfer_ratio, self.pitch_transfer_ratio), 0.0, 1.0)

    # ------------------------------------------------------------- reading

    def _read_state(self, dt: float, first: bool = False) -> None:
        d, st = self.data, self.state
        q, v = self.free_q, self.free_v
        turn = d.xmat[self.body].reshape(3, 3)
        theta = math.atan2(turn[1, 0], turn[0, 0])
        st.theta = theta if first else st.theta + math.remainder(theta - st.theta, 2.0 * math.pi)
        st.x, st.y = float(d.qpos[q]), float(d.qpos[q + 1])
        c, s = math.cos(theta), math.sin(theta)
        velocity, acceleration = d.qvel[v:v + 3], d.qacc[v:v + 3]
        st.u, st.v = c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]
        st.ax, st.ay = c * acceleration[0] + s * acceleration[1], -s * acceleration[0] + c * acceleration[1]
        yaw_rate = float(d.qvel[v + 5])
        st.alpha = 0.0 if first else (yaw_rate - st.r) / dt
        st.r = yaw_rate
        self.kinetic_energy = 0.5 * self.robot_mass * (st.u ** 2 + st.v ** 2) + 0.5 * self.chassis.inertia_z * st.r ** 2

    def _watch_tilt(self) -> None:
        """How far the chassis leans off level, toward which side of its
        wheels, and whether its centre of mass has gone past the point of it
        it would rest on (`fall_angle`)."""
        tilt = self.tilt
        if tilt.fallen:
            return
        turn = self.data.xmat[self.body].tolist()
        lean = math.acos(clamp(turn[8], -1.0, 1.0))
        com = self.data.subtree_com[self.body].tolist()
        origin = self.data.xpos[self.body].tolist()
        offset = (com[0] - origin[0], com[1] - origin[1], com[2] - origin[2])
        ahead = offset[0] * turn[0] + offset[1] * turn[3] + offset[2] * turn[6]
        beside = offset[0] * turn[1] + offset[1] * turn[4] + offset[2] * turn[7]
        side = max(self._footprint, key=lambda e: -(e.nx * turn[6] + e.ny * turn[7]))
        edge = FootprintEdge(side.nx, side.ny, side.reach - side.nx * ahead - side.ny * beside, side.wheels)
        if side is not self._leaning_toward or lean >= FALL_ANGLE_RECHECK:
            tilt.balance = self.fall_angle((side.nx, side.ny))
            self._leaning_toward = side
        tilt.edge = edge
        tilt.angle = lean
        if lean > tilt.most:
            tilt.most, tilt.most_edge, tilt.most_balance = lean, edge, tilt.balance
        tilt.fallen = lean >= tilt.balance


def settle(drive: MujocoDrive, seconds: float) -> None:
    """Step the world with the motors off until `seconds` have gone by."""
    for _ in range(int(round(seconds / drive.model.opt.timestep))):
        drive.step(drive.model.opt.timestep)


def _cross(a0: float, a1: float, a2: float, b0: float, b1: float, b2: float) -> Tuple[float, float, float]:
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


def _plus(x: float, y: float, z: float, other) -> Tuple[float, float, float]:
    return x + other[0], y + other[1], z + other[2]

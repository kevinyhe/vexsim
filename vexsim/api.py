"""A VEX-shaped API on top of the simulator.

Auton code reads sequentially -- drive a bit, turn, drive again -- so the
blocking calls here read the same way.  Each one runs the physics forward
internally at the control period, so `robot.drive_for(24)` really does simulate
every millisecond of driving 24 inches and leaves the log behind.

    robot = Robot(presets.six_motor_600())
    robot.drive_for(24)
    robot.turn_for(90)
    robot.drive_for(-12)
    print(robot.report())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .battery import Battery
from .chassis import Chassis
from .control import (PID, Feedforward, PurePursuit, TrapezoidProfile,
                      angle_error, arcade)
from .motor import BrakeMode
from .sim import SimConfig, Simulator
from .units import DEG, INCH, clamp, to_deg, to_inches, wrap_pi


@dataclass
class DriveGains:
    """Tuning for the built-in motions.  Volts out, metres and radians in."""

    drive_kp: float = 34.0        # V per metre of error
    drive_ki: float = 10.0
    drive_kd: float = 2.6
    drive_tolerance: float = 0.010     # m
    drive_velocity_tolerance: float = 0.05   # m/s

    turn_kp: float = 12.0         # V per radian of error
    turn_ki: float = 10.0
    turn_kd: float = 1.0
    turn_tolerance: float = 1.5 * DEG
    turn_velocity_tolerance: float = 8.0 * DEG

    heading_kp: float = 6.0       # V per radian, holds heading while driving

    max_volts: float = 12.0
    slew: float = 90.0            # V/s on the output, 0 = off
    settle_time: float = 0.15     # s inside tolerance before it counts as done


class Robot:
    """Drivetrain plus the routines you would write against it."""

    def __init__(self, chassis: Chassis, battery: Optional[Battery] = None,
                 config: Optional[SimConfig] = None,
                 gains: Optional[DriveGains] = None,
                 use_odometry: bool = True):
        self.sim = Simulator(chassis, battery, config)
        self.gains = gains or DriveGains()
        self.use_odometry = use_odometry
        self.feedforward = Feedforward.from_chassis(chassis)
        self.brake_mode = BrakeMode.COAST
        self._left_cmd = 0.0
        self._right_cmd = 0.0
        self.timed_out = False

    # ------------------------------------------------------------ properties
    @property
    def chassis(self) -> Chassis:
        return self.sim.chassis

    @property
    def log(self):
        return self.sim.log

    @property
    def t(self) -> float:
        return self.sim.t

    @property
    def heading(self) -> float:
        """Radians, as the robot believes them (IMU)."""
        return self.sim.imu.heading if self.use_odometry else self.sim.state.theta

    @property
    def heading_deg(self) -> float:
        return to_deg(self.heading)

    @property
    def position(self) -> Tuple[float, float]:
        if self.use_odometry:
            return self.sim.odometry.x, self.sim.odometry.y
        return self.sim.state.x, self.sim.state.y

    @property
    def true_position(self) -> Tuple[float, float]:
        return self.sim.state.x, self.sim.state.y

    @property
    def true_heading_deg(self) -> float:
        return to_deg(self.sim.state.theta)

    def travelled(self) -> float:
        """Metres of forward travel as the drive encoders see it."""
        left = self.sim._side_travel(self.chassis.left_units())
        right = self.sim._side_travel(self.chassis.right_units())
        return (left + right) / 2.0

    # ------------------------------------------------------------- open loop
    def tank(self, left_volts: float, right_volts: float) -> None:
        self._left_cmd = clamp(left_volts, -12.0, 12.0)
        self._right_cmd = clamp(right_volts, -12.0, 12.0)
        for u in self.chassis.left_units():
            for m in u.motors:
                m.set_voltage(self._left_cmd)
        for u in self.chassis.right_units():
            for m in u.motors:
                m.set_voltage(self._right_cmd)

    def arcade(self, forward_volts: float, turn_volts: float,
               desaturate: str = "scale") -> None:
        l, r = arcade(forward_volts, turn_volts, self.gains.max_volts, desaturate)
        self.tank(l, r)

    def tank_percent(self, left_pct: float, right_pct: float) -> None:
        self.tank(left_pct * 0.12, right_pct * 0.12)

    def stop(self, brake_mode: Optional[BrakeMode] = None) -> None:
        self.sim.stop_all(brake_mode or self.brake_mode)
        self._left_cmd = self._right_cmd = 0.0

    def set_pose(self, x_in: float = 0.0, y_in: float = 0.0,
                 heading_deg: float = 0.0) -> None:
        self.sim.set_pose(x_in * INCH, y_in * INCH, heading_deg * DEG)

    # ------------------------------------------------------------ time steps
    def wait(self, seconds: float) -> None:
        self.sim.run(seconds)

    def _loop(self, tick: Callable[[float], bool], timeout: float) -> bool:
        """Run `tick` every control period until it returns True or time runs out.

        `tick` gets the control period as its argument and returns done-ness.
        Returns True if it finished, False on timeout.
        """
        cfg = self.sim.config
        period = cfg.control_period
        substeps = max(1, int(round(period / cfg.dt)))
        elapsed = 0.0
        self.timed_out = False
        while elapsed < timeout:
            if tick(period):
                return True
            for _ in range(substeps):
                self.sim.step()
            elapsed += period
        self.timed_out = True
        return False

    # ------------------------------------------------------------- closed loop
    def drive_for(self, distance_in: float, max_volts: Optional[float] = None,
                  timeout: float = 5.0, hold_heading: bool = True,
                  heading_deg: Optional[float] = None) -> bool:
        """Drive a distance with a PID on encoder travel and a heading hold."""
        g = self.gains
        max_v = max_volts if max_volts is not None else g.max_volts
        target = distance_in * INCH
        start = self.travelled()
        hold = (heading_deg * DEG) if heading_deg is not None else self.heading

        pid = PID(g.drive_kp, g.drive_ki, g.drive_kd,
                  integral_limit=3.0, integral_zone=0.12,
                  output_limit=max_v, slew=g.slew)
        hpid = PID(g.heading_kp, 0.0, 0.35, output_limit=max_v)
        settle = 0.0

        def tick(dt: float) -> bool:
            nonlocal settle
            travelled = self.travelled() - start
            out = pid.update(target, travelled, dt)
            turn = hpid.update(0.0, -angle_error(hold, self.heading), dt) if hold_heading else 0.0
            self.arcade(out, turn)
            if abs(pid.error) < g.drive_tolerance and \
                    abs(self.sim.state.u) < g.drive_velocity_tolerance:
                settle += dt
            else:
                settle = 0.0
            return settle >= g.settle_time

        ok = self._loop(tick, timeout)
        self.stop(BrakeMode.BRAKE)
        return ok

    def turn_for(self, degrees: float, max_volts: Optional[float] = None,
                 timeout: float = 4.0) -> bool:
        """Turn relative to the current heading."""
        return self.turn_to(self.heading_deg + degrees, max_volts, timeout)

    def turn_to(self, heading_deg: float, max_volts: Optional[float] = None,
                timeout: float = 4.0) -> bool:
        g = self.gains
        max_v = max_volts if max_volts is not None else g.max_volts
        target = heading_deg * DEG
        pid = PID(g.turn_kp, g.turn_ki, g.turn_kd,
                  integral_limit=3.0, integral_zone=10.0 * DEG,
                  output_limit=max_v, slew=g.slew)
        settle = 0.0

        def tick(dt: float) -> bool:
            nonlocal settle
            err = angle_error(target, self.heading)
            out = pid.update(0.0, -err, dt)
            self.arcade(0.0, out)
            if abs(err) < g.turn_tolerance and abs(self.sim.state.r) < g.turn_velocity_tolerance:
                settle += dt
            else:
                settle = 0.0
            return settle >= g.settle_time

        ok = self._loop(tick, timeout)
        self.stop(BrakeMode.BRAKE)
        return ok

    def drive_profiled(self, distance_in: float, max_velocity_mps: Optional[float] = None,
                       max_accel: float = 3.0, timeout: Optional[float] = None,
                       kp: float = 25.0) -> bool:
        """Follow a trapezoid profile with feedforward plus a small correction.

        This is the motion that actually looks smooth: the feedforward does the
        work and the PID only cleans up the difference.
        """
        top = max_velocity_mps or (0.85 * self.chassis.theoretical_free_speed())
        profile = TrapezoidProfile(distance_in * INCH, top, max_accel)
        start = self.travelled()
        hold = self.heading
        hpid = PID(self.gains.heading_kp, 0.0, 0.35, output_limit=6.0)
        t0 = self.sim.t
        timeout = timeout or profile.duration + 1.5

        def tick(dt: float) -> bool:
            tt = self.sim.t - t0
            pos, vel, acc = profile.sample(tt)
            travelled = self.travelled() - start
            volts = self.feedforward.calculate(vel, acc) + kp * (pos - travelled)
            turn = hpid.update(0.0, -angle_error(hold, self.heading), dt)
            self.arcade(clamp(volts, -12.0, 12.0), turn)
            return tt > profile.duration and abs(self.sim.state.u) < 0.05

        ok = self._loop(tick, timeout)
        self.stop(BrakeMode.BRAKE)
        return ok

    def turn_to_point(self, x_in: float, y_in: float, timeout: float = 4.0) -> bool:
        px, py = self.position
        target = math.atan2(y_in * INCH - py, x_in * INCH - px)
        return self.turn_to(to_deg(target), timeout=timeout)

    def drive_to_point(self, x_in: float, y_in: float, max_volts: float = 12.0,
                       timeout: float = 6.0, tolerance_in: float = 1.0) -> bool:
        """Boomerang-free version: point at the target, then drive to it."""
        self.turn_to_point(x_in, y_in, timeout=timeout / 2.0)
        px, py = self.position
        dist = math.hypot(x_in * INCH - px, y_in * INCH - py)
        return self.drive_for(to_inches(dist), max_volts, timeout=timeout)

    def follow_path(self, points_in: Sequence[Tuple[float, float]],
                    velocity_mps: float = 0.8, lookahead_in: float = 10.0,
                    timeout: float = 12.0, turn_gain: float = 4.0) -> bool:
        """Pure pursuit over a list of (x, y) waypoints in inches."""
        path = [(x * INCH, y * INCH) for x, y in points_in]
        pp = PurePursuit(path, lookahead=lookahead_in * INCH)
        base = self.feedforward.calculate(velocity_mps)

        def tick(dt: float) -> bool:
            x, y = self.position
            if pp.finished(x, y):
                return True
            k = pp.curvature(x, y, self.heading)
            self.arcade(base, clamp(turn_gain * k * velocity_mps, -6.0, 6.0))
            return False

        ok = self._loop(tick, timeout)
        self.stop(BrakeMode.BRAKE)
        return ok

    def drive_seconds(self, seconds: float, left_volts: float,
                      right_volts: Optional[float] = None) -> None:
        """Open-loop timed drive, the thing every first auton is made of."""
        self.tank(left_volts, right_volts if right_volts is not None else left_volts)
        self.sim.run(seconds)
        self.stop()

    def push(self, seconds: float, volts: float = 12.0) -> float:
        """Hold full power against something.  Returns the average push force, N."""
        if seconds <= 0.0:
            return 0.0

        dynamics = self.sim.dynamics
        was_locked = dynamics.locked
        was_constrained = dynamics.constrained
        impulse = 0.0
        elapsed = 0.0
        self.tank(volts, volts)
        dynamics.locked = False
        dynamics.constrained = True
        try:
            while elapsed < seconds:
                dt = min(self.sim.config.dt, seconds - elapsed)
                self.sim.step(dt)
                impulse += dynamics.total_fx * dt
                elapsed += dt
        finally:
            dynamics.locked = was_locked
            dynamics.constrained = was_constrained
            self.stop()
        return impulse / seconds

    # ---------------------------------------------------------------- output
    def report(self) -> dict:
        s = self.sim
        st = s.state
        ms = s.motors
        log = s.log
        out = {
            "time_s": s.t,
            "true_x_in": to_inches(st.x),
            "true_y_in": to_inches(st.y),
            "true_heading_deg": to_deg(st.theta),
            "odom_x_in": to_inches(s.odometry.x),
            "odom_y_in": to_inches(s.odometry.y),
            "odom_heading_deg": to_deg(s.odometry.theta),
            "odom_position_error_in": to_inches(
                math.hypot(s.odometry.x - st.x, s.odometry.y - st.y)),
            "imu_heading_error_deg": to_deg(wrap_pi(s.imu.heading - st.theta)),
            "battery_soc": s.battery.soc,
            "battery_mAh_used": s.battery.charge_used_ah * 1000.0,
            "hottest_motor_C": max((m.temperature for m in ms), default=0.0),
            "timed_out": self.timed_out,
        }
        if log.n:
            out.update({
                "peak_speed_inps": to_inches(log.max("speed_mps")),
                "peak_accel_g": log.max("accel_fwd_g"),
                "peak_current_A": log.max("battery_A"),
                "min_battery_V": log.min("battery_V"),
                "peak_power_W": log.max("battery_W"),
                "avg_power_W": log.mean("battery_W"),
                "energy_J": s.energy_battery,
                "slip_energy_J": s.energy_slip,
                "wheel_slip_fraction": log.mean("any_wheel_slipping"),
                "min_tipping_margin": log.min("tipping_margin"),
            })
        return out

    def print_report(self) -> None:
        for k, v in self.report().items():
            if isinstance(v, float):
                print(f"  {k:28s} {v:12.4f}")
            else:
                print(f"  {k:28s} {v!s:>12}")

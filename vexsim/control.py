"""Controllers and motion planning, the parts an auton is actually built from.

Everything here works in SI and is deliberately plain: a PID that handles the
things teams get wrong (integral windup, derivative kick, a settling test that
looks at error *and* velocity), a trapezoid profile, a feedforward model that
knows the drivetrain's kS/kV/kA, and pure pursuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .units import clamp, sign, wrap_pi


@dataclass
class PID:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 1e9      # clamp on the integral term's output
    integral_zone: float = 1e9       # only integrate inside this error band
    output_limit: float = 12.0
    slew: float = 0.0                # max output change per second, 0 = off
    derivative_on_measurement: bool = True   # kills derivative kick on setpoint jumps

    integral: float = 0.0
    last_error: float = 0.0
    last_measurement: float = 0.0
    last_output: float = 0.0
    error: float = 0.0
    derivative: float = 0.0
    _init: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
        self._init = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        err = setpoint - measurement
        self.error = err

        if not self._init:
            self.last_error = err
            self.last_measurement = measurement
            self._init = True

        if abs(err) < self.integral_zone:
            self.integral += err * dt
        else:
            self.integral = 0.0
        if self.ki > 1e-12:
            lim = self.integral_limit / self.ki
            self.integral = clamp(self.integral, -lim, lim)

        if self.derivative_on_measurement:
            self.derivative = -(measurement - self.last_measurement) / max(dt, 1e-9)
        else:
            self.derivative = (err - self.last_error) / max(dt, 1e-9)

        out = self.kp * err + self.ki * self.integral + self.kd * self.derivative
        out = clamp(out, -self.output_limit, self.output_limit)

        if self.slew > 0.0:
            max_step = self.slew * dt
            out = clamp(out, self.last_output - max_step, self.last_output + max_step)

        self.last_error = err
        self.last_measurement = measurement
        self.last_output = out
        return out

    def settled(self, tolerance: float, velocity: float = 0.0,
                velocity_tolerance: float = 1e9) -> bool:
        return abs(self.error) < tolerance and abs(velocity) < velocity_tolerance


@dataclass
class Feedforward:
    """kS + kV*v + kA*a, the standard drivetrain characterisation, in volts.

    kS is the voltage that just barely gets the robot moving (it fights coulomb
    friction), kV is volts per m/s, kA is volts per m/s^2.  `from_chassis`
    estimates all three from the DC equivalent below current, power and speed
    limits. It is not a hardware characterization or an inverse of the V5
    governor. Requested trajectories must respect those actuator limits.
    """

    ks: float = 0.0
    kv: float = 0.0
    ka: float = 0.0

    def calculate(self, velocity: float, acceleration: float = 0.0) -> float:
        return self.ks * sign(velocity) + self.kv * velocity + self.ka * acceleration

    @staticmethod
    def from_chassis(chassis) -> "Feedforward":
        units = chassis.driven_units()
        if not units:
            return Feedforward()
        u = units[0]
        m = u.motors[0]
        spec = m.spec
        r = u.radius
        # motor-shaft rad/s per m/s of robot speed
        shaft_per_mps = m.ratio * u.gear_ratio / r
        kv = spec.ke * shaft_per_mps
        # volts needed per m/s^2: total inertia against total torque gain
        eff_mass = chassis.mass + sum(
            uu.effective_inertia / uu.radius ** 2 for uu in chassis.wheel_units)
        torque_per_volt = sum(
            spec.kt / spec.resistance * mm.ratio * spec.gearbox_efficiency
            * uu.gear_ratio * uu.efficiency / uu.radius
            for uu in units for mm in uu.motors)
        ka = eff_mass / max(torque_per_volt, 1e-9)
        # kS has to cover rolling resistance plus the motors' own coulomb drag
        share = chassis.weight / len(chassis.wheel_units)
        crr = sum(uu.wheel.tire.rolling_resistance(share, uu.radius) for uu in units) / len(units)
        f_roll = crr * chassis.weight
        f_coulomb = sum(mm.coulomb_torque * uu.gear_ratio * uu.efficiency / uu.radius
                        for uu in units for mm in uu.motors)
        ks = (f_roll + f_coulomb) / max(torque_per_volt, 1e-9)
        return Feedforward(ks=ks, kv=kv, ka=ka)


@dataclass
class TrapezoidProfile:
    """Trapezoid (or triangle) velocity profile over a fixed distance."""

    distance: float
    max_velocity: float
    max_acceleration: float
    max_deceleration: Optional[float] = None
    start_velocity: float = 0.0

    def __post_init__(self):
        self.direction = 1.0 if self.distance >= 0.0 else -1.0
        d = abs(self.distance)
        a = self.max_acceleration
        dec = self.max_deceleration or self.max_acceleration
        v = self.max_velocity

        # can it reach cruise speed?
        d_accel = v * v / (2.0 * a)
        d_decel = v * v / (2.0 * dec)
        if d_accel + d_decel > d:
            v = math.sqrt(2.0 * d * a * dec / (a + dec))
            d_accel = v * v / (2.0 * a)
            d_decel = v * v / (2.0 * dec)
        self.cruise_velocity = v
        self.t_accel = v / a
        self.t_cruise = (d - d_accel - d_decel) / v if v > 1e-9 else 0.0
        self.t_decel = v / dec
        self.duration = self.t_accel + self.t_cruise + self.t_decel
        self._d_accel = d_accel
        self._d_cruise = d - d_accel - d_decel
        self._dec = dec
        self._a = a

    def sample(self, t: float) -> Tuple[float, float, float]:
        """(position, velocity, acceleration) at time t, signed."""
        d = self.direction
        if t <= 0.0:
            return 0.0, 0.0, d * self._a
        if t < self.t_accel:
            return (d * 0.5 * self._a * t * t, d * self._a * t, d * self._a)
        t2 = t - self.t_accel
        if t2 < self.t_cruise:
            return (d * (self._d_accel + self.cruise_velocity * t2),
                    d * self.cruise_velocity, 0.0)
        t3 = t2 - self.t_cruise
        if t3 < self.t_decel:
            v = self.cruise_velocity - self._dec * t3
            pos = self._d_accel + self._d_cruise + \
                (self.cruise_velocity * t3 - 0.5 * self._dec * t3 * t3)
            return d * pos, d * v, -d * self._dec
        return self.distance, 0.0, 0.0


@dataclass
class SlewLimiter:
    rate: float = 1e9     # units per second
    value: float = 0.0

    def update(self, target: float, dt: float) -> float:
        step = self.rate * dt
        self.value = clamp(target, self.value - step, self.value + step)
        return self.value

    def reset(self, value: float = 0.0) -> None:
        self.value = value


@dataclass
class PurePursuit:
    """Follow a polyline by chasing a point a fixed distance ahead."""

    path: List[Tuple[float, float]]
    lookahead: float = 0.30
    finish_tolerance: float = 0.05
    _index: int = 0

    def target(self, x: float, y: float) -> Tuple[float, float]:
        # advance past points we are already beyond
        while self._index < len(self.path) - 1:
            px, py = self.path[self._index]
            if math.hypot(px - x, py - y) < self.lookahead:
                self._index += 1
            else:
                break
        return self.path[min(self._index, len(self.path) - 1)]

    def curvature(self, x: float, y: float, theta: float) -> float:
        tx, ty = self.target(x, y)
        dx, dy = tx - x, ty - y
        # transform into the robot frame
        c, s = math.cos(-theta), math.sin(-theta)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        l2 = lx * lx + ly * ly
        if l2 < 1e-9:
            return 0.0
        return 2.0 * ly / l2

    def finished(self, x: float, y: float) -> bool:
        ex, ey = self.path[-1]
        return (self._index >= len(self.path) - 1
                and math.hypot(ex - x, ey - y) < self.finish_tolerance)

    def reset(self) -> None:
        self._index = 0


def arcade(forward: float, turn: float, max_volts: float = 12.0,
           desaturate: str = "scale") -> Tuple[float, float]:
    """Forward/turn to left/right volts.

    Positive `turn` is counter-clockwise, matching the heading convention, so
    it slows the left side and speeds up the right.

    `desaturate` decides what happens when the sum runs past 12 V:

    "scale" pulls both sides down and keeps the turn ratio exactly, which is
    what a closed-loop routine wants -- a robot at full throttle that clips
    would stop steering.  The cost is that adding turn drops forward speed:
    full forward plus half turn ends up averaging 8 V a side instead of 12.

    "clamp" just clips each side, so forward speed survives and it is the turn
    that saturates.
    """
    left = forward - turn
    right = forward + turn
    if desaturate == "clamp":
        return (clamp(left, -max_volts, max_volts),
                clamp(right, -max_volts, max_volts))
    peak = max(abs(left), abs(right))
    if peak > max_volts:
        left *= max_volts / peak
        right *= max_volts / peak
    return left, right


def angle_error(target: float, current: float) -> float:
    """Shortest signed angle from current to target."""
    return wrap_pi(target - current)

"""Chassis layout: where the wheels are, what drives them, how heavy it is.

A `WheelUnit` is one contact patch plus everything bolted behind it: the wheel,
the external gearing (sprockets or a gear train between the cartridge and the
axle), and the motors on it.  A `Chassis` is a mass with a centre of gravity and
a list of those units, which is enough to describe a 4-motor tank, a 6-motor
drive with a traction wheel in the middle, a mecanum base, or a chassis with
undriven tracking wheels hanging off it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from . import wheels as wheel_presets
from .motor import Motor, v5_11w
from .units import G, INCH, LB
from .wheels import WheelSpec


@dataclass
class WheelUnit:
    """One wheel, its position, its gearing and its motors."""

    wheel: WheelSpec
    x: float                       # metres forward of the geometric centre
    y: float                       # metres left of the geometric centre
    motors: List[Motor] = field(default_factory=list)
    gear_ratio: float = 1.0        # cartridge output turns per wheel turn
    efficiency: float = 0.95       # chain / spur gear efficiency
    name: str = "wheel"
    steer_angle: float = 0.0       # rad, wheel heading relative to the chassis
    preload: float = 0.0           # N, spring preload for undriven odom wheels

    # --- live state ---
    omega: float = 0.0             # wheel spin, rad/s
    angle: float = 0.0             # accumulated wheel angle, rad
    fz: float = 0.0                # normal load, N
    fx: float = 0.0                # tractive force along the wheel, N
    fy: float = 0.0                # lateral force, N
    kappa: float = 0.0             # lagged slip ratio
    tan_alpha: float = 0.0         # lagged lateral slip
    kappa_ss: float = 0.0          # instantaneous (steady state) slip ratio
    tan_alpha_ss: float = 0.0
    vx: float = 0.0                # contact patch ground speed along the wheel
    vy: float = 0.0
    slipping: bool = False
    slip_measure: float = 0.0    # slip as a multiple of the tire's peak
    drive_torque: float = 0.0      # N*m applied at the axle by the motors
    roll_resist_torque: float = 0.0
    power_gear_loss: float = 0.0   # W dissipated in external chain / gearing
    _inertia_key: tuple = field(default=None, init=False, repr=False, compare=False)
    _inertia: float = field(default=0.0, init=False, repr=False, compare=False)

    @property
    def radius(self) -> float:
        return self.wheel.radius

    @property
    def driven(self) -> bool:
        return self.wheel.driven and bool(self.motors)

    @property
    def effective_inertia(self) -> float:
        """Wheel inertia plus everything upstream, reflected to the axle."""
        # Constant for a given build, and read twice per wheel per physics
        # slice, so memoise it against the inputs it actually depends on.
        key = (self.wheel.inertia, self.gear_ratio, len(self.motors))
        if key != self._inertia_key:
            i = self.wheel.inertia
            for m in self.motors:
                i += m.output_inertia * self.gear_ratio ** 2
            self._inertia_key = key
            self._inertia = i
        return self._inertia

    @property
    def free_speed(self) -> float:
        """Wheel speed at 12 V with no load, rad/s."""
        if not self.motors:
            return 0.0
        return self.motors[0].free_speed / self.gear_ratio

    @property
    def max_torque(self) -> float:
        """Axle torque with every motor at its current limit, N*m."""
        return sum(m.stall_torque for m in self.motors) * self.gear_ratio * self.efficiency

    def sync_motors(self) -> None:
        """Push wheel speed/angle back onto the motor shafts."""
        for m in self.motors:
            m.velocity = self.omega * self.gear_ratio
            m.position = self.angle * self.gear_ratio

    def motor_torque(self) -> float:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("external gearing efficiency must be in (0, 1]")
        t = 0.0
        self.power_gear_loss = 0.0
        for m in self.motors:
            ideal_torque = m.output_torque() * self.gear_ratio
            # In braking/backdrive the wheel is the power source. Producing
            # a given absorbed motor torque requires *more* wheel torque,
            # whereas forward power delivery loses torque through the gears.
            if ideal_torque * self.omega < 0.0:
                axle_torque = ideal_torque / self.efficiency
            else:
                axle_torque = ideal_torque * self.efficiency
            self.power_gear_loss += (ideal_torque - axle_torque) * self.omega
            t += axle_torque
        self.drive_torque = t
        return self.drive_torque


@dataclass
class Chassis:
    """Mass properties, wheel layout, and everything that rubs."""

    mass: float = 7.0                     # kg, robot as it rolls onto the field
    wheel_units: List[WheelUnit] = field(default_factory=list)
    cg_height: float = 0.13               # m above the floor
    cg_x: float = 0.0                     # m forward of the geometric centre
    cg_y: float = 0.0                     # m left of the geometric centre
    inertia_z: Optional[float] = None     # kg*m^2 about the vertical axis at CG
    # kg*m^2 about the level axes through the CG, side to side (pitch) and
    # front to back (roll); None is a solid box, see `level_inertia`
    pitch_inertia: Optional[float] = None
    roll_inertia: Optional[float] = None
    length: float = 15.0 * INCH          # frame, front to back
    width: float = 12.5 * INCH           # frame, side to side
    # loss terms that are not the tires
    bearing_drag: float = 0.9             # N*s/m, linear resistance of the base
    yaw_drag: float = 0.06                # N*m*s/rad, chain and bearing yaw drag
    aero_drag: float = 0.02               # N/(m/s)^2, basically nothing at VEX speeds
    surface_mu: float = 1.0               # scale for a dusty or waxed field
    gravity: float = G                    # m/s^2, set from SimConfig by Simulator
    name: str = "chassis"

    def __post_init__(self) -> None:
        if self.inertia_z is None:
            self.inertia_z = self.default_inertia()
        self._motor_cache: Optional[List[Motor]] = None

    def invalidate_motor_cache(self) -> None:
        """Call after adding or removing wheel units or their motors."""
        self._motor_cache = None

    def default_inertia(self) -> float:
        """Uniform rectangular plate about its centre, plus a parallel-axis nudge.

        Real VEX robots carry a little more mass at the corners than a uniform
        plate does, so this is scaled up by 1.10.  A plate is 1.0.  It used to
        be 1.25, which is more than this machine earns: the motors, battery and
        gearboxes all sit in close to the middle, and only the frame rails and
        the standoffs are out at the corners.  The number sets how fast the
        robot gets to its turn rate, not what that rate is -- top yaw rate is
        741 deg/s either way, and 1.10 reaches 90% of it in 0.420 s where 1.25
        took 0.478 s.
        """
        base = self.mass * (self.length ** 2 + self.width ** 2) / 12.0
        return 1.10 * base

    # ------------------------------------------------------------- accessors
    @property
    def motors(self) -> List[Motor]:
        # Rebuilt on every access this used to be one of the hottest calls in
        # the sim: the bus solve walks it several times per physics slice, at
        # 2 kHz. The layout does not change while the robot drives; call
        # invalidate_motor_cache() if it ever does.
        out = self._motor_cache
        if out is None:
            out = []
            for u in self.wheel_units:
                out.extend(u.motors)
            self._motor_cache = out
        return out

    @property
    def motor_count(self) -> int:
        return len(self.motors)

    def level_inertia(self, nx: float, ny: float) -> float:
        """Inertia about the level axis through the CG square to the body
        direction (nx, ny), which is the axis the robot rocks about when it
        tips that way.  Unless set, a solid box the frame's length and width
        and twice the CG height (see `dynamics.GUESSES`)."""
        height = 2.0 * self.cg_height
        pitch = (self.mass * (self.length ** 2 + height ** 2) / 12.0
                 if self.pitch_inertia is None else self.pitch_inertia)
        roll = (self.mass * (self.width ** 2 + height ** 2) / 12.0
                if self.roll_inertia is None else self.roll_inertia)
        return pitch * nx * nx + roll * ny * ny

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    def left_units(self) -> List[WheelUnit]:
        return [u for u in self.wheel_units if u.y > 1e-6]

    def right_units(self) -> List[WheelUnit]:
        return [u for u in self.wheel_units if u.y < -1e-6]

    def driven_units(self) -> List[WheelUnit]:
        return [u for u in self.wheel_units if u.driven]

    @property
    def track_width(self) -> float:
        ys = [u.y for u in self.wheel_units if u.driven]
        return (max(ys) - min(ys)) if ys else self.width

    @property
    def wheel_base(self) -> float:
        xs = [u.x for u in self.wheel_units if u.driven]
        return (max(xs) - min(xs)) if xs else self.length

    # --------------------------------------------------------- quick numbers
    def theoretical_free_speed(self) -> float:
        """Straight-line speed at 12 V with no load, m/s."""
        us = self.driven_units()
        if not us:
            return 0.0
        return sum(u.free_speed * u.radius for u in us) / len(us)

    def max_tractive_force(self) -> float:
        """Whichever is smaller: what the motors can push, or what the tires hold."""
        motor_limited = sum(u.max_torque / u.radius for u in self.driven_units())
        tire_limited = sum(
            u.wheel.tire.mu_long_static * self.surface_mu for u in self.driven_units()
        ) / max(len(self.driven_units()), 1) * self.mass * self.gravity
        return min(motor_limited, tire_limited)

    def max_acceleration(self) -> float:
        return self.max_tractive_force() / self.mass

    def is_traction_limited(self) -> bool:
        motor_limited = sum(u.max_torque / u.radius for u in self.driven_units())
        return motor_limited > self.max_tractive_force() + 1e-9

    def summary(self) -> dict:
        return {
            "name": self.name,
            "mass_kg": self.mass,
            "weight_N": self.weight,
            "motors": self.motor_count,
            "wheels": len(self.wheel_units),
            "track_width_m": self.track_width,
            "wheel_base_m": self.wheel_base,
            "inertia_z_kgm2": self.inertia_z,
            "free_speed_mps": self.theoretical_free_speed(),
            "max_tractive_force_N": self.max_tractive_force(),
            "max_accel_mps2": self.max_acceleration(),
            "traction_limited": self.is_traction_limited(),
            "peak_electrical_W": sum(12.0 * m.current_limit for m in self.motors),
        }


# ------------------------------------------------------------------ builders

def differential_drive(
    mass_lb: float = 15.0,
    wheel: str = "omni_3.25",
    cartridge: str = "green",
    motors_per_side: int = 3,
    external_ratio: float = 1.0,
    track_width_in: float = 12.0,
    wheel_base_in: float = 10.0,
    wheels_per_side: int = 3,
    center_traction: bool = False,
    cg_height_in: float = 4.0,
    cg_x_in: float = 0.0,
    length_in: float = 15.0,
    width_in: float = 12.5,
    name: str = "diff drive",
    motor_spec=None,
) -> Chassis:
    """Build a standard VEX tank drive.

    `external_ratio` is motor-cartridge turns per wheel turn: 1.0 for direct
    drive, 60/36 = 1.667 for a 36-tooth driving a 60-tooth (slower, more
    torque), 36/60 = 0.6 the other way.

    Motors are not flagged reversed here.  On a real robot the right side is
    mounted mirrored and the motors are reversed in software to cancel that
    out; in the sim every wheel already shares the chassis frame, so positive
    voltage means forward on both sides.  The `reversed` flag on `Motor` is
    left free for what it is actually useful for here: modelling a motor that
    somebody wired backwards.

    `center_traction` swaps the middle wheel of a 6-wheel drive for a traction
    wheel, which is the usual trick for keeping a bot from being shoved
    sideways while still turning easily.
    """
    spec_factory = motor_spec or v5_11w
    track = track_width_in * INCH
    base = wheel_base_in * INCH

    if wheels_per_side == 1:
        xs = [0.0]
    else:
        xs = [(-0.5 + i / (wheels_per_side - 1)) * base for i in range(wheels_per_side)]

    units: List[WheelUnit] = []
    for side, sy in (("L", +1.0), ("R", -1.0)):
        # spread the motors over the wheels, front first
        counts = [0] * wheels_per_side
        for i in range(motors_per_side):
            counts[i % wheels_per_side] += 1
        for i, x in enumerate(xs):
            is_center = center_traction and wheels_per_side >= 3 and i == wheels_per_side // 2
            ws = wheel_presets.traction(
                float(wheel.split("_")[1])) if is_center else wheel_presets.get(wheel)
            ms = [
                Motor(spec=spec_factory(), cartridge=cartridge,
                      name=f"{side}{i+1}-{j+1}")
                for j in range(counts[i])
            ]
            units.append(WheelUnit(
                wheel=ws, x=x, y=sy * track / 2.0, motors=ms,
                gear_ratio=external_ratio, name=f"{side}{i+1}",
            ))

    return Chassis(
        mass=mass_lb * LB,
        wheel_units=units,
        cg_height=cg_height_in * INCH,
        cg_x=cg_x_in * INCH,
        length=length_in * INCH,
        width=width_in * INCH,
        name=name,
    )


def add_tracking_wheels(chassis: Chassis, forward_offset_in: float = -4.0,
                        lateral_offset_in: float = 0.0,
                        diameter_in: float = 2.75,
                        load_n: float = 4.0) -> Chassis:
    """Bolt on spring-loaded odometry wheels: one parallel, one perpendicular.

    They are modelled as lightly loaded undriven wheels, so they pick up real
    slip when the robot is shoved but essentially none while driving normally.
    """
    par = WheelUnit(
        wheel=wheel_presets.tracking(diameter_in),
        x=forward_offset_in * INCH, y=lateral_offset_in * INCH,
        gear_ratio=1.0, name="track_parallel",
    )
    perp = WheelUnit(
        wheel=wheel_presets.tracking(diameter_in),
        x=forward_offset_in * INCH, y=0.0,
        gear_ratio=1.0, name="track_perp", steer_angle=math.pi / 2,
    )
    par.preload = load_n
    perp.preload = load_n
    chassis.wheel_units.extend([par, perp])
    return chassis


def add_sideways_tracking_wheel(chassis: Chassis, forward_in: float = 0.0, left_in: float = 0.0,
                                diameter_in: float = 2.0, load_n: float = 4.0) -> Chassis:
    """Bolt on one spring-loaded odometry wheel that rolls sideways, measuring how far the robot slides left."""
    wheel = WheelUnit(
        wheel=wheel_presets.tracking(diameter_in),
        x=forward_in * INCH, y=left_in * INCH,
        gear_ratio=1.0, name="track_perp", steer_angle=math.pi / 2,
    )
    wheel.preload = load_n
    chassis.wheel_units.append(wheel)
    return chassis

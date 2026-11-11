"""Wheel and tire model.

Each wheel gets a slip-based friction model rather than a "grip until you
exceed mu*N" switch, because the interesting behaviour of a VEX drivetrain --
wheelspin off the line, the way an omni chassis pushes sideways in a turn, how
much a 4-motor bot can shove -- all lives in the slip curve.

Longitudinal slip ratio and lateral slip angle:

    kappa   = (w * r_eff - vx) / max(|vx|, v_floor)
    tan(a)  =  vy / max(|vx|, v_floor)

Both are normalised by the slip at which the tire peaks, combined into one
slip magnitude, and run through a curve that rises to the static coefficient
and then decays to the sliding coefficient.  Longitudinal and lateral grip get
separate coefficients, which is the whole point: an omni wheel has plenty of
forward grip and almost none sideways, and that single asymmetry is what makes
an omni chassis turn like an omni chassis.

Slip does not respond instantly.  Each wheel carries its slip state through a
relaxation length, which is both physically right and keeps the equations from
going stiff at low speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import foam
from .units import INCH


@dataclass
class TireModel:
    """Friction coefficients and slip curve shape for one kind of wheel."""

    mu_long_static: float = 1.05     # peak forward/back coefficient
    mu_long_slide: float = 0.85      # coefficient once it is fully sliding
    mu_lat_static: float = 1.00
    mu_lat_slide: float = 0.80
    slip_ratio_peak: float = 0.09    # kappa at peak grip
    slip_angle_peak: float = 0.16    # rad, slip angle at peak grip
    relaxation_length: float = 0.030 # m, how far it rolls to build slip
    slide_decay: float = 1.2         # how fast grip falls past the peak
    contact_width: float = 1.0 * INCH  # m, width of wheel pressing into the tile
    axis_coupling: float = 1.0       # how much the two axes share one patch

    def rolling_resistance(self, load: float, radius: float) -> float:
        """Resisting force over load for a wheel of this radius rolling on foam."""
        return foam.rolling_resistance(load, radius, self.contact_width)

    def scrub_arm(self, load: float, radius: float) -> float:
        """Twisting torque per newton of load from the contact patch, in metres."""
        return foam.scrub_arm(load, radius, self.contact_width, self.mu_long_slide, self.mu_lat_slide)

    def curve(self, s: float) -> float:
        """Normalised grip as a function of normalised slip magnitude.

        Rises to 1.0 at s = 1, then decays toward the sliding ratio.
        """
        if s <= 0.0:
            return 0.0
        if s < 1.0:
            return s * (2.0 - s)
        return math.exp(-(s - 1.0) / self.slide_decay)

    def blend(self, s: float, static: float, slide: float) -> float:
        """Coefficient at normalised slip s, blending static toward sliding."""
        if s <= 1.0:
            return static
        f = math.exp(-(s - 1.0) / self.slide_decay)
        return slide + (static - slide) * f


@dataclass
class WheelSpec:
    """A physical wheel: size, mass, and what it is made of."""

    name: str = "4in omni"
    diameter: float = 4.0 * INCH
    mass: float = 0.115              # kg
    inertia: float = None            # kg*m^2 about its axle; None -> derive
    tire: TireModel = None
    roller_angle: float = math.pi / 2  # 0 = traction, pi/2 = omni, pi/4 = mecanum
    driven: bool = True

    def __post_init__(self) -> None:
        if self.tire is None:
            self.tire = TireModel()
        if self.inertia is None:
            # thin-rim approximation: most of an omni's mass is at the rollers
            self.inertia = 0.75 * self.mass * self.radius ** 2

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def circumference(self) -> float:
        return math.pi * self.diameter


# ---------------------------------------------------------------- presets ---
# Grip on the grey EVA foam field tiles; `foam` has the sources.

SLIDING_SHARE = 0.81
OMNI_GRIP = {2.0: foam.OMNI_GRIP_MEASURED[2.75], 2.75: foam.OMNI_GRIP_MEASURED[2.75], 3.25: 0.87,
             4.0: foam.OMNI_GRIP_MEASURED[4.0]}


def _omni_tire(diameter_in: float) -> TireModel:
    grip = OMNI_GRIP[diameter_in]
    return TireModel(
        mu_long_static=grip, mu_long_slide=grip * SLIDING_SHARE,
        mu_lat_static=0.26, mu_lat_slide=0.22,
        slip_ratio_peak=0.09, slip_angle_peak=0.22,
        contact_width=0.4 * INCH,
        axis_coupling=0.12,              # rollers: sideways is a separate contact
    )


def _traction_tire() -> TireModel:
    grip = foam.TRACTION_GRIP
    return TireModel(
        mu_long_static=grip, mu_long_slide=grip * SLIDING_SHARE,
        mu_lat_static=grip, mu_lat_slide=grip * SLIDING_SHARE,
        slip_ratio_peak=0.08, slip_angle_peak=0.14,
        contact_width=1.0 * INCH,
    )


def _flex_tire(shore: str = "45A") -> TireModel:
    soft = {"30A": 1.75, "35A": 1.65, "45A": 1.50, "60A": 1.32}[shore]
    return TireModel(
        mu_long_static=soft, mu_long_slide=soft * 0.82,
        mu_lat_static=soft * 0.9, mu_lat_slide=soft * 0.74,
        slip_ratio_peak=0.11, slip_angle_peak=0.18,
        relaxation_length=0.045, contact_width=1.0 * INCH,
    )


def _mecanum_tire() -> TireModel:
    grip = foam.MECANUM_GRIP
    return TireModel(
        mu_long_static=grip, mu_long_slide=grip * SLIDING_SHARE,
        mu_lat_static=0.30, mu_lat_slide=0.25,
        slip_ratio_peak=0.10, slip_angle_peak=0.22,
        contact_width=0.5 * INCH,
        axis_coupling=0.20,
    )


def omni(diameter_in: float = 4.0) -> WheelSpec:
    mass = {2.75: 0.055, 3.25: 0.075, 4.0: 0.115}.get(diameter_in, 0.028 * diameter_in)
    return WheelSpec(f"{diameter_in}in omni", diameter_in * INCH, mass,
                     tire=_omni_tire(diameter_in), roller_angle=math.pi / 2)


def traction(diameter_in: float = 4.0) -> WheelSpec:
    mass = {2.75: 0.062, 3.25: 0.085, 4.0: 0.130}.get(diameter_in, 0.032 * diameter_in)
    return WheelSpec(f"{diameter_in}in traction", diameter_in * INCH, mass,
                     tire=_traction_tire(), roller_angle=0.0)


def flex(diameter_in: float = 3.25, shore: str = "45A") -> WheelSpec:
    return WheelSpec(f"{diameter_in}in flex {shore}", diameter_in * INCH,
                     0.020 * diameter_in, tire=_flex_tire(shore), roller_angle=0.0)


def mecanum(diameter_in: float = 4.0, roller_sign: float = 1.0) -> WheelSpec:
    """A mecanum wheel; use opposite ``roller_sign`` values for X rollers."""
    return WheelSpec(f"{diameter_in}in mecanum", diameter_in * INCH, 0.165,
                     tire=_mecanum_tire(),
                     roller_angle=math.copysign(math.pi / 4, roller_sign))


def tracking(diameter_in: float = 2.75) -> WheelSpec:
    """An undriven odometry wheel: light, free, and barely loaded."""
    return WheelSpec(f"{diameter_in}in tracking", diameter_in * INCH, 0.030,
                     tire=_omni_tire(diameter_in), roller_angle=math.pi / 2, driven=False)


PRESETS = {
    "omni_2.75": lambda: omni(2.75),
    "omni_3.25": lambda: omni(3.25),
    "omni_4": lambda: omni(4.0),
    "traction_2.75": lambda: traction(2.75),
    "traction_3.25": lambda: traction(3.25),
    "traction_4": lambda: traction(4.0),
    "flex_3.25": lambda: flex(3.25),
    "flex_4": lambda: flex(4.0),
    "mecanum_4": lambda: mecanum(4.0),
}


def get(name: str) -> WheelSpec:
    if name not in PRESETS:
        raise KeyError(f"unknown wheel '{name}'; have {sorted(PRESETS)}")
    return PRESETS[name]()


def tire_forces(tire: TireModel, fz: float, kappa: float, tan_alpha: float,
                surface_mu: float = 1.0):
    """Contact-patch force from lagged slip state.  Returns (Fx, Fy) in N.

    Fx is along the wheel's rolling direction, Fy is to its left.  Positive
    kappa (wheel spinning faster than the ground) pushes the robot forward;
    positive lateral slip is resisted, so Fy opposes it.

    How much the two directions fight each other is `axis_coupling`.  A traction
    wheel is 1.0: one rubber patch, one friction budget, the classic ellipse --
    spin it sideways and it has less left for driving.  An omni is nearly 0:
    going sideways happens on the rollers, which is a separate rolling contact,
    so sliding sideways costs it almost none of its forward grip.  That is the
    whole reason an omni base can turn and drive at the same time.
    """
    if fz <= 0.0:
        return 0.0, 0.0

    sx = kappa / tire.slip_ratio_peak
    sy = tan_alpha / tire.slip_angle_peak
    c = tire.axis_coupling
    sx_eff = math.hypot(sx, c * sy)
    sy_eff = math.hypot(c * sx, sy)

    fx = fy = 0.0
    if sx_eff > 1e-9:
        shape = tire.curve(sx_eff) if sx_eff < 1.0 else 1.0
        mu_x = tire.blend(sx_eff, tire.mu_long_static, tire.mu_long_slide) * surface_mu
        fx = shape * mu_x * fz * (sx / sx_eff)
    if sy_eff > 1e-9:
        shape = tire.curve(sy_eff) if sy_eff < 1.0 else 1.0
        mu_y = tire.blend(sy_eff, tire.mu_lat_static, tire.mu_lat_slide) * surface_mu
        fy = -shape * mu_y * fz * (sy / sy_eff)
    return fx, fy

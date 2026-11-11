"""Unit conversions.

Everything inside the simulator is SI: metres, kilograms, seconds, newtons,
radians, volts, amps, watts, kelvin (temperatures are carried in celsius
because that is what the V5 brain reports).

VEX teams think in inches, pounds and RPM, so these helpers exist to keep the
conversions in one place instead of scattering magic numbers through the model.
"""

# Length
INCH = 0.0254           # m per inch
FOOT = 0.3048
MM = 1e-3
TILE = 0.6096           # one VRC field tile, 24 in

# Mass / force
LB = 0.45359237         # kg per pound-mass
OZ = LB / 16.0
G = 9.80665             # m/s^2

# Torque
IN_LB = 0.1129848290276167   # N*m per inch-pound
IN_OZ = IN_LB / 16.0

# Rotation
import math

RPM = 2.0 * math.pi / 60.0   # rad/s per rpm
DEG = math.pi / 180.0


def inches(x: float) -> float:
    return x * INCH


def feet(x: float) -> float:
    return x * FOOT


def tiles(x: float) -> float:
    return x * TILE


def lbs(x: float) -> float:
    return x * LB


def ounces(x: float) -> float:
    return x * OZ


def rpm(x: float) -> float:
    return x * RPM


def deg(x: float) -> float:
    return x * DEG


def to_inches(x: float) -> float:
    return x / INCH


def to_feet(x: float) -> float:
    return x / FOOT


def to_lbs(x: float) -> float:
    return x / LB


def to_rpm(x: float) -> float:
    return x / RPM


def to_deg(x: float) -> float:
    return x / DEG


def to_in_lb(x: float) -> float:
    return x / IN_LB


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def sign(x: float) -> float:
    return 0.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0)

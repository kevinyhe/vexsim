"""Sensors, with the errors that make autonomous routines drift.

A routine that works against perfect sensors is not a routine that works.  The
things that actually bite -- encoder quantisation, the 10 ms update period on
the smart port, gyro bias that walks over a match, a tracking wheel that skips
when the robot gets shoved -- are all modelled here.

Randomness comes from a seeded generator on each sensor, so a run is
repeatable, and re-seeding lets you do Monte Carlo over an auton.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .units import DEG, G, INCH, clamp, wrap_pi


_SAMPLE_TIME_EPSILON = 1e-12


@dataclass
class Encoder:
    """Motor built-in encoder, reported on the output shaft.

    V5 reports position in degrees; resolution depends on the cartridge because
    the encoder is on the rotor: 1800 / 900 / 300 ticks per output revolution
    for red / green / blue.
    """

    ticks_per_rev: float = 300.0
    update_period: float = 0.010     # s, smart port refresh
    noise_deg: float = 0.0
    seed: int = 0

    position: float = 0.0            # rad, quantised and delayed
    velocity: float = 0.0            # rad/s as the brain reports it
    raw_position: float = 0.0
    _last_update: float = -1.0
    _last_pos: float = 0.0
    _rng: random.Random = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def update(self, true_position: float, t: float) -> None:
        self.raw_position = true_position
        if (self._last_update >= 0.0 and
                t - self._last_update + _SAMPLE_TIME_EPSILON < self.update_period):
            return
        dt = t - self._last_update if self._last_update >= 0.0 else self.update_period
        p = true_position
        if self.noise_deg:
            p += self._rng.gauss(0.0, self.noise_deg * DEG)
        step = 2.0 * math.pi / self.ticks_per_rev
        q = round(p / step) * step
        self.velocity = (q - self._last_pos) / max(dt, 1e-9)
        self._last_pos = q
        self.position = q
        self._last_update = t

    def reset(self) -> None:
        self.position = self.raw_position = self._last_pos = 0.0
        self.velocity = 0.0
        self._last_update = -1.0
        self._rng.seed(self.seed)


@dataclass
class RotationSensor(Encoder):
    """V5 rotation sensor: 0.088 degree resolution, same 10 ms port period."""

    ticks_per_rev: float = 4096.0
    noise_deg: float = 0.02


@dataclass
class IMU:
    """V5 inertial sensor: heading from an integrated rate gyro.

    Three separate errors, all of which show up in real autons:
      * `bias` -- a constant rate offset left over from calibration
      * `bias_walk` -- that offset wandering over the match
      * `scale_error` -- turns that come up consistently short or long

    Every supplied rate sample advances the internal gyro integral; the smart
    port only publishes a held heading/rate/acceleration at `update_period`.
    Bias-walk and rate-noise draws keep that publication cadence and are held
    between reports, so changing the integration tick does not retune noise.
    """

    bias: float = 0.02 * DEG          # rad/s
    bias_walk: float = 0.004 * DEG    # rad/s per sqrt(s)
    noise: float = 0.05 * DEG         # rad/s
    scale_error: float = 0.002        # fraction, +0.2% reads 0.2% long
    update_period: float = 0.010
    accel_noise: float = 0.03         # m/s^2
    seed: int = 1

    heading: float = 0.0              # rad, what the brain reports
    rate: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    valid: bool = True
    _bias: float = 0.0
    _last_update: float = -1.0
    _rng: random.Random = None
    _last_sample: float = field(default=-1.0, init=False, repr=False)
    _previous_rate: float = field(default=0.0, init=False, repr=False)
    _integrated_heading: float = field(default=0.0, init=False, repr=False)
    _held_noise: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._bias = self.bias
        self._integrated_heading = self.heading

    def update(self, true_rate: float, ax: float, ay: float, t: float) -> None:
        if self._last_sample >= 0.0 and t < self._last_sample:
            return
        measured = true_rate * (1.0 + self.scale_error) + self._bias + self._held_noise
        current_rate = clamp(measured, -1000 * DEG, 1000 * DEG)
        if self._last_sample >= 0.0:
            elapsed = t - self._last_sample
            self._integrated_heading = wrap_pi(
                self._integrated_heading + (self._previous_rate + current_rate) * elapsed / 2)
        # The first observation only establishes the integration time origin.
        self._last_sample = t
        self._previous_rate = current_rate
        if (self._last_update >= 0.0 and
                t - self._last_update + _SAMPLE_TIME_EPSILON < self.update_period):
            return
        dt = t - self._last_update if self._last_update >= 0.0 else 0.0
        self._last_update = t

        # The previous interval used its held error. Draw the error for the
        # next interval now, rather than applying a new draw retroactively.
        self._bias += self._rng.gauss(0.0, self.bias_walk * math.sqrt(dt))
        self._held_noise = self._rng.gauss(0.0, self.noise)
        measured = true_rate * (1.0 + self.scale_error) + self._bias + self._held_noise
        measured_ax = ax + self._rng.gauss(0.0, self.accel_noise)
        measured_ay = ay + self._rng.gauss(0.0, self.accel_noise)
        # Published limits: +/-1000 deg/s and +/-4g. Saturating and marking
        # overload invalid is a simulator policy, not measured VEX firmware.
        self.valid = (abs(measured) <= 1000 * DEG and
                      max(abs(measured_ax), abs(measured_ay)) <= 4 * G)
        self.rate = clamp(measured, -1000 * DEG, 1000 * DEG)
        self._previous_rate = self.rate
        self.heading = self._integrated_heading
        self.accel_x = clamp(measured_ax, -4 * G, 4 * G)
        self.accel_y = clamp(measured_ay, -4 * G, 4 * G)

    def reset(self, heading: float = 0.0) -> None:
        self.heading = heading
        self._integrated_heading = heading
        self.rate = 0.0
        self._last_update = -1.0
        self._last_sample = -1.0
        self._previous_rate = self._held_noise = 0.0
        self._bias = self.bias
        self.accel_x = self.accel_y = 0.0
        self.valid = True
        self._rng.seed(self.seed)


@dataclass
class GPSSensor:
    """VEX GPS sensor: absolute pose from the field strips, when it can see them."""

    position_noise: float = 0.012     # m, 1 sigma
    heading_noise: float = 1.0 * DEG
    update_period: float = 0.020
    dropout_rate: float = 0.05        # fraction of reads that come back stale
    seed: int = 2

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    valid: bool = True
    _last_update: float = -1.0
    _rng: random.Random = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self.x = self.y = self.heading = 0.0
        self.valid = False
        self._last_update = -1.0
        self._rng.seed(self.seed)

    def update(self, x: float, y: float, theta: float, t: float) -> None:
        if (self._last_update >= 0.0 and
                t - self._last_update + _SAMPLE_TIME_EPSILON < self.update_period):
            return
        self._last_update = t
        if self._rng.random() < self.dropout_rate:
            self.valid = False
            return
        self.valid = True
        self.x = x + self._rng.gauss(0.0, self.position_noise)
        self.y = y + self._rng.gauss(0.0, self.position_noise)
        self.heading = wrap_pi(theta + self._rng.gauss(0.0, self.heading_noise))


@dataclass
class Odometry:
    """Dead reckoning from wheel travel and a heading source.

    Two flavours, both common in VRC:
      * tracking wheels -- feed `parallel` and `perpendicular` travel
      * drive encoders  -- feed left and right travel, no perpendicular term

    The heading comes from the IMU when one is supplied, otherwise from the
    difference between the two drive sides, which is where track-width error
    turns into heading error.
    """

    track_width: float = 0.3
    use_imu: bool = True
    # where the tracking wheels sit relative to the tracking centre, metres.
    # A perpendicular wheel mounted ahead of or behind centre rolls during a
    # point turn even though the robot has not moved, and a parallel wheel
    # mounted off to one side does the same; both are corrected out here.
    perp_offset: float = 0.0        # forward of centre
    parallel_offset: float = 0.0    # left of centre

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    _last_left: float = 0.0
    _last_right: float = 0.0
    _last_perp: float = 0.0
    _last_theta: float = 0.0
    _init: bool = False

    def update(self, left: float, right: float, heading: Optional[float] = None,
               perpendicular: Optional[float] = None) -> None:
        if not self._init:
            self._last_left, self._last_right = left, right
            self._last_perp = perpendicular or 0.0
            self._last_theta = heading if heading is not None else self.theta
            self._init = True
            return

        dl = left - self._last_left
        dr = right - self._last_right
        dp = (perpendicular - self._last_perp) if perpendicular is not None else 0.0
        self._last_left, self._last_right = left, right
        if perpendicular is not None:
            self._last_perp = perpendicular

        if heading is not None and self.use_imu:
            dtheta = wrap_pi(heading - self._last_theta)
            self._last_theta = heading
        else:
            dtheta = (dr - dl) / self.track_width

        # take the turn back out of the offset wheels
        dp -= self.perp_offset * dtheta
        dfwd = (dl + dr) / 2.0 + self.parallel_offset * dtheta

        # arc integration: over a small step the robot follows a circular arc,
        # which matters at high yaw rates where the straight-line version drifts
        if abs(dtheta) > 1e-6:
            radius_f = dfwd / dtheta
            radius_s = dp / dtheta
            sin_d, cos_d = math.sin(dtheta), math.cos(dtheta)
            local_x = radius_f * sin_d + radius_s * (cos_d - 1.0)
            local_y = radius_f * (1.0 - cos_d) + radius_s * sin_d
        else:
            local_x, local_y = dfwd, dp

        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += local_x * c - local_y * s
        self.y += local_x * s + local_y * c
        self.theta = wrap_pi(self.theta + dtheta)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x, self.y, self.theta = x, y, theta
        self._init = False


@dataclass
class DistanceSensor:
    """V5 distance sensor against an axis-aligned field wall box."""

    field_half: float = 140.5 * INCH / 2.0  # m, V5RC inside perimeter
    max_range: float = 2.0
    noise: float = 0.005
    update_period: float = 0.020
    seed: int = 3
    min_range: float = 0.020

    distance: float = 0.0
    valid: bool = False
    _last_update: float = -1.0
    _rng: random.Random = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self.distance = 0.0
        self.valid = False
        self._last_update = -1.0
        self._rng.seed(self.seed)

    def update(self, x: float, y: float, theta: float, t: float) -> None:
        if (self._last_update >= 0.0 and
                t - self._last_update + _SAMPLE_TIME_EPSILON < self.update_period):
            return
        self._last_update = t
        h = self.field_half
        c, s = math.cos(theta), math.sin(theta)
        best = float("inf")
        for wall, comp, pos, cross_pos, cross_comp in (
                (h, c, x, y, s), (-h, c, x, y, s),
                (h, s, y, x, c), (-h, s, y, x, c)):
            if abs(comp) > 1e-9:
                d = (wall - pos) / comp
                if 0.0 <= d < best and abs(cross_pos + d * cross_comp) <= h + 1e-12:
                    best = d
        if best < self.min_range or best > self.max_range:
            self.valid = False
            self.distance = self.max_range
        else:
            self.valid = True
            self.distance = clamp(best + self._rng.gauss(0.0, self.noise),
                                  self.min_range, self.max_range)

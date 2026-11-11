"""Brushed DC motor model with gearbox, current limit and thermal derating.

The electrical model is the textbook one:

    I = (V_applied - Ke * w_rotor) / R
    torque_rotor = Kt * I

Everything else is the stuff that makes a V5 motor behave like a V5 motor
rather than like an ideal DC motor:

  * a 2.5 A per-motor current limit, which is what actually sets stall torque
  * a cartridge reduction (36:1 / 18:1 / 6:1) with a real efficiency
  * coulomb + viscous drag that eats the no-load current
  * a first-order thermal model and the firmware's step derating table
  * the brain's control modes: voltage, internal velocity PI, position PID,
    and the three stopping behaviours (coast / brake / hold)

The V5 models are approximate electrical equivalents constrained by published
output torque, speed and power, not measured winding identifications. A speed
governor preserves the documented loaded-speed plateau; a conservative power
envelope uses the continuous rating. R, inertia, inductance, gearbox efficiency
and thermal RC values remain assumptions requiring hardware calibration.

Manufacturer references:
https://kb.vex.com/hc/en-us/articles/360044325872
https://kb.vex.com/hc/en-us/articles/10002101702932
https://projectpartners.pltw.org/Training3/story_content/external_files/VEX_2_Wire_Motor_393.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .units import clamp


class MotorMode(Enum):
    VOLTAGE = "voltage"     # direct voltage command, what most autons use
    VELOCITY = "velocity"   # brain-side velocity PI + feedforward
    POSITION = "position"   # brain-side position PID
    COAST = "coast"         # leads open, motor freewheels
    BRAKE = "brake"         # leads shorted, back-EMF braking
    HOLD = "hold"           # position PID onto the position where it stopped


class BrakeMode(Enum):
    COAST = "coast"
    BRAKE = "brake"
    HOLD = "hold"


STOP_MODES = {BrakeMode.COAST: MotorMode.COAST, BrakeMode.BRAKE: MotorMode.BRAKE, BrakeMode.HOLD: MotorMode.HOLD}


@dataclass
class MotorSpec:
    """Physical constants of the motor itself, before any cartridge."""

    name: str = "V5 11W"
    kt: float = 0.02612915266890631 # equivalent: 2.1 Nm at 2.5 A, red cartridge
    ke: float = 0.02612915266890631 # SI reciprocity: Kt == Ke
    resistance: float = 2.2         # assumed equivalent winding + brushes, ohms
    inductance: float = 1.5e-3      # H, only used if electrical dynamics are on
    no_load_current: float = 0.15   # A at 12 V, free spinning
    rotor_inertia: float = 2.0e-6   # kg*m^2 at the rotor
    viscous_damping: float = 2.0e-7 # N*m per rad/s at the rotor
    nominal_voltage: float = 12.0
    current_limit: float = 2.5      # A, firmware default
    gearbox_efficiency: float = 0.95 # assumed; already included in equivalent fit
    # thermal
    thermal_resistance: float = 3.0   # K/W, winding to ambient
    thermal_capacitance: float = 25.0 # J/K
    max_temperature: float = 70.0     # C, documented V5 shutdown
    peak_power_rating: float = 11.0   # W, conservative continuous output envelope
    speed_limit: float = 120.0 * math.pi  # rotor rad/s: 100/200/600 rpm output
    speed_droop: float = 0.05         # approximate governor droop, not firmware gain
    limit_power: bool = True
    thermal_derating: bool = True
    thermal_policy: str = "documented V5 11W current derating"
    fixed_ratio: float = 0.0         # nonzero for motors without cartridge options

    @property
    def free_speed(self) -> float:
        """Rotor speed at nominal voltage, no load, rad/s."""
        natural = (self.nominal_voltage - self.no_load_current * self.resistance) / self.ke
        return min(natural, self.speed_limit) if self.speed_limit > 0.0 else natural

    @property
    def stall_current_unlimited(self) -> float:
        return self.nominal_voltage / self.resistance

    @property
    def stall_torque_rotor(self) -> float:
        """Rotor stall torque with the current limit in force."""
        return self.kt * min(self.current_limit, self.stall_current_unlimited)

    @property
    def peak_power(self) -> float:
        """Approximate peak before gearing, including no-load drag and power cap."""
        available = max(0.0, self.nominal_voltage - self.no_load_current * self.resistance)
        peak = self.kt / self.ke * available ** 2 / (4.0 * self.resistance)
        if self.limit_power:
            peak = min(peak, self.peak_power_rating / self.gearbox_efficiency)
        return peak


def v5_11w() -> MotorSpec:
    """The 11 W V5 Smart Motor (276-4840)."""
    return MotorSpec()


def v5_5_5w() -> MotorSpec:
    """The 5.5 W motor's fixed 200 rpm equivalent.

    Its exact temperature-current table is not established by the referenced
    5.5 W datasheet; reusing the 11 W policy is an explicit model assumption.
    """
    return MotorSpec(
        name="V5 5.5W",
        kt=0.014061849639454177,  # 0.5 Nm with the equivalent fixed 18:1 gearing
        ke=0.014061849639454177,
        resistance=1.5,
        nominal_voltage=7.2,     # approximate internal PWM voltage, not V5 bus
        no_load_current=0.12,
        rotor_inertia=1.1e-6,
        current_limit=2.5,
        gearbox_efficiency=0.83,
        thermal_resistance=4.2,
        thermal_capacitance=14.0,
        peak_power_rating=5.5,
        fixed_ratio=18.0,
        thermal_policy="assumed V5 11W-style steps; 5.5 W thresholds uncalibrated",
    )


def cortex_393() -> MotorSpec:
    """Legacy 2-wire 393 motor, for comparing old and new drivetrains.

    The manufacturer specifies 7.2 V, 100 rpm, 1.67 Nm, 4.8 A stall and
    0.37 A free current. R = 7.2/4.8; Ke follows from the free-speed point;
    an equivalent efficiency matches stall torque. These are a lumped fit,
    not a measurement of the internal gear train. The Cortex motor has no
    V5 firmware speed/power governor or stepwise temperature derating.
    """
    return MotorSpec(
        name="393",
        kt=0.01618751933947415,
        ke=0.01618751933947415,
        resistance=1.5,
        nominal_voltage=7.2,
        no_load_current=0.37,
        rotor_inertia=2.4e-6,
        current_limit=4.8,
        gearbox_efficiency=0.5940819776570643,
        viscous_damping=0.0,  # measured free current represented by Coulomb drag
        thermal_resistance=5.0,
        thermal_capacitance=18.0,
        max_temperature=90.0,
        peak_power_rating=4.4,
        speed_limit=0.0,
        limit_power=False,
        thermal_derating=False,
        thermal_policy="approximate thermal cutoff; no V5 firmware steps",
    )


# Cartridge name -> (reduction, nominal output rpm)
CARTRIDGES = {
    "red": (36.0, 100.0),
    "green": (18.0, 200.0),
    "blue": (6.0, 600.0),
    "direct": (1.0, 3600.0),
    # the 393's internal gearbox, standard and high-speed builds
    "393_torque": (39.2, 100.0),
    "393_speed": (24.5, 160.0),
}


@dataclass
class Motor:
    """One smart motor: spec + cartridge + firmware behaviour + live state."""

    spec: MotorSpec = field(default_factory=v5_11w)
    cartridge: str = "green"
    reversed: bool = False
    name: str = "motor"
    ambient_temperature: float = 22.0
    # firmware velocity/position gains, on the OUTPUT shaft
    vel_kp: float = 0.35            # V per (rad/s) of error
    vel_ki: float = 1.2
    vel_kff: float = 1.0            # fraction of the ideal feedforward voltage
    pos_kp: float = 6.0             # V per rad of error
    pos_kd: float = 0.35
    electrical_dynamics: bool = False  # include L di/dt (adds a stiff state)

    # --- commanded ---
    mode: MotorMode = MotorMode.COAST
    command: float = 0.0            # volts, rad/s, or rad depending on mode
    current_limit: float = None     # per-motor override, A
    # On a brain, a command waits in `sent` for the brain's next write to
    # the motor (`receive`); on its own a motor takes it at once.
    on_brain: bool = False
    sent: Optional[Tuple[MotorMode, float]] = None

    # --- state ---
    position: float = 0.0           # output shaft angle, rad
    velocity: float = 0.0           # output shaft speed, rad/s
    temperature: float = 22.0       # C
    current: float = 0.0            # A (positive = motoring forward)
    applied_voltage: float = 0.0    # V actually across the winding
    commanded_voltage: float = 0.0  # V the controller asked for
    torque: float = 0.0             # N*m at the OUTPUT shaft
    power_electrical: float = 0.0   # W drawn from the bus
    power_mechanical: float = 0.0   # W leaving the output shaft
    power_copper: float = 0.0       # W burned in the winding
    power_friction: float = 0.0     # W burned in gearbox + rotor drag
    efficiency: float = 0.0
    derate: float = 1.0             # firmware thermal derate factor, 0-1
    over_temp: bool = False
    current_limited: bool = False
    _vel_integral: float = 0.0
    _hold_target: float = 0.0
    _scale: float = 1.0             # brain-side scaling when the pack is maxed
    _controller_elapsed: float = 0.0
    regenerative_overcurrent: bool = False  # forced speed exceeds controllable range
    # exp() of the RL time constant, remembered because the bus solve calls
    # electrical() three or four times a step with the same dt
    _decay_key: tuple = None
    _decay: float = 0.0
    _ratio_key: tuple = None
    _ratio: float = 0.0
    _step_key: tuple = None
    _step_cache: tuple = None

    def __post_init__(self) -> None:
        self.temperature = self.ambient_temperature
        if self.current_limit is None:
            self.current_limit = self.spec.current_limit

    # ------------------------------------------------------------------ specs
    @property
    def ratio(self) -> float:
        """Rotor revolutions per output revolution."""
        # Read hundreds of thousands of times a second's worth of sim; the
        # dict lookup is memoised against the cartridge it came from.
        if self._ratio_key != (self.spec.fixed_ratio, self.cartridge):
            self._ratio_key = (self.spec.fixed_ratio, self.cartridge)
            self._ratio = self.spec.fixed_ratio or CARTRIDGES[self.cartridge][0]
        return self._ratio

    @property
    def free_speed(self) -> float:
        """Output shaft free speed at 12 V, rad/s."""
        return self.spec.free_speed / self.ratio

    @property
    def stall_torque(self) -> float:
        """Output stall torque with the current limit, N*m."""
        s = self.spec
        return (s.stall_torque_rotor * self.ratio * s.gearbox_efficiency
                - self.coulomb_torque)

    @property
    def coulomb_torque(self) -> float:
        """Constant drag at the output shaft, N*m."""
        s = self.spec
        return s.kt * s.no_load_current * self.ratio * s.gearbox_efficiency

    @property
    def output_inertia(self) -> float:
        """Rotor inertia reflected to the output shaft, kg*m^2."""
        return self.spec.rotor_inertia * self.ratio ** 2

    # --------------------------------------------------------------- commands
    def set_voltage(self, volts: float) -> None:
        self._send(MotorMode.VOLTAGE, volts)

    def set_percent(self, pct: float) -> None:
        """-100..100, the way `motor.spin(fwd, pct, percent)` behaves."""
        self.set_voltage(clamp(pct, -100.0, 100.0) / 100.0 * self.spec.nominal_voltage)

    def set_velocity(self, rad_s: float) -> None:
        self._send(MotorMode.VELOCITY, rad_s)

    def set_position(self, rad: float) -> None:
        self._send(MotorMode.POSITION, rad)

    def stop(self, brake_mode: BrakeMode = BrakeMode.COAST) -> None:
        self._send(STOP_MODES[brake_mode], 0.0)

    def receive(self) -> None:
        """The brain's write reaches the motor: it takes the latest command sent."""
        if self.sent is not None:
            self._take(*self.sent)
            self.sent = None

    def _send(self, mode: MotorMode, command: float) -> None:
        if self.on_brain:
            self.sent = (mode, command)
        else:
            self._take(mode, command)

    def _take(self, mode: MotorMode, command: float) -> None:
        self.mode = mode
        self.command = command
        if mode in STOP_MODES.values():
            if mode is MotorMode.HOLD:
                self._hold_target = self.position
            self._vel_integral = 0.0
            # Cancel the old drive at once, including HOLD between PID ticks.
            # The new hold target is sampled on the existing 10 ms firmware clock.
            self.commanded_voltage = 0.0

    # ------------------------------------------------------------- simulation
    def _thermal_derate(self) -> float:
        """V5 current limits: half at 55 C, quarter at 60 C, eighth at 65 C."""
        t = self.temperature
        if t >= self.spec.max_temperature:
            return 0.0
        if not self.spec.thermal_derating:
            return 1.0
        if t < 55.0:
            return 1.0
        if t < 60.0:
            return 0.5
        if t < 65.0:
            return 0.25
        if t < 70.0:
            return 0.125
        return 0.0

    def advance_control(self, dt: float) -> None:
        """Sample internal PID at 10 ms; direct voltage/stopping commands are immediate."""
        self._controller_elapsed += dt
        if self.mode is MotorMode.VOLTAGE:
            d = -1.0 if self.reversed else 1.0
            self.commanded_voltage = d * clamp(
                self.command, -self.spec.nominal_voltage, self.spec.nominal_voltage)
        elif self.mode in (MotorMode.COAST, MotorMode.BRAKE):
            self.commanded_voltage = 0.0
        while self._controller_elapsed >= 0.010 - 1e-12:
            if self.mode in (MotorMode.VELOCITY, MotorMode.POSITION, MotorMode.HOLD):
                self.control_voltage(0.010)
            self._controller_elapsed = max(0.0, self._controller_elapsed - 0.010)

    def control_voltage(self, dt: float) -> float:
        """Run the firmware controller and return the requested winding volts."""
        d = -1.0 if self.reversed else 1.0
        w = self.velocity * d
        p = self.position * d

        if self.mode is MotorMode.VOLTAGE:
            v = self.command
        elif self.mode is MotorMode.VELOCITY:
            target = self.command
            err = target - w
            ff = self.vel_kff * target * self.spec.ke * self.ratio
            self._vel_integral += err * dt
            # anti-windup: the integrator can never ask for more than the bus
            lim = self.spec.nominal_voltage / max(self.vel_ki, 1e-9)
            self._vel_integral = clamp(self._vel_integral, -lim, lim)
            v = ff + self.vel_kp * err + self.vel_ki * self._vel_integral
        elif self.mode in (MotorMode.POSITION, MotorMode.HOLD):
            target = self.command if self.mode is MotorMode.POSITION else self._hold_target * d
            err = target - p
            v = self.pos_kp * err - self.pos_kd * w
        else:  # COAST or BRAKE
            v = 0.0

        v = clamp(v, -self.spec.nominal_voltage, self.spec.nominal_voltage)
        self.commanded_voltage = v * d
        return self.commanded_voltage

    def electrical(self, bus_voltage: float, dt: float, *,
                   update_control: bool = True,
                   initial_current: float = None) -> float:
        """Solve the electrical side for this step.  Returns bus current draw.

        Called before the mechanical integration so the brain can look at the
        total pack draw and scale everybody back if it has to.  The optional
        arguments let the simulator retry its bus-voltage solve without
        advancing the controller or inductance state more than once.
        """
        s = self.spec
        if update_control:
            self.advance_control(dt)

        # The bus solve calls this three or four times a physics slice with the
        # same shaft speed, temperature and dt -- only bus_voltage and _scale
        # move between passes. Everything those two do not touch is worked out
        # once and reused, which is where most of the repeated cost was.
        i0 = self.current if initial_current is None else initial_current
        key = (dt, i0, self.velocity, self.temperature, self.mode,
               self.current_limit, self.cartridge, self.electrical_dynamics)
        if key == self._step_key:
            (derate, back_emf, decay, response, lim, rotor_speed,
             drive_limit) = self._step_cache
        else:
            derate = self._thermal_derate()
            ratio = self.ratio
            back_emf = s.ke * self.velocity * ratio
            if self.electrical_dynamics and s.inductance > 0.0:
                dkey = (dt, s.resistance, s.inductance)
                if dkey != self._decay_key:
                    self._decay_key = dkey
                    self._decay = math.exp(-s.resistance * dt / s.inductance)
                decay = self._decay
            else:
                decay = 0.0
            response = max(1.0 - decay, 1e-15)
            lim = max(0.0, self.current_limit * derate)
            rotor_speed = self.velocity * ratio
            drive_limit = None
            if rotor_speed != 0.0:
                # Torque after drag is eta*Kt*(I-I_idle). A small finite droop
                # avoids an infinitely stiff, discontinuous ideal speed clamp.
                idle = s.no_load_current + (s.viscous_damping * abs(rotor_speed)
                                           / (s.kt * s.gearbox_efficiency))
                drive_limit = lim
                if s.speed_limit > 0.0:
                    drive_limit = min(drive_limit, idle + (s.current_limit - idle)
                                      * (s.speed_limit - abs(rotor_speed))
                                      / (s.speed_droop * s.speed_limit))
                if s.limit_power:
                    drive_limit = min(drive_limit, idle + s.peak_power_rating
                                      / (s.gearbox_efficiency * s.kt * abs(rotor_speed)))
            self._step_key = key
            self._step_cache = (derate, back_emf, decay, response, lim,
                                rotor_speed, drive_limit)

        self.derate = derate
        self.over_temp = derate < 1.0
        bus_voltage = max(0.0, bus_voltage)
        v_cmd = self.commanded_voltage * self._scale
        v_cmd = clamp(v_cmd, -bus_voltage, bus_voltage)

        self.regenerative_overcurrent = False
        if self.mode is MotorMode.COAST or bus_voltage <= 1e-9 or derate <= 0.0:
            i = 0.0
            v_app = back_emf
            self.current_limited = False
        else:
            # Exact RL endpoint is affine in bridge voltage. Clip its requested
            # current, invert that relation, then enforce the realizable voltage.
            # This preserves inductance during retries and current-limit changes.
            requested = decay * i0 + response * (v_cmd - back_emf) / s.resistance
            desired = clamp(requested, -lim, lim)
            if desired * rotor_speed > 0.0:
                # Excess external speed is handled by feasible bridge voltage,
                # not by an unpowered speed governor inventing braking energy.
                desired = math.copysign(min(abs(desired), max(0.0, drive_limit)), desired)
            required = back_emf + s.resistance * (desired - decay * i0) / response
            v_app = clamp(required, -bus_voltage, bus_voltage)
            i = decay * i0 + response * (v_app - back_emf) / s.resistance
            self.current_limited = abs(requested) > lim + 1e-12
            # At E > Vbus + I_limit*R the diode/bridge path necessarily exceeds
            # the firmware limit. Report it rather than fabricating >bus volts.
            self.regenerative_overcurrent = abs(i) > lim + 1e-9

        self.current = i
        self.applied_voltage = v_app
        self.power_copper = i * i * s.resistance
        self.power_electrical = v_app * i
        # `current` is winding current.  With PWM the pack supplies only the
        # average output power of the H-bridge, not that winding current for the
        # entire cycle.  A shorted motor in BRAKE circulates current locally.
        if self.mode is MotorMode.COAST or bus_voltage <= 1e-9 or self.derate <= 0.0:
            return 0.0
        return self.power_electrical / bus_voltage

    def output_torque(self) -> float:
        """Torque delivered to the output shaft this step, N*m.

        Moving Coulomb drag is blended with tanh so it does not chatter; at
        rest, static friction opposes the impending torque.
        """
        s = self.spec
        w = self.velocity
        ideal = s.kt * self.current * self.ratio
        gross = (ideal * s.gearbox_efficiency if ideal * w >= 0.0
                 else ideal / s.gearbox_efficiency)
        if abs(w) < 1e-12:
            # At rest, static friction opposes the impending motion.  This also
            # makes the simulated zero-speed torque agree with `stall_torque`.
            coulomb = clamp(gross, -self.coulomb_torque, self.coulomb_torque)
        else:
            coulomb = self.coulomb_torque * math.tanh(w / 0.05)
        drag = coulomb + s.viscous_damping * self.ratio ** 2 * w
        self.torque = gross - drag
        self.power_mechanical = self.torque * w
        gearbox_loss = max(0.0, (ideal - gross) * w)
        self.power_friction = max(0.0, drag * w) + gearbox_loss
        pe = self.power_electrical
        self.efficiency = (self.power_mechanical / pe) if pe > 1e-6 else 0.0
        return self.torque

    def thermal_step(self, dt: float) -> None:
        s = self.spec
        heat = self.power_copper + self.power_friction
        cooling = (self.temperature - self.ambient_temperature) / s.thermal_resistance
        self.temperature += (heat - cooling) * dt / s.thermal_capacitance

    def integrate_free(self, dt: float) -> None:
        """Integrate the shaft when the motor is not attached to anything."""
        acc = self.output_torque() / max(self.output_inertia, 1e-12)
        self.velocity += acc * dt
        self.position += self.velocity * dt
        self.thermal_step(dt)

    def reset(self) -> None:
        self.position = 0.0
        self.velocity = 0.0
        self.temperature = self.ambient_temperature
        self._vel_integral = 0.0
        self._controller_elapsed = 0.0
        self._scale = 1.0
        self.current = 0.0
        self.torque = 0.0
        self.mode = MotorMode.COAST
        self.command = 0.0
        self.sent = None
        self.commanded_voltage = 0.0
        self.applied_voltage = 0.0
        self.power_electrical = 0.0
        self.power_mechanical = 0.0
        self.power_copper = 0.0
        self.power_friction = 0.0
        self.efficiency = 0.0
        self.derate = 1.0
        self.over_temp = False
        self.current_limited = False
        self.regenerative_overcurrent = False

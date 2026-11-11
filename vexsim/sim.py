"""The simulator: wires battery, motors, dynamics and sensors together.

Order of operations in one physics step, which matters:

  0. If the brain's 5 ms write to the motors falls due, each motor takes the
     last command user code sent it.
  1. Each motor's firmware controller picks a voltage.
  2. The electrical side is solved against the battery.  Bus voltage depends on
     total current and total current depends on bus voltage, so it iterates a
     few times -- three passes is plenty at these resistances.
  3. If the pack is over the brain's total current budget, every motor gets
     scaled back together, which is what a real brain does.
  4. Forces, then integration.
  5. Heat, sensors, log.

User code runs on its own period (10 ms by default, like a real task loop), not
every physics tick, so a routine written against this sim sees the same control
latency it will see on the field.  A command it sends reaches the motor at the
brain's next write: every 5 ms, PROS's `DATA_WRITE_RATE`
(https://docs.rs/pros-devices/0.2.0/src/pros_devices/smart/motor.rs.html).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .battery import Battery
from .chassis import Chassis
from .dynamics import Dynamics
from .motor import Motor
from .sensors import IMU, DistanceSensor, Encoder, GPSSensor, Odometry, RotationSensor
from .units import G, INCH, clamp, to_deg, to_inches, to_rpm

GUESSES: Dict[str, str] = {
    "Motor write phase": "The brain writes to the motors at 2.5 ms and every 5 ms after. PROS gives the 5 ms "
                         "rate but not where the writes fall against the user's loop: that depends on when the "
                         "program started. With a 10 ms loop every command waits the same time, from 0 to 5 ms. "
                         "A loop at 0, 10, 20 ms, like mclib's in the planner, waits 2.5 ms, the middle. "
                         "`Simulator.run` calls its control 1 ms earlier, at 9, 19 ms, so it waits 3.5 ms.",
}


class DataLog:
    """Column store for every output variable, with CSV/summary export."""

    def __init__(self):
        self.columns: Dict[str, List[float]] = {}
        self.n = 0

    def record(self, row: Dict[str, float]) -> None:
        if self.n == 0:
            for k in row:
                self.columns[k] = []
        columns = self.columns
        # setdefault([0.0] * self.n) built and threw away a back-fill list on
        # every key of every row, which made logging cost O(rows^2). Only a key
        # that is genuinely new needs the back-fill.
        for k, v in row.items():
            column = columns.get(k)
            if column is None:
                column = columns[k] = [0.0] * self.n
            column.append(v)
        self.n += 1

    def __getitem__(self, key: str) -> List[float]:
        return self.columns[key]

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def keys(self):
        return self.columns.keys()

    def last(self, key: str) -> float:
        return self.columns[key][-1]

    def max(self, key: str) -> float:
        return max(self.columns[key])

    def min(self, key: str) -> float:
        return min(self.columns[key])

    def mean(self, key: str) -> float:
        c = self.columns[key]
        return sum(c) / len(c) if c else 0.0

    def integrate(self, key: str, dt_key: str = "t") -> float:
        """Trapezoid integral of a column against time."""
        ys = self.columns[key]
        ts = self.columns[dt_key]
        total = 0.0
        for i in range(1, len(ys)):
            total += 0.5 * (ys[i] + ys[i - 1]) * (ts[i] - ts[i - 1])
        return total

    def to_csv(self, path: str) -> None:
        keys = list(self.columns)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(self.n):
                w.writerow([f"{self.columns[k][i]:.6g}" for k in keys])

    def downsample(self, every: int) -> "DataLog":
        out = DataLog()
        for i in range(0, self.n, every):
            out.record({k: v[i] for k, v in self.columns.items()})
        return out


@dataclass
class SimConfig:
    dt: float = 0.001              # public simulation tick, s
    control_period: float = 0.010  # how often user code runs, s
    log_every: int = 10            # log one row per N physics steps; 0 = off
    brain_current_limit: float = 20.0   # A, total across all ports
    gravity: float = G
    field_half: float = 140.5 * INCH / 2.0  # V5RC inside perimeter, m
    log_per_motor: bool = True
    log_per_wheel: bool = True
    max_physics_dt: float = 0.0005  # internal integration bound, s
    motor_write_period: float = 0.005  # brain-to-motor writes, s (PROS DATA_WRITE_RATE)
    motor_write_phase: float = 0.0025  # first write, s; see GUESSES


class Simulator:
    def __init__(self, chassis: Chassis, battery: Optional[Battery] = None,
                 config: Optional[SimConfig] = None):
        self.chassis = chassis
        self.battery = battery or Battery()
        self.config = config or SimConfig()
        self.chassis.gravity = self.config.gravity
        self.dynamics = Dynamics(chassis, self.config.gravity)
        self.mechanism_motors: List[Motor] = []
        for motor in chassis.motors:
            motor.on_brain = True
        self._next_write = self.config.motor_write_phase
        self.t = 0.0
        self.steps = 0
        self.log = DataLog()

        # sensors
        self.imu = IMU()
        self.gps = GPSSensor()
        self.distance = DistanceSensor(field_half=self.config.field_half)
        self.encoders: Dict[str, Encoder] = {}
        for i, m in enumerate(chassis.motors):
            ticks = {"red": 1800.0, "green": 900.0, "blue": 300.0}.get(
                m.cartridge, 900.0)
            if m.spec.fixed_ratio > 0.0:
                ticks = 900.0  # fixed 200 rpm equivalent, not a removable cartridge
            self.encoders[m.name] = Encoder(ticks_per_rev=ticks, seed=100 + i)
        self.odometry = Odometry(track_width=chassis.track_width)
        # tracking wheels, if this chassis has them, get their own sensors and
        # their mounting offsets handed to the odometry
        self.tracking: Dict[str, RotationSensor] = {}
        for u in chassis.wheel_units:
            if u.name in ("track_parallel", "track_perp"):
                self.tracking[u.name] = RotationSensor(seed=200 + len(self.tracking))
                if u.name == "track_perp":
                    self.odometry.perp_offset = u.x - chassis.cg_x
                else:
                    self.odometry.parallel_offset = u.y - chassis.cg_y

        # energy book-keeping
        self.energy_battery = 0.0     # J out of the pack
        self.energy_mechanical = 0.0  # J into the wheels
        self.energy_copper = 0.0
        self.energy_slip = 0.0
        self.energy_rolling = 0.0
        self.energy_motor_friction = 0.0
        self.energy_gear = 0.0
        self.energy_drag = 0.0

        self._control_accum = 0.0
        self._on_control: Optional[Callable[["Simulator"], None]] = None

    # ------------------------------------------------------------- accessors
    @property
    def state(self):
        return self.dynamics.state

    @property
    def motors(self) -> List[Motor]:
        return self.chassis.motors

    def motor(self, name: str) -> Motor:
        for m in self.motors:
            if m.name == name:
                return m
        raise KeyError(name)

    def set_pose(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.dynamics.reset(x, y, theta)
        self.imu.reset(theta)
        self.odometry.reset(x, y, theta)

    def power_mechanisms(self, motors: List[Motor]) -> None:
        """Put a mechanism's motors on the battery with the drive's.  Each
        step's bus solve then sets their current too, and the mechanism moves
        them with the torque that current makes.  Their commands go out on the
        brain's writes like the drive's."""
        self.mechanism_motors = list(motors)
        for motor in self.mechanism_motors:
            motor.on_brain = True

    # ------------------------------------------------------------------ step
    def _solve_electrical(self, dt: float) -> float:
        """Converged bus solve, then a feasible common duty/current budget.

        Controller clocks and inductance advance once, even when a voltage or
        current-limit search retries the electrical equations many times.
        """
        motors = self.chassis.motors + self.mechanism_motors
        battery = self.battery
        terminal_voltage = battery.terminal_voltage
        initial_currents = [m.current for m in motors]
        pairs = list(zip(motors, initial_currents))
        for m in motors:
            m.advance_control(dt)

        def solve(scale: float) -> float:
            for m in motors:
                m._scale = scale
            bus = max(0.0, battery.voltage)
            previous_error = None
            for _ in range(80):
                total = 0.0
                for m, i0 in pairs:
                    total += m.electrical(bus, dt, update_control=False,
                                          initial_current=i0)
                error = terminal_voltage(total) - bus
                if abs(error) <= 1e-10:
                    return total
                # Normal V5 pack impedances converge directly in a few passes.
                # Relax only a poorly contracting iteration, without advancing
                # any motor or controller state.
                step = error
                if previous_error is not None and abs(error) > 0.8 * previous_error:
                    step *= 0.5
                previous_error = abs(error)
                bus += step
            raise RuntimeError("electrical bus solve did not converge")

        limit = max(0.0, min(self.config.brain_current_limit,
                             battery.discharge_current_limit(dt)))
        total = solve(1.0)
        if total > limit + 1e-10:
            # Scaling voltage once is insufficient on the flat winding-current
            # plateau. Bracket the allowed duty all the way to zero and retain
            # only the feasible side of the search.
            low, high = 0.0, 1.0
            total = solve(low)
            if total > limit + 1e-10:
                raise RuntimeError("no feasible drive duty within battery current budget")
            # A fixed 40 halvings resolved the duty to 1e-12, far past what a
            # 1e-10 A current budget can distinguish, and each halving is a
            # full bus solve. Stop once the bracket stops mattering.
            while high - low > 1e-9:
                middle = 0.5 * (low + high)
                candidate = solve(middle)
                if candidate <= limit:
                    low = middle
                else:
                    high = middle
            total = solve(low)

        self.battery.update(total, dt)
        return total

    def step(self, dt: Optional[float] = None) -> None:
        dt = self.config.dt if dt is None else dt
        bound = self.config.max_physics_dt
        if not math.isfinite(dt) or dt <= 0.0 or not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("simulation timestep and max_physics_dt must be finite and positive")

        # user code on its own clock
        if self._on_control is not None:
            self._control_accum += dt
            if self._control_accum >= self.config.control_period - 1e-12:
                self._control_accum -= self.config.control_period
                self._update_sensors()
                self._on_control(self)

        # The governed motor/tire system is stiffer than the unrestricted DC
        # model. Keep integration below its measured convergence threshold,
        # without changing public ticks, user-control cadence or log frequency.
        slices = max(1, math.ceil(dt / bound))
        for index in range(slices):
            self._write_motors(self.t + index * dt / slices)
            self._integrate_physics(dt / slices)

        self.t += dt
        self.steps += 1
        self._update_sensors()
        if self.config.log_every > 0 and self.steps % self.config.log_every == 0:
            self.log.record(self.sample())

    def _write_motors(self, now: float) -> None:
        """Each brain write that has fallen due by `now` hands every motor its latest command."""
        while self._next_write <= now + 1e-9:
            for motor in self.chassis.motors + self.mechanism_motors:
                motor.receive()
            self._next_write += self.config.motor_write_period

    def _integrate_physics(self, dt: float) -> None:
        self._solve_electrical(dt)
        self.dynamics.step(dt)

        # Heat and the energy ledger in one pass over the motors: this ran
        # five separate generator sums over the same six objects, 2000 times
        # a simulated second.
        copper = mechanical = friction = 0.0
        for m in self.chassis.motors:
            m.thermal_step(dt)
            copper += m.power_copper
            mechanical += m.power_mechanical
            friction += m.power_friction

        gear_loss = 0.0
        for u in self.chassis.wheel_units:
            gear_loss += u.power_gear_loss

        dyn = self.dynamics
        self.energy_battery += self.battery.power_out * dt
        self.energy_copper += copper * dt
        # Use the same pre-integration shaft speeds as the torque/loss solve.
        # Multiplying old torque by newly advanced wheel speed invents work.
        self.energy_mechanical += (mechanical - gear_loss) * dt
        self.energy_motor_friction += friction * dt
        self.energy_gear += gear_loss * dt
        self.energy_drag += dyn.drag_power * dt
        self.energy_slip += dyn.wheel_slip_power * dt
        self.energy_rolling += dyn.rolling_loss_power * dt

        if not math.isfinite(self.state.u + self.state.r + self.state.x):
            raise RuntimeError(
                f"simulation diverged at t={self.t:.3f}s -- check the drag "
                f"coefficients or lower SimConfig.dt")

    def _update_sensors(self) -> None:
        st = self.state
        self.imu.update(st.r, st.ax, st.ay, self.t)
        self.gps.update(st.x, st.y, st.theta, self.t)
        self.distance.update(st.x, st.y, st.theta, self.t)
        for m in self.motors:
            self.encoders[m.name].update(m.position, self.t)

        left = self._side_travel(self.chassis.left_units())
        right = self._side_travel(self.chassis.right_units())
        perp = None
        for u in self.chassis.wheel_units:
            sensor = self.tracking.get(u.name)
            if sensor is None:
                continue
            sensor.update(u.angle, self.t)
            if u.name == "track_perp":
                perp = sensor.position * u.radius
            else:
                left = right = sensor.position * u.radius
        self.odometry.update(left, right, self.imu.heading, perp)

    def _side_travel(self, units) -> float:
        if not units:
            return 0.0
        # use the encoder-reported angle, quantised, not the true wheel angle
        total = 0.0
        count = 0
        for u in units:
            for m in u.motors:
                enc = self.encoders[m.name]
                d = -1.0 if m.reversed else 1.0
                total += enc.position * d / u.gear_ratio * u.radius
                count += 1
        return total / count if count else 0.0

    # ------------------------------------------------------------------- run
    def run(self, seconds: float, control: Optional[Callable] = None,
            until: Optional[Callable[["Simulator"], bool]] = None) -> "Simulator":
        if control is not None:
            self._on_control = control
        n = int(round(seconds / self.config.dt))
        for _ in range(n):
            self.step()
            if until is not None and until(self):
                break
        return self

    # ---------------------------------------------------------------- output
    def sample(self) -> Dict[str, float]:
        """Every output variable for this instant, flattened for the log."""
        st = self.state
        ch = self.chassis
        dyn = self.dynamics
        ms = self.motors

        row: Dict[str, float] = {
            "t": self.t,
            # pose, truth
            "x_m": st.x, "y_m": st.y, "heading_rad": st.theta,
            "x_in": to_inches(st.x), "y_in": to_inches(st.y),
            "heading_deg": to_deg(st.theta),
            # velocities
            "vel_fwd_mps": st.u, "vel_lat_mps": st.v, "yaw_rate_dps": to_deg(st.r),
            "speed_mps": st.speed, "speed_inps": to_inches(st.speed),
            # accelerations
            "accel_fwd_mps2": st.ax, "accel_lat_mps2": st.ay,
            "accel_fwd_g": st.ax / G, "yaw_accel_dps2": to_deg(st.alpha),
            # forces
            "force_fwd_N": dyn.total_fx, "force_lat_N": dyn.total_fy,
            "yaw_moment_Nm": dyn.total_mz, "normal_total_N": dyn.total_normal,
            "tipping_margin": dyn.tipping_margin,
            "lateral_transfer_ratio": dyn.lateral_transfer_ratio,
            "pitch_transfer_ratio": dyn.pitch_transfer_ratio,
            "normal_balance_feasible": float(dyn.normal_balance_feasible),
            "normal_pitch_residual_Nm": dyn.normal_moment_residual[0],
            "normal_roll_residual_Nm": dyn.normal_moment_residual[1],
            "any_wheel_slipping": 1.0 if dyn.any_slipping else 0.0,
            # electrical
            "battery_V": self.battery.voltage,
            "battery_A": self.battery.current,
            "battery_W": self.battery.power_out,
            "battery_soc": self.battery.soc,
            "battery_loss_W": self.battery.power_loss,
            "battery_rejected_W": self.battery.power_rejected,
            "battery_chemical_W": self.battery.power_chemical,
            "brownout": 1.0 if self.battery.brownout else 0.0,
            "motor_current_total_A": sum(abs(m.current) for m in ms),
            "electrical_W": sum(m.power_electrical for m in ms),
            "mechanical_W": sum(m.power_mechanical for m in ms),
            "copper_loss_W": sum(m.power_copper for m in ms),
            "friction_loss_W": sum(m.power_friction for m in ms),
            "gear_loss_W": sum(u.power_gear_loss for u in ch.wheel_units),
            "slip_loss_W": dyn.wheel_slip_power,
            "rolling_loss_W": dyn.rolling_loss_power,
            "drag_loss_W": dyn.drag_power,
            "scrub_loss_W": dyn.scrub_loss_power,  # already included in drag_loss_W
            "drivetrain_efficiency": self._efficiency(),
            "max_motor_temp_C": max(m.temperature for m in ms) if ms else 0.0,
            "min_motor_derate": min(m.derate for m in ms) if ms else 1.0,
            "kinetic_energy_J": dyn.kinetic_energy,
            "energy_battery_J": self.energy_battery,
            "energy_rejected_J": self.battery.energy_rejected,
            "energy_slip_J": self.energy_slip,
            # sensors
            "imu_heading_deg": to_deg(self.imu.heading),
            "imu_rate_dps": to_deg(self.imu.rate),
            "imu_valid": float(self.imu.valid),
            "odom_x_in": to_inches(self.odometry.x),
            "odom_y_in": to_inches(self.odometry.y),
            "odom_theta_deg": to_deg(self.odometry.theta),
            "odom_error_in": to_inches(math.hypot(self.odometry.x - st.x,
                                                  self.odometry.y - st.y)),
            "gps_x_in": to_inches(self.gps.x), "gps_y_in": to_inches(self.gps.y),
            "distance_sensor_in": to_inches(self.distance.distance),
            "distance_sensor_valid": float(self.distance.valid),
        }

        if self.config.log_per_motor:
            for m in ms:
                p = f"m_{m.name}"
                row[p + "_V"] = m.applied_voltage
                row[p + "_cmdV"] = m.commanded_voltage
                row[p + "_A"] = m.current
                row[p + "_Nm"] = m.torque
                row[p + "_rpm"] = to_rpm(m.velocity)
                row[p + "_W_in"] = m.power_electrical
                row[p + "_W_out"] = m.power_mechanical
                row[p + "_eff"] = m.efficiency
                row[p + "_tempC"] = m.temperature
                row[p + "_derate"] = m.derate
                row[p + "_ilimited"] = 1.0 if m.current_limited else 0.0

        if self.config.log_per_wheel:
            for u in ch.wheel_units:
                p = f"w_{u.name}"
                row[p + "_rpm"] = to_rpm(u.omega)
                row[p + "_Fz_N"] = u.fz
                row[p + "_Fx_N"] = u.fx
                row[p + "_Fy_N"] = u.fy
                row[p + "_slip_ratio"] = u.kappa
                row[p + "_slip_angle_deg"] = to_deg(math.atan(u.tan_alpha))
                row[p + "_surface_mps"] = u.omega * u.radius
                row[p + "_ground_mps"] = u.vx
                row[p + "_slipping"] = 1.0 if u.slipping else 0.0
                row[p + "_slip_measure"] = u.slip_measure
                row[p + "_torque_Nm"] = u.drive_torque

        return row

    def _efficiency(self) -> float:
        pin = self.battery.power_out
        if pin <= 1e-6:
            return 0.0
        st = self.state
        useful = abs(self.chassis.mass * st.ax * st.u) + abs(
            self.chassis.inertia_z * st.alpha * st.r)
        return clamp(useful / pin, 0.0, 1.0)

    def energy_report(self) -> Dict[str, float]:
        """Electrical/shaft work ledger, not a tire elastic-storage model.

        Optional inductance uses endpoint power quadrature, so its stored
        energy and finite-step integration error remain in unaccounted_J.
        """
        out = self.energy_battery
        return {
            "battery_J": self.energy_battery,
            "battery_rejected_J": self.battery.energy_rejected,
            "copper_J": self.energy_copper,
            "wheel_shaft_J": self.energy_mechanical,
            "tire_slip_J": self.energy_slip,
            "rolling_J": self.energy_rolling,
            "motor_friction_J": self.energy_motor_friction,
            "external_gear_J": self.energy_gear,
            "chassis_drag_J": self.energy_drag,
            "battery_mAh": self.battery.charge_used_ah * 1000.0,
            "unaccounted_J": (out - self.energy_copper - self.energy_mechanical
                               - self.energy_motor_friction - self.energy_gear),
        }

    def stop_all(self, brake_mode=None) -> None:
        from .motor import BrakeMode
        for m in self.motors:
            m.stop(brake_mode or BrakeMode.COAST)

    def reset(self) -> None:
        self.t = 0.0
        self.steps = 0
        self.log = DataLog()
        self.battery.reset()
        self.dynamics.reset()
        for m in self.motors:
            m.reset()
        for e in self.encoders.values():
            e.reset()
        for sensor in self.tracking.values():
            sensor.reset()
        self.imu.reset()
        self.gps.reset()
        self.distance.reset()
        self.odometry.reset()
        self._control_accum = 0.0
        self.energy_battery = self.energy_copper = self.energy_mechanical = 0.0
        self.energy_slip = self.energy_rolling = 0.0
        self.energy_motor_friction = self.energy_gear = self.energy_drag = 0.0

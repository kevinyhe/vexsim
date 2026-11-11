"""Physics checks: does the model reproduce numbers we already know?

Run:  python3 -m unittest discover -s tests -v
   or python3 tests/test_physics.py
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import (Battery, Chassis, Dynamics, Feedforward, InfiniteBattery,
                    Motor, PID, Robot, Simulator, TrapezoidProfile, WheelUnit,
                    differential_drive, presets, wheels)
from vexsim.dynamics import normal_loads
from vexsim.motor import BrakeMode, cortex_393, v5_11w
from vexsim.sim import SimConfig
from vexsim.units import G, to_inches, to_rpm
from vexsim.wheels import tire_forces


class TestMotorAgainstDatasheet(unittest.TestCase):
    """The 11 W V5 motor's published numbers, per cartridge."""

    CASES = [
        # cartridge, free rpm, stall N*m, tolerance on torque
        ("red", 100.0, 2.10, 0.20),
        ("green", 200.0, 1.05, 0.10),
        ("blue", 600.0, 0.35, 0.05),
    ]

    def test_free_speed_and_stall_torque(self):
        for cart, free_rpm, stall, tol in self.CASES:
            m = Motor(cartridge=cart)
            self.assertAlmostEqual(to_rpm(m.free_speed), free_rpm, delta=free_rpm * 0.03)
            self.assertAlmostEqual(m.stall_torque, stall, delta=tol)

    def test_peak_power_is_eleven_watts(self):
        spec = v5_11w()
        self.assertAlmostEqual(spec.peak_power * spec.gearbox_efficiency, 11.0, delta=0.5)

    def test_stall_current_is_the_limit(self):
        m = Motor(cartridge="green")
        m.velocity = 0.0
        m.set_voltage(12.0)
        m.electrical(12.0, 0.001)
        self.assertAlmostEqual(m.current, m.current_limit, places=6)

    def test_pack_current_accounts_for_pwm_duty_cycle(self):
        m = Motor(cartridge="green")
        m.set_voltage(6.0)
        draw = m.electrical(12.0, 0.001)
        # Current limiting can reduce the applied voltage below the command.
        self.assertLessEqual(abs(m.applied_voltage), 6.0)
        self.assertAlmostEqual(draw, m.applied_voltage * m.current / 12.0, places=9)
        self.assertAlmostEqual(12.0 * draw, m.power_electrical, places=9)

    def test_brake_current_circulates_without_pack_draw(self):
        m = Motor(cartridge="green")
        m.velocity = m.free_speed * 0.5
        m.stop(BrakeMode.BRAKE)
        draw = m.electrical(12.0, 0.001)
        self.assertEqual(draw, 0.0)
        self.assertLess(m.current, 0.0)

    def test_optional_inductance_delays_current(self):
        algebraic = Motor(electrical_dynamics=False)
        dynamic = Motor(electrical_dynamics=True)
        algebraic.set_voltage(6.0)
        dynamic.set_voltage(6.0)
        algebraic.electrical(12.0, 10e-6)
        dynamic.electrical(12.0, 10e-6)
        self.assertGreater(dynamic.current, 0.0)
        self.assertLess(dynamic.current, algebraic.current * 0.05)

    def test_zero_speed_output_matches_stall_torque(self):
        m = Motor(cartridge="red")
        m.set_voltage(12.0)
        m.electrical(12.0, 0.001)
        self.assertAlmostEqual(m.output_torque(), m.stall_torque, places=9)

    def test_no_current_at_free_speed(self):
        m = Motor(cartridge="green")
        m.velocity = m.free_speed
        m.set_voltage(12.0)
        m.electrical(12.0, 0.001)
        self.assertLess(abs(m.current), 0.2)

    def test_back_emf_opposes_motion(self):
        m = Motor(cartridge="green")
        m.set_voltage(6.0)
        m.velocity = 0.0
        m.electrical(12.0, 0.001)
        slow = m.current
        m.velocity = m.free_speed * 0.4
        m.electrical(12.0, 0.001)
        self.assertLess(m.current, slow)

    def test_coast_draws_nothing(self):
        m = Motor(cartridge="green")
        m.velocity = m.free_speed * 0.5
        m.stop(BrakeMode.COAST)
        draw = m.electrical(12.0, 0.001)
        self.assertEqual(draw, 0.0)
        self.assertEqual(m.current, 0.0)

    def test_brake_produces_regen_torque(self):
        m = Motor(cartridge="green")
        m.velocity = m.free_speed * 0.5
        m.stop(BrakeMode.BRAKE)
        m.electrical(12.0, 0.001)
        self.assertLess(m.current, 0.0)          # current flows backwards
        self.assertLess(m.output_torque(), 0.0)  # torque opposes the spin

    def test_393_matches_its_datasheet(self):
        m = Motor(spec=cortex_393(), cartridge="393_torque")
        self.assertAlmostEqual(to_rpm(m.free_speed), 100.0, delta=5.0)
        self.assertAlmostEqual(m.stall_torque, 1.67, delta=0.20)
        self.assertAlmostEqual(m.spec.peak_power * m.spec.gearbox_efficiency,
                               4.4, delta=0.3)

    def test_gearing_trades_speed_for_torque(self):
        red, blue = Motor(cartridge="red"), Motor(cartridge="blue")
        self.assertAlmostEqual(red.free_speed * red.stall_torque,
                               blue.free_speed * blue.stall_torque,
                               delta=0.30 * red.free_speed * red.stall_torque)
        self.assertAlmostEqual(blue.free_speed / red.free_speed, 6.0, delta=0.1)


class TestThermal(unittest.TestCase):
    def test_stall_heats_the_motor_to_its_steady_state(self):
        m = Motor(cartridge="green")
        expected = m.ambient_temperature + \
            (m.current_limit ** 2 * m.spec.resistance) * m.spec.thermal_resistance
        for _ in range(200000):     # 200 s at 1 ms
            m.velocity = 0.0
            m.set_voltage(12.0)
            m.electrical(12.0, 0.001)
            m.output_torque()
            m.thermal_step(0.001)
        self.assertGreater(m.temperature, 45.0)
        self.assertLess(m.temperature, expected + 1.0)

    def test_derating_steps(self):
        m = Motor()
        # Published V5 11 W firmware schedule, not the old model's 20% steps.
        for temp, expected in ((20, 1.0), (57, 0.5), (62, 0.25),
                               (67, 0.125), (70, 0.0), (80, 0.0)):
            m.temperature = temp
            self.assertAlmostEqual(m._thermal_derate(), expected)

    def test_a_cool_motor_does_not_derate(self):
        s = Simulator(presets.six_motor_200())
        s.run(2.0, lambda sim: [mm.set_voltage(12.0) for mm in sim.motors])
        self.assertEqual(min(m.derate for m in s.motors), 1.0)


class TestBattery(unittest.TestCase):
    def test_sag_follows_ohms_law(self):
        b = Battery(soc=1.0)
        ocv = b.open_circuit_voltage()
        b.update(10.0, 0.001)
        self.assertAlmostEqual(b.voltage, ocv - 10.0 * b.internal_resistance, places=6)

    def test_discharge_reduces_soc(self):
        b = Battery(soc=1.0)
        for _ in range(1000):
            b.update(10.0, 0.01)      # 10 A for 10 s
        self.assertAlmostEqual(b.charge_used_ah, 10.0 * 10.0 / 3600.0, places=6)
        self.assertLess(b.soc, 1.0)

    def test_ocv_curve_is_monotonic(self):
        b = Battery()
        last = -1.0
        for i in range(101):
            b.soc = i / 100.0
            v = b.open_circuit_voltage()
            self.assertGreaterEqual(v, last)
            last = v

    def test_exhausted_pack_cannot_drive_the_robot(self):
        speeds = []
        # A lightly loaded governed motor can hold speed at 5% SOC. Exhausted
        # finite charge, unlike a specific uncalibrated OCV, is a hard bound.
        for soc in (1.0, 0.0):
            r = Robot(presets.six_motor_200(), Battery(soc=soc),
                      SimConfig(log_every=0))
            r.tank(12.0, 12.0)
            r.sim.run(4.0)
            speeds.append(r.sim.state.u)
        self.assertGreater(speeds[0], 0.5)
        self.assertEqual(speeds[1], 0.0)


class TestNormalLoads(unittest.TestCase):
    def setUp(self):
        self.c = presets.six_motor_200()
        self.pos = [(u.x - self.c.cg_x, u.y - self.c.cg_y) for u in self.c.wheel_units]

    def test_static_loads_sum_to_weight(self):
        loads = normal_loads(self.c, 0.0, 0.0, self.pos)
        self.assertAlmostEqual(sum(loads), self.c.mass * G, places=6)

    def test_static_loads_are_even_on_a_symmetric_robot(self):
        loads = normal_loads(self.c, 0.0, 0.0, self.pos)
        self.assertAlmostEqual(max(loads), min(loads), places=6)

    def test_acceleration_transfers_load_rearward(self):
        loads = normal_loads(self.c, 5.0, 0.0, self.pos)
        front = sum(f for f, p in zip(loads, self.pos) if p[0] > 0)
        rear = sum(f for f, p in zip(loads, self.pos) if p[0] < 0)
        self.assertGreater(rear, front)
        self.assertAlmostEqual(sum(loads), self.c.mass * G, places=6)

    def test_pitch_moment_balances(self):
        ax = 4.0
        loads = normal_loads(self.c, ax, 0.0, self.pos)
        moment = sum(f * p[0] for f, p in zip(loads, self.pos))
        expected = -self.c.cg_height * self.c.mass * ax
        self.assertAlmostEqual(moment, expected, delta=abs(expected) * 0.02 + 1e-6)

    def test_lateral_acceleration_transfers_sideways(self):
        loads = normal_loads(self.c, 0.0, 6.0, self.pos)
        left = sum(f for f, p in zip(loads, self.pos) if p[1] > 0)
        right = sum(f for f, p in zip(loads, self.pos) if p[1] < 0)
        self.assertGreater(right, left)

    def test_wheels_lift_but_loads_stay_positive(self):
        loads = normal_loads(self.c, 40.0, 0.0, self.pos)
        self.assertTrue(all(f >= 0.0 for f in loads))
        self.assertAlmostEqual(sum(loads), self.c.mass * G, places=6)
        self.assertEqual(min(loads), 0.0)

    def test_gravity_sets_total_normal_load(self):
        moon_g = 1.62
        loads = normal_loads(self.c, 0.0, 0.0, self.pos, moon_g)
        self.assertAlmostEqual(sum(loads), self.c.mass * moon_g, places=6)

    def test_simulator_passes_configured_gravity_to_dynamics(self):
        moon_g = 1.62
        s = Simulator(self.c, config=SimConfig(gravity=moon_g, log_every=0))
        s.step()
        self.assertAlmostEqual(s.dynamics.total_normal, self.c.mass * moon_g, places=6)
        self.assertAlmostEqual(s.chassis.weight, self.c.mass * moon_g, places=6)


class TestTireModel(unittest.TestCase):
    def test_an_omni_keeps_its_forward_grip_while_sliding_sideways(self):
        """Rollers are a separate contact, so sideways costs it almost nothing."""
        t = wheels.omni(3.25).tire
        straight, _ = tire_forces(t, 12.0, t.slip_ratio_peak, 0.0)
        sliding, fy = tire_forces(t, 12.0, t.slip_ratio_peak, 1.0)
        self.assertGreater(sliding, 0.85 * straight)
        self.assertLess(abs(fy), 0.32 * 12.0)      # mu_lat is 0.26

    def test_an_omni_still_scrubs_going_sideways(self):
        """Rollers roll, but a half-inch plastic roller on foam is not a caster."""
        t = wheels.omni(3.25).tire
        _, fy = tire_forces(t, 12.0, 0.0, 1.0)
        self.assertGreater(abs(fy), 0.15 * 12.0)
        self.assertLess(abs(fy), 0.32 * 12.0)

    def test_a_traction_wheel_obeys_the_friction_ellipse(self):
        t = wheels.traction(3.25).tire
        straight, _ = tire_forces(t, 12.0, t.slip_ratio_peak, 0.0)
        sliding, fy = tire_forces(t, 12.0, t.slip_ratio_peak, 1.0)
        self.assertLess(sliding, 0.25 * straight)
        self.assertAlmostEqual(abs(fy), t.mu_lat_slide * 12.0, delta=0.05 * 12.0)

    def test_no_slip_no_force(self):
        t = wheels.omni(4.0).tire
        fx, fy = tire_forces(t, 30.0, 0.0, 0.0)
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0)

    def test_peak_force_is_mu_times_load(self):
        t = wheels.traction(4.0).tire
        fx, _ = tire_forces(t, 50.0, t.slip_ratio_peak, 0.0)
        self.assertAlmostEqual(fx, t.mu_long_static * 50.0, delta=0.01)

    def test_force_falls_off_once_it_slides(self):
        t = wheels.traction(4.0).tire
        peak, _ = tire_forces(t, 50.0, t.slip_ratio_peak, 0.0)
        sliding, _ = tire_forces(t, 50.0, 5.0, 0.0)
        self.assertLess(sliding, peak)
        self.assertAlmostEqual(sliding, t.mu_long_slide * 50.0, delta=0.5)

    def test_an_omni_goes_sideways_far_more_easily_than_a_traction_wheel(self):
        """Far more easily, but not freely -- the rollers still plow foam.

        A traction wheel holds 0.81 of its load sideways on foam (foam.TRACTION_GRIP)
        and an omni about a quarter of its load."""
        _, omni_side = tire_forces(wheels.omni(4.0).tire, 50.0, 0.0, 1.0)
        _, trac_side = tire_forces(wheels.traction(4.0).tire, 50.0, 0.0, 1.0)
        ratio = abs(omni_side) / abs(trac_side)
        self.assertLess(ratio, 0.40)
        self.assertGreater(ratio, 0.10)

    def test_force_opposes_lateral_slip(self):
        t = wheels.traction(4.0).tire
        _, fy = tire_forces(t, 50.0, 0.0, 0.5)
        self.assertLess(fy, 0.0)

    def test_combined_slip_stays_inside_the_friction_envelope(self):
        t = wheels.traction(4.0).tire
        fz = 40.0
        limit = max(t.mu_long_static, t.mu_lat_static) * fz * 1.001
        for k in (-2.0, -0.2, 0.0, 0.05, 0.2, 2.0):
            for a in (-1.0, -0.1, 0.0, 0.1, 1.0):
                fx, fy = tire_forces(t, fz, k, a)
                self.assertLessEqual(math.hypot(fx, fy), limit)

    def test_zero_load_carries_no_force(self):
        fx, fy = tire_forces(wheels.omni(4.0).tire, 0.0, 0.5, 0.5)
        self.assertEqual((fx, fy), (0.0, 0.0))


class TestStraightLine(unittest.TestCase):
    def test_top_speed_matches_the_gearing(self):
        for name in ("four_motor_200", "six_motor_600", "speed_base"):
            c = presets.get(name)
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(6.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
            theory = c.theoretical_free_speed()
            self.assertLess(s.state.u, theory)                 # losses cost you some
            self.assertGreater(s.state.u, theory * 0.90)       # but not much

    def test_acceleration_never_beats_the_tires(self):
        c = presets.get("six_motor_600")
        s = Simulator(c)
        s.run(1.5, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        limit = c.max_tractive_force() / c.mass
        self.assertLessEqual(s.log.max("accel_fwd_mps2"), limit * 1.05)

    def test_nothing_moves_without_voltage(self):
        s = Simulator(presets.six_motor_200())
        s.run(1.0)
        self.assertAlmostEqual(s.state.x, 0.0, places=9)
        self.assertAlmostEqual(s.state.u, 0.0, places=9)

    def test_a_robot_coasts_to_a_stop(self):
        s = Simulator(presets.six_motor_600())
        s.run(2.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        moving = s.state.u
        s.run(6.0, lambda sim: [m.stop() for m in sim.motors])
        self.assertGreater(moving, 0.5)
        self.assertLess(abs(s.state.u), 0.05)

    def test_reverse_is_symmetric(self):
        fwd = Simulator(presets.six_motor_200(), InfiniteBattery(), SimConfig(log_every=0))
        fwd.run(3.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        rev = Simulator(presets.six_motor_200(), InfiniteBattery(), SimConfig(log_every=0))
        rev.run(3.0, lambda sim: [m.set_voltage(-12.0) for m in sim.motors])
        self.assertAlmostEqual(fwd.state.x, -rev.state.x, places=6)

    def test_heavier_is_slower_to_accelerate(self):
        times = []
        for lb in (12.0, 30.0):
            c = differential_drive(mass_lb=lb, wheel="omni_3.25", cartridge="blue",
                                   motors_per_side=3, wheels_per_side=3)
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(2.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors],
                  until=lambda sim: sim.state.x > 0.5)
            times.append(s.t)
        self.assertGreater(times[1], times[0])

    def test_a_bigger_wheel_goes_faster(self):
        speeds = []
        for w in ("omni_2.75", "omni_4"):
            c = differential_drive(wheel=w, cartridge="green")
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(5.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
            speeds.append(s.state.u)
        self.assertGreater(speeds[1], speeds[0] * 1.3)


class TestTurning(unittest.TestCase):
    def test_positive_turn_is_counter_clockwise(self):
        r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
        r.arcade(0.0, 6.0)
        r.sim.run(0.5)
        self.assertGreater(r.sim.state.theta, 0.0)

    def test_a_point_turn_stays_put(self):
        r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
        r.tank(8.0, -8.0)
        r.sim.run(1.0)
        self.assertLess(math.hypot(r.sim.state.x, r.sim.state.y), 0.02)
        self.assertGreater(abs(r.sim.state.r), 1.0)

    def test_omnis_turn_more_easily_than_traction(self):
        rates = []
        for w in ("traction_3.25", "omni_3.25"):
            c = differential_drive(wheel=w, cartridge="green",
                                   motors_per_side=3, wheels_per_side=3)
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(3.0, lambda sim: self._spin(sim))
            rates.append(abs(s.state.r))
        self.assertGreater(rates[1], rates[0])

    @staticmethod
    def _spin(sim):
        for u in sim.chassis.left_units():
            for m in u.motors:
                m.set_voltage(-12.0)
        for u in sim.chassis.right_units():
            for m in u.motors:
                m.set_voltage(12.0)

    def test_a_wider_track_turns_more_slowly(self):
        rates = []
        for track in (9.0, 15.0):
            c = differential_drive(track_width_in=track, cartridge="green",
                                   motors_per_side=3, wheels_per_side=3)
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(3.0, lambda sim: self._spin(sim))
            rates.append(abs(s.state.r))
        self.assertGreater(rates[0], rates[1])

    def test_body_acceleration_excludes_rotating_frame_terms(self):
        c = Chassis(mass=5.0, wheel_units=[], bearing_drag=0.0,
                    yaw_drag=0.0, aero_drag=0.0)
        d = Dynamics(c)
        d.state.u = 2.0
        d.state.r = 3.0
        d.step(0.001)
        self.assertAlmostEqual(d.state.ax, 0.0, places=12)
        self.assertAlmostEqual(d.state.ay, 0.0, places=12)
        self.assertLess(d.state.v, 0.0)  # body axes rotate under inertial velocity

    def test_alternating_mecanum_rollers_strafe(self):
        layout = [
            (0.12, 0.12, +1.0, +8.0, "FL"),
            (0.12, -0.12, -1.0, -8.0, "FR"),
            (-0.12, 0.12, -1.0, -8.0, "RL"),
            (-0.12, -0.12, +1.0, +8.0, "RR"),
        ]
        units = [
            WheelUnit(wheel=wheels.mecanum(4.0, sign), x=x, y=y,
                      motors=[Motor(cartridge="green", name=name)], name=name)
            for x, y, sign, _, name in layout
        ]
        c = Chassis(mass=7.0, wheel_units=units, bearing_drag=0.2)
        s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
        commands = {name: volts for _, _, _, volts, name in layout}
        s.run(1.5, lambda sim: [m.set_voltage(commands[m.name]) for m in sim.motors])
        self.assertGreater(s.state.y, 0.15)
        self.assertLess(abs(s.state.x), 0.03)
        self.assertLess(abs(s.state.theta), 0.05)


class TestEnergy(unittest.TestCase):
    def test_kinetic_energy_never_exceeds_what_the_pack_supplied(self):
        s = Simulator(presets.six_motor_600())
        s.run(2.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        self.assertLess(s.log.max("kinetic_energy_J"), s.energy_battery)

    def test_losses_do_not_exceed_the_input(self):
        s = Simulator(presets.six_motor_200())
        s.run(3.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        e = s.energy_report()
        self.assertLessEqual(e["copper_J"], e["battery_J"] * 1.001)
        self.assertGreater(e["battery_J"], 0.0)

    def test_slip_costs_energy(self):
        low = differential_drive(cartridge="blue", wheel="omni_3.25",
                                 motors_per_side=3, wheels_per_side=3)
        low.surface_mu = 0.3
        high = differential_drive(cartridge="blue", wheel="omni_3.25",
                                  motors_per_side=3, wheels_per_side=3)
        out = []
        for c in (high, low):
            s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
            s.run(1.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
            out.append(s.energy_slip)
        self.assertGreater(out[1], out[0])


class TestNumerics(unittest.TestCase):
    def test_result_is_independent_of_timestep(self):
        finals = []
        for dt in (0.001, 0.0002):
            s = Simulator(presets.six_motor_600(), InfiniteBattery(),
                          SimConfig(dt=dt, log_every=0))
            s.run(2.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
            finals.append(s.state.x)
        self.assertAlmostEqual(finals[0], finals[1], delta=abs(finals[0]) * 0.01)

    def test_runs_are_deterministic(self):
        def run():
            r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
            r.drive_for(24)
            r.turn_for(90)
            return r.sim.state.x, r.sim.state.y, r.sim.state.theta
        self.assertEqual(run(), run())

    def test_divergence_is_reported_not_silent(self):
        c = presets.six_motor_200()
        c.bearing_drag = 1e6           # nonsense value on purpose
        s = Simulator(c, config=SimConfig(log_every=0))
        with self.assertRaises(RuntimeError):
            s.run(1.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])


class TestControl(unittest.TestCase):
    def test_pid_respects_its_output_limit(self):
        pid = PID(kp=1000.0, output_limit=12.0)
        self.assertLessEqual(pid.update(100.0, 0.0, 0.01), 12.0)

    def test_pid_integral_does_not_run_away(self):
        pid = PID(kp=0.0, ki=5.0, integral_limit=3.0, output_limit=12.0)
        for _ in range(10000):
            out = pid.update(1.0, 0.0, 0.01)
        self.assertLessEqual(abs(out), 3.0 + 1e-9)

    def test_pid_integral_zone(self):
        pid = PID(kp=0.0, ki=5.0, integral_zone=0.1, output_limit=12.0)
        pid.update(10.0, 0.0, 0.5)
        self.assertEqual(pid.integral, 0.0)

    def test_trapezoid_covers_the_distance(self):
        p = TrapezoidProfile(2.0, 1.0, 2.0)
        pos, vel, _ = p.sample(p.duration)
        self.assertAlmostEqual(pos, 2.0, places=6)
        self.assertAlmostEqual(vel, 0.0, places=6)

    def test_trapezoid_never_exceeds_its_limits(self):
        p = TrapezoidProfile(3.0, 1.2, 2.5)
        for i in range(500):
            _, vel, acc = p.sample(p.duration * i / 499.0)
            self.assertLessEqual(abs(vel), 1.2 + 1e-9)
            self.assertLessEqual(abs(acc), 2.5 + 1e-9)

    def test_short_moves_become_triangles(self):
        p = TrapezoidProfile(0.02, 2.0, 3.0)
        self.assertEqual(p.t_cruise, 0.0)
        self.assertLess(p.cruise_velocity, 2.0)
        self.assertAlmostEqual(p.sample(p.duration)[0], 0.02, places=6)

    def test_negative_distance_goes_backwards(self):
        p = TrapezoidProfile(-1.0, 1.0, 2.0)
        self.assertLess(p.sample(p.duration / 2.0)[1], 0.0)
        self.assertAlmostEqual(p.sample(p.duration)[0], -1.0, places=6)

    def test_feedforward_predicts_the_real_free_speed(self):
        c = presets.six_motor_600()
        ff = Feedforward.from_chassis(c)
        s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
        s.run(6.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        predicted = (12.0 - ff.ks) / ff.kv
        # Linear feedforward describes the DC equivalent below its firmware
        # limits. A V5 speed governor additionally bounds the equilibrium.
        governed = min(m.free_speed / u.gear_ratio * u.radius
                       for u in c.driven_units() for m in u.motors)
        predicted = min(predicted, governed)
        self.assertAlmostEqual(predicted, s.state.u, delta=s.state.u * 0.12)


class TestRoutines(unittest.TestCase):
    def test_drive_for_lands_close(self):
        r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
        self.assertTrue(r.drive_for(24, timeout=6.0))
        self.assertAlmostEqual(to_inches(r.sim.state.x), 24.0, delta=1.5)
        self.assertLess(abs(to_inches(r.sim.state.y)), 1.0)

    def test_turn_for_lands_close(self):
        r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
        self.assertTrue(r.turn_for(90, timeout=6.0))
        self.assertAlmostEqual(r.true_heading_deg, 90.0, delta=4.0)

    def test_encoders_over_read_when_the_wheels_slip(self):
        r = Robot(presets.speed_base(), config=SimConfig(log_every=0))
        r.tank(12.0, 12.0)
        r.sim.run(1.2)
        self.assertGreater(r.travelled(), r.sim.state.x)

    def test_profiled_drive_beats_the_pid_on_time(self):
        a = Robot(presets.six_motor_600(), config=SimConfig(log_every=0))
        a.drive_profiled(48, max_accel=3.5)
        b = Robot(presets.six_motor_600(), config=SimConfig(log_every=0))
        b.drive_for(48, timeout=6.0)
        self.assertLess(a.t, b.t)

    def test_a_square_comes_back_near_the_start(self):
        r = Robot(presets.six_motor_200(), config=SimConfig(log_every=0))
        for _ in range(4):
            r.drive_for(24, timeout=4.0)
            r.turn_for(90, timeout=4.0)
        self.assertLess(to_inches(math.hypot(r.sim.state.x, r.sim.state.y)), 6.0)

    def test_odometry_tracks_the_truth(self):
        r = Robot(presets.odom_bot(), config=SimConfig(log_every=0))
        r.drive_for(36, timeout=5.0)
        r.turn_for(90, timeout=4.0)
        r.drive_for(24, timeout=5.0)
        err = math.hypot(r.sim.odometry.x - r.sim.state.x,
                         r.sim.odometry.y - r.sim.state.y)
        self.assertLess(to_inches(err), 4.0)

    def test_log_has_every_channel(self):
        r = Robot(presets.six_motor_200())
        r.drive_for(12, timeout=3.0)
        for key in ("battery_V", "battery_A", "battery_W", "vel_fwd_mps",
                    "accel_fwd_mps2", "force_fwd_N", "copper_loss_W",
                    "slip_loss_W", "max_motor_temp_C", "odom_error_in",
                    "m_L1-1_A", "m_L1-1_Nm", "m_L1-1_W_in", "m_L1-1_tempC",
                    "w_L1_slip_ratio", "w_L1_Fz_N", "w_L1_Fx_N"):
            self.assertIn(key, r.log, key)
        self.assertGreater(r.log.n, 10)

    def test_push_pins_the_chassis_and_measures_contact_force(self):
        r = Robot(presets.push_bot(), config=SimConfig(log_every=0))
        force = r.push(0.75)
        self.assertGreater(force, 50.0)
        self.assertAlmostEqual(r.sim.state.x, 0.0, places=12)
        self.assertFalse(r.sim.dynamics.constrained)

    def test_v5_encoder_counts_match_each_cartridge(self):
        for cartridge, expected in (("red", 1800.0), ("green", 900.0),
                                    ("blue", 300.0)):
            c = differential_drive(cartridge=cartridge, motors_per_side=1,
                                   wheels_per_side=1)
            s = Simulator(c, config=SimConfig(log_every=0))
            self.assertTrue(all(e.ticks_per_rev == expected
                                for e in s.encoders.values()))

    def test_default_distance_sensor_uses_v5rc_field_size(self):
        s = Simulator(presets.four_motor_200(), config=SimConfig(log_every=0))
        self.assertAlmostEqual(s.distance.field_half, 140.5 * 0.0254 / 2.0,
                               places=12)


class TestChassisMath(unittest.TestCase):
    def test_summary_agrees_with_the_sim(self):
        c = presets.four_motor_200()
        s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
        s.run(6.0, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        self.assertAlmostEqual(c.theoretical_free_speed(), s.state.u,
                               delta=c.theoretical_free_speed() * 0.1)

    def test_external_gearing_changes_speed_and_torque(self):
        fast = differential_drive(cartridge="blue", external_ratio=1.0)
        slow = differential_drive(cartridge="blue", external_ratio=2.0)
        self.assertAlmostEqual(fast.theoretical_free_speed(),
                               2.0 * slow.theoretical_free_speed(), places=6)
        self.assertAlmostEqual(2.0 * fast.driven_units()[0].max_torque,
                               slow.driven_units()[0].max_torque, places=6)

    def test_more_motors_means_more_force(self):
        four = differential_drive(cartridge="red", motors_per_side=2, wheels_per_side=2)
        six = differential_drive(cartridge="red", motors_per_side=3, wheels_per_side=3)
        f4 = sum(u.max_torque / u.radius for u in four.driven_units())
        f6 = sum(u.max_torque / u.radius for u in six.driven_units())
        self.assertGreater(f6, f4)

    def test_reflected_inertia_dominates_on_a_high_reduction(self):
        red = differential_drive(cartridge="red").driven_units()[0]
        self.assertGreater(red.effective_inertia, red.wheel.inertia * 5.0)

    def test_a_reversed_motor_drives_backwards(self):
        c = differential_drive(cartridge="green")
        for m in c.motors:
            m.reversed = True
        s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
        s.run(1.5, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        self.assertLess(s.state.x, -0.05)


class TestBrainLimits(unittest.TestCase):
    def test_total_current_is_capped(self):
        c = differential_drive(cartridge="red", motors_per_side=4, wheels_per_side=2)
        s = Simulator(c, config=SimConfig(brain_current_limit=8.0, log_every=0))
        s.dynamics.locked = True
        s.run(0.5, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        self.assertLessEqual(s.battery.current, 8.0 + 0.1)

    def test_stalled_robot_pack_draw_matches_h_bridge_power(self):
        c = presets.six_motor_200()
        s = Simulator(c, config=SimConfig(log_every=0))
        s.dynamics.locked = True
        s.run(0.2, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        winding = sum(abs(m.current) for m in s.motors)
        expected = sum(m.power_electrical for m in s.motors) / s.battery.voltage
        self.assertAlmostEqual(s.battery.current, expected, delta=0.02)
        self.assertLess(s.battery.current, winding)

    def test_locked_robot_does_not_move(self):
        s = Simulator(presets.six_motor_200(), config=SimConfig(log_every=0))
        s.dynamics.locked = True
        s.run(0.5, lambda sim: [m.set_voltage(12.0) for m in sim.motors])
        self.assertEqual(s.state.x, 0.0)

    def test_velocity_integral_advances_once_per_firmware_period(self):
        c = differential_drive(motors_per_side=1, wheels_per_side=1)
        s = Simulator(c, InfiniteBattery(), SimConfig(log_every=0))
        for m in s.motors:
            m.set_velocity(1.0)
            m.receive()
        for _ in range(9):
            s._solve_electrical(0.001)
        for m in s.motors:
            self.assertEqual(m._vel_integral, 0.0)
        s._solve_electrical(0.001)
        for m in s.motors:
            self.assertAlmostEqual(m._vel_integral, 0.010, places=12)


class TestFastBase(unittest.TestCase):
    def test_450_rpm_base_gears_and_runs_where_it_should(self):
        c = presets.six_motor_450()
        u = c.driven_units()[0]
        self.assertAlmostEqual(u.gear_ratio, 48.0 / 36.0, places=6)
        self.assertEqual(len(c.motors), 6)
        # 600 rpm cartridge through 36:48 is 450 rpm at the wheel
        self.assertAlmostEqual(to_rpm(u.free_speed), 450.0, delta=1.0)
        # 450 rpm on a 3.25 in omni is 76.6 in/s
        self.assertAlmostEqual(to_inches(c.theoretical_free_speed()), 76.6, delta=0.5)

    def test_it_actually_gets_there(self):
        r = Robot(presets.six_motor_450(), config=SimConfig(log_every=0))
        r.arcade(12, 0)
        for _ in range(3000):
            r.sim.step()
        self.assertGreater(to_inches(r.sim.state.u), 70.0)
        self.assertLess(to_inches(r.sim.state.u), 77.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

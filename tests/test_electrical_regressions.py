"""Physics regressions independent of the historical fitted constants."""

import math
import unittest

from vexsim import Battery, InfiniteBattery, Motor, Simulator, presets
from vexsim.motor import BrakeMode, cortex_393, v5_11w, v5_5_5w
from vexsim.sim import SimConfig


class ElectricalRegressions(unittest.TestCase):
    def test_motor_energy_conserves_in_both_directions(self):
        for factory, cartridge in ((v5_11w, "green"), (v5_5_5w, "green"),
                                   (cortex_393, "393_torque")):
            for speed in (-0.5, 0.5):
                for voltage in (-12.0, 0.0, 12.0):
                    motor = Motor(spec=factory(), cartridge=cartridge)
                    motor.velocity = speed * motor.free_speed
                    motor.set_voltage(voltage)
                    draw = motor.electrical(12.0, 0.001)
                    motor.output_torque()
                    self.assertGreaterEqual(motor.power_friction, 0.0)
                    self.assertAlmostEqual(
                        12.0 * draw, motor.power_mechanical + motor.power_copper
                        + motor.power_friction, places=10)

    def test_current_limited_brake_returns_regeneration_to_bus(self):
        motor = Motor()
        motor.velocity = motor.free_speed
        motor.stop(BrakeMode.BRAKE)
        draw = motor.electrical(12.0, 0.001)
        self.assertLess(draw, 0.0)
        self.assertAlmostEqual(draw * 12, motor.power_electrical, places=12)
        self.assertLessEqual(abs(motor.applied_voltage), 12.0)

    def test_forced_overspeed_reports_unavoidable_regenerative_overcurrent(self):
        motor = Motor()
        motor.velocity = motor.free_speed * 3
        motor.set_voltage(0.0)
        motor.electrical(12.0, 0.001)
        self.assertLessEqual(abs(motor.applied_voltage), 12.0)
        self.assertTrue(motor.regenerative_overcurrent)
        self.assertGreater(abs(motor.current), motor.current_limit)
        emf = motor.spec.ke * motor.velocity * motor.ratio
        self.assertAlmostEqual(motor.applied_voltage,
                               emf + motor.current * motor.spec.resistance, places=12)

    def test_small_motor_gearing_is_fixed_even_with_cartridge_label(self):
        values = []
        for cartridge in ("red", "green", "blue", "direct"):
            motor = Motor(spec=v5_5_5w(), cartridge=cartridge)
            motor.set_voltage(12.0)
            motor.electrical(12.0, 0.001)
            values.append((motor.ratio, motor.free_speed, motor.output_torque()))
        self.assertTrue(all(value == values[0] for value in values))
        self.assertAlmostEqual(values[0][1] * 30 / math.pi, 200.0)
        self.assertAlmostEqual(values[0][2], 0.5)

    def test_full_battery_accounts_for_rejected_regeneration(self):
        battery = Battery(soc=1.0)
        for _ in range(10):
            battery.update(-2.0, 0.1)
            self.assertEqual(battery.soc, 1.0)
            self.assertEqual(battery.power_chemical, 0.0)
            self.assertGreater(battery.power_rejected, 0.0)
            self.assertAlmostEqual(
                battery.power_chemical,
                battery.power_out + battery.power_loss + battery.power_rejected,
                places=10)
        self.assertAlmostEqual(battery.energy_rejected, 28.8, places=10)
        self.assertEqual(battery.charge_used_ah, 0.0)

    def test_partial_last_discharge_cannot_exceed_charge(self):
        battery = Battery(soc=1e-7)
        available = battery.soc * battery.capacity_ah
        battery.update(20.0, 1.0)
        self.assertAlmostEqual(battery.charge_used_ah, available, places=14)
        self.assertEqual(battery.soc, 0.0)
        battery.update(20.0, 1.0)
        self.assertEqual(battery.current, 0.0)
        self.assertEqual(battery.power_out, 0.0)
        self.assertTrue(battery.brownout)

    def test_simulator_budget_and_nearly_empty_pack_conserve_power(self):
        for limit in (0.0, 1.0, 4.0, 8.0):
            sim = Simulator(presets.six_motor_200(),
                            config=SimConfig(brain_current_limit=limit, log_every=0))
            for motor in sim.motors:
                motor.set_voltage(12.0)
            sim._solve_electrical(0.001)
            self.assertLessEqual(sim.battery.current, limit + 1e-9)
            self.assertAlmostEqual(sim.battery.power_out,
                                   sum(m.power_electrical for m in sim.motors), places=8)
        sim = Simulator(presets.four_motor_200(), Battery(soc=1e-9),
                        SimConfig(log_every=0))
        available = sim.battery.soc * sim.battery.capacity_ah
        for motor in sim.motors:
            motor.set_voltage(12.0)
        sim._solve_electrical(0.001)
        self.assertLessEqual(sim.battery.charge_used_ah, available + 1e-15)
        self.assertAlmostEqual(sim.battery.power_out,
                               sum(m.power_electrical for m in sim.motors), places=8)

    def test_pid_clock_and_reset(self):
        sim = Simulator(presets.four_motor_200(), InfiniteBattery(),
                        SimConfig(log_every=0))
        motor = sim.motors[0]
        motor.set_velocity(10.0)
        motor.receive()
        for _ in range(9):
            sim._solve_electrical(0.001)
        self.assertEqual(motor._vel_integral, 0.0)
        sim._solve_electrical(0.001)
        self.assertAlmostEqual(motor._vel_integral, 0.1)
        motor.reset()
        self.assertEqual(motor._controller_elapsed, 0.0)
        self.assertEqual(motor.applied_voltage, 0.0)
        self.assertEqual(motor.power_electrical, 0.0)
        self.assertEqual(motor._scale, 1.0)

    def test_stop_cancels_previous_drive_before_next_pid_tick(self):
        for mode in (BrakeMode.COAST, BrakeMode.BRAKE, BrakeMode.HOLD):
            with self.subTest(mode=mode):
                motor = Motor()
                motor.set_voltage(12.0)
                motor.electrical(12.0, 0.001)
                self.assertGreater(motor.current, 0.0)
                motor.stop(mode)
                self.assertEqual(motor.commanded_voltage, 0.0)
                motor.electrical(12.0, 0.001)
                self.assertEqual(motor.current, 0.0)
                self.assertEqual(motor.commanded_voltage, 0.0)
                self.assertAlmostEqual(motor._controller_elapsed, 0.002)


if __name__ == "__main__":
    unittest.main()

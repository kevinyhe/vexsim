"""The brain sends motor commands every 5 ms (PROS DATA_WRITE_RATE), so a
command waits for the next write before the motor acts on it."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import InfiniteBattery, Simulator, presets
from vexsim.motor import Motor, MotorMode
from vexsim.sim import SimConfig


def simulator():
    return Simulator(presets.four_motor_450(), InfiniteBattery(), SimConfig(log_every=0))


class TestMotorWrites(unittest.TestCase):
    def test_a_command_reaches_the_motor_at_the_brains_next_write(self):
        sim = simulator()
        motor = sim.motors[0]
        motor.set_voltage(12.0)
        sim.step()
        sim.step()
        self.assertIs(motor.mode, MotorMode.COAST)
        self.assertEqual(motor.current, 0.0)
        sim.step()
        self.assertIs(motor.mode, MotorMode.VOLTAGE)
        self.assertGreater(motor.current, 0.0)

    def test_the_brain_writes_every_5_ms_and_sends_the_latest_command(self):
        sim = simulator()
        motor = sim.motors[0]
        for _ in range(3):
            sim.step()
        motor.set_voltage(6.0)
        motor.set_voltage(-6.0)
        for _ in range(4):
            sim.step()
        self.assertIs(motor.mode, MotorMode.COAST)
        sim.step()
        self.assertEqual((motor.mode, motor.command), (MotorMode.VOLTAGE, -6.0))

    def test_a_10_ms_loop_on_the_tick_waits_2_5_ms_for_every_command(self):
        sim = simulator()
        motor = sim.motors[0]
        waits = []
        for tick in range(8):
            volts = 12.0 if tick % 2 == 0 else -12.0
            for each in sim.motors:
                each.set_voltage(volts)
            sent = sim.t
            arrived = None
            for _ in range(20):
                start = sim.t
                sim.step(0.0005)
                if arrived is None and motor.command == volts:
                    arrived = start
            waits.append(round((arrived - sent) * 1000, 6))
        self.assertEqual(waits, [2.5] * 8)

    def test_a_stop_also_waits_for_the_write(self):
        sim = simulator()
        motor = sim.motors[0]
        motor.set_voltage(12.0)
        for _ in range(10):
            sim.step()
        motor.stop()
        sim.step()
        self.assertIs(motor.mode, MotorMode.VOLTAGE)
        self.assertGreater(motor.current, 0.0)
        for _ in range(5):
            sim.step()
        self.assertIs(motor.mode, MotorMode.COAST)

    def test_a_mechanism_motor_on_the_battery_waits_too(self):
        sim = simulator()
        lift = Motor(name="lift")
        sim.power_mechanisms([lift])
        lift.set_voltage(12.0)
        sim.step()
        self.assertIs(lift.mode, MotorMode.COAST)
        for _ in range(2):
            sim.step()
        self.assertIs(lift.mode, MotorMode.VOLTAGE)

    def test_a_motor_on_its_own_takes_a_command_at_once(self):
        motor = Motor()
        motor.set_voltage(12.0)
        self.assertEqual((motor.mode, motor.command), (MotorMode.VOLTAGE, 12.0))


if __name__ == "__main__":
    unittest.main()

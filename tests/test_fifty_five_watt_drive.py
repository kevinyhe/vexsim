"""A drive with 11 W and 5.5 W motors: Override's 55 W allowance of four
11 W motors and two 5.5 W motors, every wheel geared to 450 rpm."""

import math
import unittest

from vexsim import Simulator, presets
from vexsim.sim import SimConfig
from vexsim.units import INCH


def straight_run(chassis, volts_11w, volts_5_5w, seconds):
    """Speed (in/s) after `seconds` of fixed voltages from a stop."""
    sim = Simulator(chassis, config=SimConfig(log_every=0))
    for _ in range(round(seconds * 1000)):
        for motor in chassis.motors:
            motor.set_voltage(volts_5_5w if motor.spec.name == "V5 5.5W" else volts_11w)
        sim.step(0.001)
    return sim.state.u / INCH


class TestFiftyFiveWattDrive(unittest.TestCase):
    def test_four_11w_and_two_5_5w_motors_make_55_watts(self):
        chassis = presets.six_motor_55w()
        self.assertEqual([[motor.spec.name for motor in unit.motors] for unit in chassis.left_units()],
                         [["V5 11W"], ["V5 5.5W"], ["V5 11W"]])
        self.assertEqual(sum(motor.spec.peak_power_rating for motor in chassis.motors), 55.0)

    def test_every_wheel_tops_out_at_450_rpm(self):
        for unit in presets.six_motor_55w().wheel_units:
            self.assertAlmostEqual(unit.free_speed * 60 / (2 * math.pi), 450.0, places=6)

    def test_it_reaches_the_six_11w_drives_top_speed_more_slowly(self):
        mixed = presets.six_motor_55w()
        mixed_early, mixed_top = straight_run(mixed, 12.0, 12.0, 0.3), straight_run(mixed, 12.0, 12.0, 2.0)
        full = presets.six_motor_450()
        full_early, full_top = straight_run(full, 12.0, 12.0, 0.3), straight_run(full, 12.0, 12.0, 2.0)
        self.assertAlmostEqual(mixed_top, full_top, delta=0.03 * full_top)
        self.assertLess(mixed_early, full_early)


if __name__ == "__main__":
    unittest.main()

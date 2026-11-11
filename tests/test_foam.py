"""Wheels on the foam tiles: the measured grip, the dent, and what the dent costs."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import InfiniteBattery, Simulator, foam, presets, wheels
from vexsim.sim import SimConfig
from vexsim.units import G, INCH, LB

ROBOT_15_LB_ON_SIX_WHEELS = 15.0 * LB * G / 6.0


class TestGripOnFoam(unittest.TestCase):
    def test_the_wheels_grip_what_the_forum_measured(self):
        self.assertEqual(wheels.omni(4.0).tire.mu_long_static, 0.90)
        self.assertEqual(wheels.omni(2.75).tire.mu_long_static, 0.84)
        self.assertEqual(wheels.traction(4.0).tire.mu_long_static, 0.81)
        self.assertEqual(wheels.mecanum(4.0).tire.mu_long_static, 0.85)

    def test_an_omni_grips_foam_better_than_a_traction_wheel(self):
        for size in (2.75, 3.25, 4.0):
            self.assertGreater(wheels.omni(size).tire.mu_long_static,
                               wheels.traction(size).tire.mu_long_static)

    def test_a_tile_weighs_what_vex_lists(self):
        self.assertAlmostEqual(foam.TILE_DENSITY, 84.0, delta=1.0)


    def test_an_omni_slides_sideways_with_the_carpet_figure_above_what_the_foam_alone_gives(self):
        tire = wheels.omni(3.25).tire
        low, high = foam.OMNI_SIDEWAYS_CARPET
        self.assertTrue(low <= tire.mu_lat_slide < tire.mu_lat_static <= high)
        quarter_of_the_robot = presets.four_motor_450().mass * G / 4.0
        rollers = [foam.rolling_resistance(quarter_of_the_robot, radius * INCH, width * INCH)
                   for radius in (0.2, 0.3) for width in (0.2, 0.4)]
        self.assertAlmostEqual(min(rollers), 0.046, places=3)
        self.assertAlmostEqual(max(rollers), 0.076, places=3)
        self.assertLess(max(rollers), tire.mu_lat_slide)


class TestTheDent(unittest.TestCase):
    def test_a_robot_sinks_less_than_a_millimetre(self):
        wheel = wheels.omni(3.25)
        depth = foam.sink(ROBOT_15_LB_ON_SIX_WHEELS, wheel.radius, wheel.tire.contact_width)
        self.assertGreater(depth, 0.0003)
        self.assertLess(depth, 0.001)
        self.assertLess(depth / foam.TILE_THICKNESS, 0.10)

    def test_a_heavier_robot_sinks_further_and_rolls_harder(self):
        tire = wheels.omni(3.25).tire
        radius = wheels.omni(3.25).radius
        light, heavy = 0.5 * ROBOT_15_LB_ON_SIX_WHEELS, 2.0 * ROBOT_15_LB_ON_SIX_WHEELS
        self.assertGreater(foam.sink(heavy, radius, tire.contact_width),
                           foam.sink(light, radius, tire.contact_width))
        self.assertGreater(tire.rolling_resistance(heavy, radius), tire.rolling_resistance(light, radius))

    def test_a_bigger_wheel_rolls_easier(self):
        small, big = wheels.omni(2.75), wheels.omni(4.0)
        self.assertGreater(small.tire.rolling_resistance(ROBOT_15_LB_ON_SIX_WHEELS, small.radius),
                           big.tire.rolling_resistance(ROBOT_15_LB_ON_SIX_WHEELS, big.radius))

    def test_rolling_resistance_is_about_one_percent(self):
        wheel = wheels.omni(3.25)
        crr = wheel.tire.rolling_resistance(ROBOT_15_LB_ON_SIX_WHEELS, wheel.radius)
        self.assertGreater(crr, 0.008)
        self.assertLess(crr, 0.02)

    def test_a_wheel_in_the_air_does_not_sink(self):
        self.assertEqual(foam.sink(0.0, 0.04, 0.01), 0.0)

    def test_a_traction_wheel_scrubs_more_than_an_omni_turning_in_place(self):
        omni, traction = wheels.omni(3.25), wheels.traction(3.25)
        self.assertGreater(traction.tire.scrub_arm(ROBOT_15_LB_ON_SIX_WHEELS, traction.radius),
                           2.0 * omni.tire.scrub_arm(ROBOT_15_LB_ON_SIX_WHEELS, omni.radius))


class TestDrivingOnFoam(unittest.TestCase):
    def test_a_fast_base_spins_its_omnis_off_the_line(self):
        base = presets.six_motor_450()
        self.assertTrue(base.is_traction_limited())
        sim = Simulator(base, InfiniteBattery(), SimConfig())
        sim.run(0.4, lambda s: [m.set_voltage(12.0) for m in s.motors])
        self.assertLessEqual(sim.log.max("accel_fwd_mps2"), 0.87 * G * 1.05)


if __name__ == "__main__":
    unittest.main()

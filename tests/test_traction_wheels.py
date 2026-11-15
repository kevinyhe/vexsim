"""Traction wheels in place of omnis: they grip sideways, so the robot
slides less in a curve and turns more slowly."""

import unittest

from vexsim import Simulator, cascade, presets
from vexsim.chassis import with_traction_wheels
from vexsim.override import OverrideWorld
from vexsim.sim import SimConfig
from vexsim.units import INCH


def six_wheels(where):
    chassis = presets.six_motor_450()
    return chassis if where == "none" else with_traction_wheels(chassis, where)


def sideways_slide(chassis, in_world):
    """Mean sideways speed (in/s) and yaw rate (rad/s) over the last half
    second of 1.2 s with the left side at 12 V and the right at 7 V."""
    sim = Simulator(chassis, config=SimConfig(log_every=0))
    sim.set_pose(-40 * INCH, -12 * INCH, 0.0)
    step = OverrideWorld(sim, robot=cascade.CascadeRobot(), objects=False).step if in_world else sim.step
    left = [m for u in chassis.left_units() for m in u.motors]
    right = [m for u in chassis.right_units() for m in u.motors]
    slide = []
    for k in range(1200):
        for motor in left:
            motor.set_voltage(12.0)
        for motor in right:
            motor.set_voltage(7.0)
        step(0.001)
        if k >= 700:
            slide.append(abs(sim.state.v))
    return sum(slide) / len(slide) / INCH, abs(sim.state.r)


class TestTractionWheels(unittest.TestCase):
    def test_middle_and_all_swap_the_driven_wheels_only(self):
        chassis = with_traction_wheels(presets.override_cascade(), "all")
        self.assertEqual({u.name: u.wheel.name for u in chassis.wheel_units},
                         {"L1": "2.75in traction", "L2": "2.75in traction", "R1": "2.75in traction",
                          "R2": "2.75in traction", "track_perp": "2.0in tracking"})
        chassis = six_wheels("middle")
        self.assertEqual([u.wheel.name for u in chassis.left_units()],
                         ["3.25in omni", "3.25in traction", "3.25in omni"])

    def test_middle_traction_needs_three_driven_wheels_a_side(self):
        with self.assertRaisesRegex(ValueError, "6-wheel drive"):
            with_traction_wheels(presets.four_motor_450(), "middle")
        with self.assertRaises(ValueError):
            with_traction_wheels(presets.six_motor_450(), "front")

    def test_traction_wheels_slide_less_and_turn_slower_in_a_curve(self):
        omni, middle, every = (sideways_slide(six_wheels(where), False) for where in ("none", "middle", "all"))
        self.assertGreater(omni[0], 4.0 * middle[0])
        self.assertGreater(middle[0], every[0])
        self.assertGreater(omni[1], middle[1])
        self.assertGreater(middle[1], every[1])

    def test_the_mujoco_drivetrain_grips_sideways_with_middle_traction_wheels(self):
        omni, middle = (sideways_slide(six_wheels(where), True) for where in ("none", "middle"))
        self.assertGreater(omni[0], 4.0 * middle[0])


if __name__ == "__main__":
    unittest.main()

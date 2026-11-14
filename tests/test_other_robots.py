"""The partner's and opponents' robots: drivetrains with a box for a frame
that hold still braking or drive a path, and that the robot, the pieces and
the field hit."""

import math
import unittest

from vexsim import Simulator, cascade, presets
from vexsim.other_robots import OtherRobot
from vexsim.override import OverrideWorld, StartOverlapError
from vexsim.sim import SimConfig
from vexsim.units import INCH

START = (-40.0 * INCH, -12.0 * INCH, 0.0)


def world_with(others, start=START):
    sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
    sim.set_pose(*start)
    return sim, OverrideWorld(sim, robot=cascade.CascadeRobot(), objects=False, others=others)


def drive_into(other_heading, seconds=1.5, volts=8.0):
    """The robot at START driving forward into an opponent 28 in ahead of it."""
    sim, world = world_with([OtherRobot("opponent 1", -12.0 * INCH, -12.0 * INCH, other_heading)])
    for _ in range(round(seconds * 1000)):
        for motor in sim.chassis.motors:
            motor.set_voltage(volts)
        world.step(0.001)
    return world, world.others.reports()[0]


class TestPushingAnotherRobot(unittest.TestCase):
    def test_a_braking_robot_hit_from_behind_is_pushed_and_the_hit_is_logged_under_its_name(self):
        world, report = drive_into(0.0)
        self.assertIn("the opponent 1 robot", [hit.element for hit in world.reported_contacts()])
        self.assertGreater(report["moved"], 5.0 * INCH)
        self.assertLess(abs(math.degrees(report["theta"])), 10.0)

    def test_hit_from_the_side_it_rolls_farther_on_its_omnis(self):
        _, behind = drive_into(0.0)
        _, side = drive_into(math.pi / 2.0)
        self.assertGreater(side["moved"], behind["moved"] + 3.0 * INCH)


class TestDrivingAPath(unittest.TestCase):
    def test_it_holds_still_while_the_world_settles_before_the_match(self):
        path = ((-38.0 * INCH, 6.0 * INCH),)
        _, world = world_with([OtherRobot("partner", -58.0 * INCH, 6.0 * INCH, 0.0, path=path)])
        self.assertLess(world.others.reports()[0]["moved"], 0.1 * INCH)
        for _ in range(300):
            world.step(0.001)
        self.assertGreater(world.others.reports()[0]["moved"], 1.0 * INCH)

    def test_it_waits_then_drives_its_path_and_stops_at_the_end(self):
        path = ((-38.0 * INCH, 6.0 * INCH), (-26.0 * INCH, 14.0 * INCH))
        partner = OtherRobot("partner", -58.0 * INCH, 6.0 * INCH, 0.0, path=path, volts=8.0, start_s=0.5)
        _, world = world_with([partner])
        for _ in range(400):
            world.step(0.001)
        self.assertLess(world.others.reports()[0]["moved"], 0.1 * INCH)
        for _ in range(2100):
            world.step(0.001)
        report = world.others.reports()[0]
        self.assertTrue(report["finished"])
        self.assertLess(report["from_end"], 7.0 * INCH)
        frame = world.telemetry()["others"]["partner"]
        self.assertAlmostEqual(frame["x"], report["x"], places=3)


class TestPlacing(unittest.TestCase):
    def test_a_robot_put_inside_a_goal_is_refused_by_name(self):
        with self.assertRaises(StartOverlapError) as caught:
            world_with([OtherRobot("partner", -30.0 * INCH, 35.0 * INCH, 0.0)])
        self.assertEqual(caught.exception.robot, "the partner robot")
        self.assertIn("Goal", caught.exception.overlaps[0].name)

    def test_the_robot_put_inside_another_robot_is_refused(self):
        with self.assertRaises(StartOverlapError) as caught:
            world_with([OtherRobot("opponent 2", -48.0 * INCH, -12.0 * INCH, 0.0)])
        self.assertEqual(caught.exception.robot, "the robot")
        self.assertEqual(caught.exception.overlaps[0].name, "the opponent 2 robot")

    def test_other_robots_count_as_robots_for_the_toggles(self):
        _, world = world_with([OtherRobot("partner", -58.0 * INCH, 6.0 * INCH, 0.0)])
        frame = set(world.others.drives[0].frame_geoms)
        self.assertTrue(frame)
        self.assertTrue(frame <= world.pieces.every_robot_geom)

    def test_roles_sizes_masses_and_speeds_are_checked(self):
        for bad in (dict(role="referee"), dict(length=30.0 * INCH), dict(height=60.0 * INCH), dict(mass_lb=1.0),
                    dict(volts=13.0), dict(start_s=-1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                OtherRobot(**dict(dict(role="partner", x=0.0, y=0.0, theta=0.0), **bad))
        with self.assertRaisesRegex(ValueError, "its own role"):
            world_with([OtherRobot("partner", -58.0 * INCH, 6.0 * INCH, 0.0),
                        OtherRobot("partner", -58.0 * INCH, 40.0 * INCH, 0.0)])


if __name__ == "__main__":
    unittest.main()

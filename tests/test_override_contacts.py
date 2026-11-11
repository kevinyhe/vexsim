"""The robot against Override field elements."""

import math
import unittest

from vexsim import Simulator, arena, cascade, presets
from vexsim import override_field as of
from vexsim.override import OverrideWorld, StartOverlapError
from vexsim.pieces import WALLS
from vexsim.sim import SimConfig
from vexsim.units import INCH


def robot(x_in, y_in, heading_deg):
    sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
    sim.set_pose(x_in * INCH, y_in * INCH, math.radians(heading_deg))
    return sim


def override(sim, **options):
    return OverrideWorld(sim, robot=cascade.CascadeRobot(), **options)


def drive(world, sim, volts, seconds):
    for motor in sim.motors:
        motor.set_voltage(volts)
    for _ in range(round(seconds * 1000)):
        world.step(0.001)


def wheel_surface_speed(sim):
    units = sim.chassis.wheel_units
    return sum(abs(u.omega) * u.radius for u in units if u.motors) / sum(1 for u in units if u.motors)


class TestWall(unittest.TestCase):
    def test_wall_stops_the_robot_and_the_motors_stall(self):
        cruising = robot(0.0, 0.0, 180.0)
        floor = arena.open_floor(cruising)
        drive(floor, cruising, 12.0, 1.5)
        cruising_current = floor.drive_current()
        sim = robot(-40.0, 40.0, 180.0)
        world = override(sim, objects=False)
        drive(world, sim, 12.0, 2.5)
        front = sim.state.x - sim.chassis.length / 2.0
        self.assertGreater(front, -of.HALF - 0.01 * INCH)
        self.assertLess(abs(sim.state.u), 0.05)
        self.assertGreater(wheel_surface_speed(sim), 0.25)
        self.assertGreater(world.log.drive_current(), 2.0 * cruising_current)
        walls = [c for c in world.reported_contacts() if c.element in WALLS]
        self.assertEqual({c.element for c in walls}, {"the red Alliance Station wall"})
        self.assertGreater(max(c.peak_force for c in walls), 20.0)
        self.assertTrue(world.touching_perimeter)

    def test_open_floor_has_no_wall(self):
        sim = robot(-40.0, 30.0, 180.0)
        world = arena.open_floor(sim)
        drive(world, sim, 12.0, 2.0)
        self.assertLess(sim.state.x - sim.chassis.length / 2.0, -of.HALF - 0.5)
        self.assertEqual(world.contacts, [])

    def test_glancing_wall_contact_slides_along_it(self):
        sim = robot(-55.0, -10.0, 160.0)
        world = override(sim, objects=False)
        drive(world, sim, 8.0, 2.5)
        self.assertGreater(sim.state.y, -10.0 * INCH + 0.05)


class TestGoals(unittest.TestCase):
    def test_backing_into_an_alliance_goal_stops_the_robot_at_its_base(self):
        goal = of.goal("red left")
        sim = robot(goal.x / INCH, goal.y / INCH + 20.0, 90.0)
        world = override(sim, objects=False)
        drive(world, sim, -6.0, 2.0)
        hits = [c for c in world.reported_contacts() if c.element == "the red left Goal"]
        self.assertTrue(hits)
        self.assertLess(abs(sim.state.u), 0.05)
        gap = sim.state.y - sim.chassis.length / 2.0 - goal.y
        self.assertGreater(gap, of.GOAL_BASE_ACROSS_FLATS / 2.0 - 0.01 * INCH)


class TestStartOverlaps(unittest.TestCase):
    def overlaps(self, x_in, y_in, heading_deg):
        with self.assertRaises(StartOverlapError) as raised:
            override(robot(x_in, y_in, heading_deg))
        return raised.exception.overlaps

    def test_a_robot_backed_into_the_wall_is_inside_it(self):
        [wall] = self.overlaps(-63.0, -36.0, 0.0)
        self.assertEqual(wall.name, "the red Alliance Station wall")
        self.assertAlmostEqual(wall.depth / INCH, 63.0 + 7.5 - of.HALF / INCH, delta=0.01)

    def test_a_robot_on_a_stack_names_the_cup_and_the_pin(self):
        x, y = of.from_corner(46.66, 70.20)
        cup, pin = self.overlaps(x / INCH, y / INCH, 0.0)
        self.assertEqual((cup.name, pin.name), ("a Cup", "a blue/red Pin"))
        self.assertGreater(cup.depth, pin.depth)

    def test_backed_onto_the_wall_at_a_toggle_its_roller_flaps_are_inside_the_bar(self):
        [toggle] = self.overlaps(-of.HALF / INCH + 7.5 + 0.01, -7.0, 0.0)
        self.assertEqual(toggle.name, "the left Toggle")

    def test_a_robot_against_the_wall_and_clear_of_everything_starts_with_its_preload_in_the_claw(self):
        world = override(robot(-of.HALF / INCH + 7.5 + 0.01, -36.0, 0.0), preload="yellow")
        self.assertIs(world.grip.piece, world.preload)


if __name__ == "__main__":
    unittest.main()

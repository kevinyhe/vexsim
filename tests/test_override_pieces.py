"""Pins and Cups as rigid bodies with the CAD's shapes, and the cascade robot among them."""

import math
import unittest

import mujoco
import numpy as np

from vexsim import Simulator, presets
from vexsim import override_cad as cad
from vexsim.cascade import CascadeRobot
from vexsim import override_field as of
from vexsim import pieces as pc
from vexsim.override import OverrideWorld
from vexsim.sim import SimConfig
from vexsim.units import G, INCH

#: A clear spot on the field, away from the Goals, Loaders and walls
OPEN_FLOOR = (-35.0 * INCH, 0.0)


def run_pieces(world, seconds, each_step=None):
    for _ in range(round(seconds / pc.DT)):
        world.step(pc.DT)
        if each_step is not None:
            each_step()


def robot_world(x_in, y_in, heading_deg, **options):
    sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
    sim.set_pose(x_in * INCH, y_in * INCH, math.radians(heading_deg))
    return sim, OverrideWorld(sim, robot=CascadeRobot(), **options)


def run(world, seconds):
    for _ in range(round(seconds * 1000)):
        world.step(0.001)


def drive(world, volts, seconds):
    for motor in world.sim.motors:
        motor.set_voltage(volts)
    run(world, seconds)
    for motor in world.sim.motors:
        motor.set_voltage(0.0)


def tilt_deg(piece):
    return math.degrees(math.acos(min(1.0, abs(piece.axis()[2]))))


def lateral(a, b):
    return math.hypot(a.center()[0] - b.center()[0], a.center()[1] - b.center()[1])


def standing_cup(world, clear_up=True, at=OPEN_FLOOR):
    return world.add_cup((at[0], at[1], world.standing_height("cup")), of.UP if clear_up else of.DOWN)


def energy(world):
    """Kinetic plus gravitational energy of every piece on the field, joules."""
    total = 0.0
    for piece in world.pieces():
        v = world.data.qvel[world.qvel_at[piece.body]:][:6]
        axial, across = pc.kinds()[piece.kind].shape.gyration
        spin = piece.mass * (across * (v[3] ** 2 + v[4] ** 2) + axial * v[5] ** 2)
        total += 0.5 * piece.mass * float(v[:3] @ v[:3]) + 0.5 * spin + piece.mass * G * piece.center()[2]
    return total


def hardest_press(world):
    """The largest push across any contact, newtons.  MuJoCo's contacts give a
    little, and energy held in that give is not in `energy`; while no contact
    pushes harder than a Cup's weight, that held energy is under 6e-6 J."""
    force = np.zeros(6)
    hardest = 0.0
    for i in range(world.data.ncon):
        mujoco.mj_contactForce(world.model, world.data, i, force)
        hardest = max(hardest, force[0])
    return hardest


def tipped(axis_deg, toward=(1.0, 0.0)):
    """An axis tipped `axis_deg` from straight up toward `toward`."""
    a = math.radians(axis_deg)
    return (math.sin(a) * toward[0], math.sin(a) * toward[1], math.cos(a))


class TestShapes(unittest.TestCase):
    def test_masses_are_the_weighed_ones_and_inertia_the_cad_solids_at_one_density(self):
        for kind, shape, grams, density in (("pin", cad.pin(), 72.0, 1.06), ("cup", cad.cup(), 77.0, 1.18)):
            with self.subTest(kind=kind):
                piece = pc.kinds()[kind]
                self.assertAlmostEqual(piece.mass, grams / 1000.0)
                self.assertAlmostEqual(piece.mass / shape.volume / 1000.0, density, delta=0.005)
                axial, across = shape.gyration
                self.assertEqual(piece.inertia, (piece.mass * across, piece.mass * across, piece.mass * axial))
                self.assertGreater(axial, 0.0)
                self.assertGreater(across, axial)

    def test_the_pieces_are_the_lengths_the_manual_gives(self):
        self.assertAlmostEqual(2.0 * cad.pin().half_length / INCH, of.PIN_LENGTH / INCH, delta=0.02)
        self.assertAlmostEqual(2.0 * cad.cup().half_length / INCH, of.CUP_LENGTH / INCH, delta=0.02)

    def test_a_cups_hollow_is_open_from_rim_to_rim(self):
        self.assertGreater(float(np.min(cad.cup_column().cavity)), 0.5)


class TestPinInCup(unittest.TestCase):
    def test_a_pin_dropped_into_a_cup_settles_upright_and_centred(self):
        world = pc.Pieces()
        cup = standing_cup(world)
        pin = world.add_pin(("red", "yellow"), (OPEN_FLOOR[0] + 0.15 * INCH, OPEN_FLOOR[1],
                                                cup.center()[2] + cad.seats().pin_in_cup + 3.0 * INCH), tipped(4.0))
        run_pieces(world, 3.5)
        self.assertTrue(pin.asleep and cup.asleep)
        self.assertLess(tilt_deg(pin), 0.5)
        self.assertLess(lateral(pin, cup) / INCH, 0.02)
        self.assertAlmostEqual((pin.center()[2] - cup.center()[2]) / INCH, cad.seats().pin_in_cup / INCH, delta=0.02)
        self.assertIs(pin.below, cup)

    def test_a_pin_started_tilted_in_a_cup_leans_on_the_inner_wall(self):
        world = pc.Pieces()
        cup = standing_cup(world)
        limit = lean_limit_deg()
        start = tipped(limit + 6.0)
        top = cup.center()[2] + cup.half_length
        pin = world.add_pin(("red", "yellow"), (OPEN_FLOOR[0], OPEN_FLOOR[1], top + 1.5 * INCH), start)
        run_pieces(world, 4.5)
        self.assertTrue(pin.asleep)
        self.assertIs(pin.below, cup)
        self.assertAlmostEqual(tilt_deg(pin), limit, delta=0.5)

    def test_a_cup_holding_a_pin_comes_back_short_of_its_edge_and_falls_past_it(self):
        edge = tipping_angle_deg()
        for past, falls in ((-3.0, False), (3.0, True)):
            with self.subTest(past=past):
                world = pc.Pieces()
                cup, pin = tipped_stack(world, edge + past)
                pivot = rim_edge(cup)
                run_pieces(world, 6.0)
                if falls:
                    self.assertGreater(tilt_deg(cup), 60.0)
                else:
                    self.assertLess(tilt_deg(cup), 0.5)
                    self.assertIs(pin.below, cup)
                self.assertLess(math.dist(pivot[:2], rim_point_nearest(cup, pivot)[:2]) / INCH, 0.25)


class TestOnTheFoam(unittest.TestCase):
    def test_a_dropped_pin_comes_to_rest_lying_on_its_flange_and_a_tip(self):
        world = pc.Pieces()
        pin = world.add_pin(("red", "yellow"), (OPEN_FLOOR[0], OPEN_FLOOR[1], 20.0 * INCH), (0.3, 0.1, 1.0))
        run_pieces(world, 4.0)
        self.assertTrue(pin.asleep)
        on_flats, flats_tilt = pc.lying_pin_rest().height, pc.lying_pin_rest().tilt
        on_round, round_tilt = pin_on_its_round_rest()
        self.assertGreater(pin.center()[2] / INCH, on_flats / INCH - 0.02)
        self.assertLess(pin.center()[2] / INCH, on_round / INCH + 0.02)
        tilt = abs(math.asin(pin.axis()[2]))
        self.assertGreater(tilt, min(flats_tilt, round_tilt) - math.radians(0.5))
        self.assertLess(tilt, max(flats_tilt, round_tilt) + math.radians(0.5))

    def test_a_pin_thrown_at_the_wall_stays_in_the_field(self):
        world = pc.Pieces()
        pin = world.add_pin(("red", "yellow"), (60.0 * INCH, 10.0 * INCH, 3.0 * INCH), (0.0, 1.0, 0.0))
        world.place(pin, pin.center(), pin.axis(), velocity=(3.0, 0.0, 0.0))
        run_pieces(world, 2.0)
        self.assertLess(pin.center()[0] + of.PIN_FLANGE_ACROSS_CORNERS / 2.0, of.HALF + 0.005)

    def test_a_rolling_pin_goes_round_the_point_its_taper_points_at(self):
        world = pc.Pieces()
        rest = pc.lying_pin_rest()
        pin = world.add_lying_pin(("red", "yellow"), OPEN_FLOOR[0], OPEN_FLOOR[1], 0.0)
        roll_rate = 8.0
        world.place_turned(pin, pin.center(), rest.quat(0.0), velocity=(0.0, -roll_rate * rest.height, 0.0),
                            spin=(roll_rate, 0.0, 0.0))
        path = []
        run_pieces(world, 0.8, lambda: path.append(pin.center()[:2]))
        expected = pin_roll_radius()
        self.assertAlmostEqual(fitted_radius(path[25:]) / INCH, expected / INCH, delta=0.1 * expected / INCH)

    def test_a_rolling_cup_goes_straight_because_its_rims_are_the_same(self):
        world = pc.Pieces()
        rim = float(cad.cup_column().outer[-1]) * INCH
        cup = world.add_cup((OPEN_FLOOR[0], OPEN_FLOOR[1], rim), (1.0, 0.0, 0.0))
        world.place(cup, cup.center(), (1.0, 0.0, 0.0), velocity=(0.0, 0.6, 0.0), spin=(-0.6 / rim, 0.0, 0.0))
        start = np.array(cup.center()[:2])
        run_pieces(world, 0.6)
        moved = np.array(cup.center()[:2]) - start
        self.assertGreater(moved[1] / INCH, 6.0)
        self.assertLess(abs(moved[0]), 0.02 * moved[1])

    def test_pieces_at_rest_do_not_creep_or_jitter(self):
        world = pc.Pieces()
        world.place_layout(of.starting_layout())
        before = {p.id: (p.center(), p.axis()) for p in world.pieces()}
        run_pieces(world, 10.0)
        for piece in world.pieces():
            center, axis = before[piece.id]
            self.assertLess(math.dist(center, piece.center()) / INCH, 0.002, piece.id)
            self.assertGreater(float(np.dot(axis, piece.axis())), math.cos(math.radians(0.05)), piece.id)

    def test_energy_never_grows(self):
        world = pc.Pieces()
        cup = standing_cup(world)
        world.add_pin(("red", "yellow"), (OPEN_FLOOR[0] + 0.4 * INCH, OPEN_FLOOR[1], 16.0 * INCH), tipped(25.0))
        world.add_pin(("blue", "yellow"), (OPEN_FLOOR[0] + 6.0 * INCH, OPEN_FLOOR[1], 10.0 * INCH), (1.0, 0.3, 0.4))
        world.add_cup((OPEN_FLOOR[0] - 7.0 * INCH, OPEN_FLOOR[1], 12.0 * INCH), (0.2, 1.0, 0.5))
        lowest = [energy(world)]
        rises = []

        def check():
            if hardest_press(world) > cup.mass * G:
                return
            now = energy(world)
            rises.append(now - lowest[0])
            lowest[0] = min(lowest[0], now)
        run_pieces(world, 3.0, check)
        self.assertGreater(len(rises), 500)
        self.assertLess(max(rises), 0.001 * cup.mass * G * INCH)


class TestStacks(unittest.TestCase):
    def test_a_pin_dropped_over_a_goal_nests_and_scores(self):
        world = pc.Pieces()
        g = of.goal("red left")
        pin = world.add_pin(("yellow", "red"), (g.x + 0.2 * INCH, g.y, g.height + 4.0 * INCH), tipped(3.0))
        run_pieces(world, 1.5)
        self.assertEqual(world.stacks(), [of.GoalStack("red left", [of.StackPin("yellow", "red")])])
        self.assertAlmostEqual((pin.center()[2] - g.riser) / INCH, cad.seats().pin_in_goal / INCH, delta=0.02)
        self.assertEqual(of.score(world.stacks(), {"left": "yellow"})["red"].total, 5)
        self.assertEqual(of.score(world.stacks(), {"left": "red"})["red"].total, 5)

    def test_stacks_nest_at_the_heights_the_shapes_give(self):
        world = pc.Pieces()
        g = of.goal("neutral left")
        seats = cad.seats()
        low = world.add_pin(("yellow", "red"), (g.x, g.y, g.height + 4.0 * INCH), of.UP)
        run_pieces(world, 1.0)
        cup = world.add_cup((g.x, g.y + 0.1 * INCH, low.center()[2] + seats.cup_on_pin + 2.0 * INCH), of.DOWN)
        run_pieces(world, 1.5)
        high = world.add_pin(("blue", "yellow"), (g.x, g.y, cup.center()[2] + seats.pin_in_cup + 2.0 * INCH), of.UP)
        run_pieces(world, 1.5)
        self.assertEqual(world.stacks()[0].goal, "neutral left")
        self.assertEqual(world.stacks()[0].layers, [of.StackPin("yellow", "red"), of.StackCup(lower_opaque=False),
                                                    of.StackPin("blue", "yellow")])
        self.assertEqual(of.scored_halves(world.stacks()[0]),
                         [("yellow", False), ("red", True), ("blue", False), ("yellow", True)])
        for upper, base, seat in ((low, g.riser, seats.pin_in_goal), (cup, low.center()[2], seats.cup_on_pin),
                                  (high, cup.center()[2], seats.pin_in_cup)):
            with self.subTest(piece=upper.kind, seat=seat):
                self.assertAlmostEqual((upper.center()[2] - base) / INCH, seat / INCH, delta=0.02)
                self.assertLess(tilt_deg(upper), 0.5)

    def test_the_starting_layout_is_at_rest_and_scores_nothing(self):
        world = pc.Pieces()
        world.place_layout(of.starting_layout())
        self.assertEqual(len(world.pins), 37)
        self.assertEqual(len(world.stacks()), 5)
        toggles = {q: "yellow" for q in of.QUADRANT_COLOR}
        totals = of.score(world.stacks(), toggles)
        self.assertEqual((totals["red"].total, totals["blue"].total), (0, 0))
        self.assertTrue(all(p.asleep for p in world.pieces()))


class TestRobotAndPieces(unittest.TestCase):
    def test_robot_pushes_a_lying_pin(self):
        sim, world = robot_world(0.0, -30.0, 0.0, objects=False)
        pin = world.pieces.add_lying_pin(("red", "yellow"), 14.0 * INCH, -26.0 * INCH, math.pi / 2.0)
        run(world, 0.3)
        start = pin.center()
        drive(world, 5.0, 1.2)
        self.assertGreater(pin.center()[0] - start[0], 4.0 * INCH)

    def test_the_preload_hangs_in_the_closed_claw_by_its_flange_and_does_not_slip(self):
        sim, world = robot_world(-61.7, -7.0, 0.0, preload="yellow")
        pin = world.preload
        self.assertIs(world.grip.piece, pin)
        self.assertTrue(world.robot.claw.closed)
        start = pin.center()[2]
        run(world, 2.0)
        self.assertIs(world.grip.piece, pin)
        self.assertLess(abs(pin.center()[2] - start) / INCH, 0.001)
        self.assertGreater(world.mechanism.squeeze, 40.0)

    def test_preload_orientation_puts_the_chosen_half_up(self):
        for upper in ("red", "yellow"):
            _, world = robot_world(-61.7, -7.0, 0.0, preload=upper)
            pin = world.grip.piece
            up_half = pin.colors[1] if pin.axis()[2] > 0.0 else pin.colors[0]
            self.assertEqual(up_half, upper)
            self.assertLess(tilt_deg(pin), 1.0)

    def test_opening_the_claw_drops_the_preload(self):
        sim, world = robot_world(-61.7, -7.0, 0.0, preload="yellow")
        pin = world.preload
        world.mechanism.claw.command(False)
        run(world, 1.0)
        self.assertIsNone(world.grip)
        self.assertEqual(world.drops[-1]["reason"], "the claw opened")
        self.assertLess(pin.center()[2], world.geometry.grip_lowest * INCH - 1.0 * INCH)


# ----------------------------------------------------------------- what the shapes say should happen

def lean_limit_deg():
    """The most a Pin standing in a Cup can lean, in the plane it leans in:
    lowered as far as it goes at each lean and slid to wherever that is
    lowest, the Pin cannot lean further once it would have to go up."""
    pin, cup = cad.pin_column(), cad.cup_column()
    s = np.linspace(pin.bottom, pin.top, 400)
    radius = np.interp(s, pin.z, pin.outer)
    outline = np.array([(r * a, z) for z, r in zip(s, radius) for a in np.linspace(-1.0, 1.0, 21)])

    def clear(angle, x, z):
        c, si = math.cos(angle), math.sin(angle)
        px = x + outline[:, 0] * c + outline[:, 1] * si
        pz = z - outline[:, 0] * si + outline[:, 1] * c
        inside = (pz > cup.bottom) & (pz < cup.top)
        reach = np.abs(px[inside])
        cavity = np.interp(pz[inside], cup.z, cup.cavity)
        wall = np.interp(pz[inside], cup.z, cup.outer)
        return not np.any((reach > cavity) & (reach < wall))

    def lowest(angle):
        best = math.inf
        for x in np.linspace(-0.8, 0.8, 81):
            z = cup.top + pin.top
            step = 0.05
            while step > 0.0005:
                while clear(angle, x, z - step):
                    z -= step
                step /= 2.0
            best = min(best, z)
        return best

    heights = [(angle, lowest(math.radians(angle))) for angle in np.arange(0.0, 30.0, 0.25)]
    leaning = [angle for (angle, z), (_, z_next) in zip(heights, heights[1:]) if z_next > z + 0.0005]
    return float(leaning[0]) if leaning else 30.0


def tipping_angle_deg():
    """How far a Cup standing on its rim with a Pin in it tips before its
    weight passes over the rim's edge."""
    cup_mass, pin_mass = pc.cup_mass(), pc.pin_mass()
    half = cad.cup().half_length
    cg = (cup_mass * half + pin_mass * (half + cad.seats().pin_in_cup)) / (cup_mass + pin_mass)
    return math.degrees(math.atan2(float(cad.cup_column().outer[0]) * INCH, cg))


def tipped_stack(world, angle_deg):
    """A Cup holding a Pin, tipped `angle_deg` over on the edge of its rim."""
    a = math.radians(angle_deg)
    half, rim = cad.cup().half_length, float(cad.cup_column().outer[0]) * INCH
    edge = np.array([OPEN_FLOOR[0] + rim, OPEN_FLOOR[1], 0.0])
    axis = np.array([math.sin(a), 0.0, math.cos(a)])
    across = np.array([math.cos(a), 0.0, -math.sin(a)])
    middle = edge - across * rim + axis * half
    cup = world.add_cup(tuple(middle), tuple(axis))
    pin = world.add_pin(("red", "yellow"), tuple(middle + axis * cad.seats().pin_in_cup), tuple(axis))
    return cup, pin


def rim_points(cup):
    a = np.array(cup.axis())
    u = np.cross(a, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    w = np.cross(a, u)
    rim = float(cad.cup_column().outer[0]) * INCH
    return [np.array(cup.center()) + a * cup.half_length * end + rim * (u * math.cos(t) + w * math.sin(t))
            for end in (-1.0, 1.0) for t in np.linspace(0.0, 2.0 * math.pi, 144, endpoint=False)]


def rim_edge(cup):
    """The lowest point of the Cup's rims."""
    return tuple(min(rim_points(cup), key=lambda p: p[2]))


def rim_point_nearest(cup, point):
    return tuple(min(rim_points(cup), key=lambda p: math.dist(p, point)))


def pin_roll_radius():
    """A lying Pin touches the floor on its flange and on one tip; the line
    through the two meets its axis at the point it rolls around.  The radius
    of the circle its middle goes round."""
    column = cad.pin_column()
    upper = column.z >= 0.0
    z, outer = column.z[upper], column.outer[upper]
    flange = int(np.argmax(outer))
    beyond = z > z[flange] + 0.5
    slope = float(np.min((outer[flange] - outer[beyond]) / (z[beyond] - z[flange])))
    apex = z[flange] + outer[flange] / slope
    return float(apex) * math.cos(math.atan(slope)) * INCH


def pin_on_its_round_rest():
    """A Pin's flange and tips are round with three flats.  Lying on the
    round part it touches the floor on the flange's widest edge and the tip,
    along the line tangent to both: (height of its middle, tilt of its axis)."""
    column = cad.pin_column()
    upper = column.z >= 0.0
    z, outer = column.z[upper], column.outer[upper]
    flange = int(np.argmax(outer))
    beyond = z > z[flange] + 0.5
    slope = float(np.min((outer[flange] - outer[beyond]) / (z[beyond] - z[flange])))
    height = (outer[flange] + slope * z[flange]) / math.sqrt(1.0 + slope ** 2)
    return float(height) * INCH, math.atan(slope)


def fitted_radius(points):
    """Least-squares circle through 2D points."""
    p = np.array(points)
    a = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))])
    cx, cy, c = np.linalg.lstsq(a, np.sum(p ** 2, axis=1), rcond=None)[0]
    return math.sqrt(c + cx ** 2 + cy ** 2)


if __name__ == "__main__":
    unittest.main()

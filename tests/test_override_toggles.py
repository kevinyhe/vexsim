"""The Toggles as MuJoCo bodies: a bar on hubs in slotted brackets that
comes to rest on a face, or wedges between its slots and a bracket's plate,
and the colour facing into the field decides who owns the yellow Pin halves
in that Quadrant's Goals (<SC4>)."""

import math
import unittest

import mujoco

from vexsim import Simulator, cascade, pieces, presets, toggles
from vexsim import override_field as of
from vexsim.override import OverrideWorld
from vexsim.sim import SimConfig
from vexsim.units import INCH

CLEAR_OF_EVERYTHING = (-40.0 * INCH, -12.0 * INCH, 0.0)


def let_go(degrees):
    """The left Toggle's bar let go turned `degrees` with its hubs at the top
    of their slots, on a field with nothing else on it: (where it ends up
    turned, degrees, how far its hubs are up the slots, inches, its colour)."""
    field = pieces.Pieces(pins=0, cups=0)
    bar = next(b for b in field.toggles.bars if b.mount.quadrant == "left")
    field.data.qpos[bar._hinge] = math.radians(degrees)
    field.data.qpos[bar._slide] = field.model.joint(f"{toggles.bar_name(bar.mount)} slot").range[1]
    mujoco.mj_forward(field.model, field.data)
    for _ in range(round(2.0 / pieces.DT)):
        field.step()
    return math.degrees(bar.angle), bar.lift / INCH, bar.color()


def world_at_rest():
    sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
    sim.set_pose(*CLEAR_OF_EVERYTHING)
    return OverrideWorld(sim, robot=cascade.CascadeRobot(), objects=False)


def run(world, seconds):
    for _ in range(round(seconds * 1000)):
        world.step(0.001)


class TestToggleAtRest(unittest.TestCase):
    def test_every_toggle_starts_yellow_and_stays_on_its_face(self):
        world = world_at_rest()
        run(world, 0.5)
        self.assertEqual(world.toggle_colors, {q: "yellow" for q in of.QUADRANT_COLOR})
        for bar in world.toggles.bars:
            self.assertLess(abs(bar.angle), math.radians(1.0))

    def test_let_go_20_degrees_off_its_face_it_falls_back(self):
        for degrees in (-20.0, 20.0):
            with self.subTest(degrees=degrees):
                angle, lift, color = let_go(degrees)
                self.assertLess(abs(angle), 1.0)
                self.assertLess(lift, 0.01)
                self.assertEqual(color, "yellow")

    def test_let_go_100_degrees_round_it_drops_onto_the_next_face(self):
        for degrees, face, color in ((-100.0, -120.0, "red"), (100.0, 120.0, "blue")):
            with self.subTest(degrees=degrees):
                angle, lift, set_color = let_go(degrees)
                self.assertAlmostEqual(angle, face, delta=1.0)
                self.assertLess(lift, 0.01)
                self.assertEqual(set_color, color)

    def test_let_go_50_to_80_degrees_round_it_wedges_between_its_slots_and_a_plate(self):
        for degrees in (-80.0, -50.0, 50.0, 80.0):
            with self.subTest(degrees=degrees):
                angle, lift, color = let_go(degrees)
                self.assertAlmostEqual(angle, degrees, delta=3.0)
                self.assertGreater(lift, 0.25)
                self.assertEqual(color, "yellow")


class TestToggleScoring(unittest.TestCase):
    def test_each_toggle_setting_changes_who_owns_the_yellow_half(self):
        world = world_at_rest()
        g = of.goal("red left")
        world.pieces.add_pin(("red", "yellow"), (g.x, g.y, g.height + 4.0 * INCH), (0.0, 0.0, 1.0))
        run(world, 1.2)
        self.assertEqual(world.stacks(), [of.GoalStack("red left", [of.StackPin("red", "yellow")])])
        bar = next(b for b in world.toggles.bars if b.mount.quadrant == "left")
        totals = {}
        for turn, color in ((0.0, "yellow"), (-120.0, "red"), (120.0, "blue")):
            world.data.qpos[bar._hinge] = math.radians(turn)
            self.assertEqual(world.toggle_colors["left"], color)
            score = world.score()
            totals[color] = (score["red"].total, score["blue"].total)
        self.assertEqual(totals, {"yellow": (0, 0), "red": (10, 0), "blue": (0, 10)})

    def test_a_toggle_off_its_face_counts_as_yellow(self):
        world = world_at_rest()
        bar = next(b for b in world.toggles.bars if b.mount.quadrant == "left")
        world.data.qpos[bar._hinge] = math.radians(-120.0 + 10.0)
        self.assertEqual(bar.color(), "yellow")


if __name__ == "__main__":
    unittest.main()

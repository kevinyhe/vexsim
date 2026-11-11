"""Override field geometry, starting layout and scoring against the manual."""

import math
import unittest

from vexsim import override_cad as cad
from vexsim import override_field as of
from vexsim.units import INCH, MM


def stack(goal, *layers):
    return of.GoalStack(goal, list(layers))


P = of.StackPin
C = of.StackCup


class TestGeometry(unittest.TestCase):
    def test_goals_match_the_vr_field_gps_positions(self):
        expected_mm = {
            "center": (0, 0), "neutral top": (-600, 1200), "neutral left": (-1200, 600),
            "neutral right": (1200, -600), "neutral bottom": (600, -1200),
            "red left": (-1200, -600), "red bottom": (-600, -1200),
            "blue top": (600, 1200), "blue right": (1200, 600),
        }
        for name, (x_mm, y_mm) in expected_mm.items():
            g = of.goal(name)
            self.assertAlmostEqual(g.x, x_mm * MM, delta=15 * MM, msg=name)
            self.assertAlmostEqual(g.y, y_mm * MM, delta=15 * MM, msg=name)

    def test_alliance_goals_sit_in_their_own_colour_quadrant(self):
        for g in of.GOALS:
            if g.kind in ("red", "blue"):
                self.assertEqual(of.QUADRANT_COLOR[g.quadrant], g.kind, g.name)
                self.assertEqual(of.side_of(g.x, g.y), g.kind, g.name)

    def test_each_quadrant_has_two_goals_and_one_toggle(self):
        for quadrant in of.QUADRANT_COLOR:
            goals = [g for g in of.GOALS if g.quadrant == quadrant]
            self.assertEqual(len(goals), 2, quadrant)
            toggles = [t for t in of.TOGGLES if of.quadrant_of(t.x, t.y) == quadrant]
            self.assertEqual(len(toggles), 1, quadrant)

    def test_only_the_center_goal_is_in_the_midfield(self):
        self.assertEqual([g.name for g in of.GOALS if of.in_midfield(g.x, g.y)], ["center"])

    def test_goal_heights(self):
        self.assertAlmostEqual(of.goal("red left").height / INCH, 3.25)
        self.assertAlmostEqual(of.goal("neutral top").height / INCH, 5.77)
        self.assertAlmostEqual(of.goal("center").height / INCH, 8.77)

    def test_loaders_match_the_vr_field_and_sit_in_their_load_zones(self):
        for loader in of.LOADERS:
            self.assertAlmostEqual(abs(loader.x), 1740 * MM, delta=10 * MM)
            self.assertAlmostEqual(abs(loader.y), 1490 * MM, delta=10 * MM)
            self.assertEqual(loader.x < 0.0, loader.alliance == "red")
            xmin, xmax, ymin, ymax = of.load_zone(loader)
            self.assertTrue(xmin <= loader.x <= xmax and ymin <= loader.y <= ymax)


class TestStartingLayout(unittest.TestCase):
    def setUp(self):
        self.layout = of.starting_layout()

    def test_counts_match_the_field_overview(self):
        self.assertEqual(len(self.layout.pins), 37)
        self.assertEqual(len(self.layout.cups), 36)
        clear_up = [c for c in self.layout.cups if c.axis[2] > 0.0]
        self.assertEqual(len(clear_up), 12)
        kinds = {}
        for pin in self.layout.pins:
            key = "/".join(sorted(pin.colors))
            kinds[key] = kinds.get(key, 0) + 1
        self.assertEqual(kinds, {"yellow/yellow": 17, "red/yellow": 8, "blue/yellow": 8,
                                 "blue/red": 4})

    def test_every_neutral_goal_starts_with_a_yellow_pin(self):
        in_goals = {p.in_goal for p in self.layout.pins if p.in_goal}
        self.assertEqual(in_goals, {g.name for g in of.GOALS if g.kind in ("neutral", "center")})

    def test_everything_starts_inside_the_field(self):
        for item in self.layout.pins + self.layout.cups:
            x, y, _ = item.position
            self.assertLess(abs(x), of.HALF)
            self.assertLess(abs(y), of.HALF)

    def test_standing_pins_rest_on_their_cup(self):
        for pin in self.layout.pins:
            if pin.in_cup is None:
                continue
            cup = self.layout.cups[pin.in_cup]
            self.assertAlmostEqual(pin.position[0], cup.position[0])
            self.assertAlmostEqual(pin.position[2] - cup.position[2], cad.seats().pin_in_cup)


class TestToggle(unittest.TestCase):
    def test_starts_yellow_and_turns_to_either_alliance(self):
        left = next(t for t in of.TOGGLES if t.quadrant == "left")
        step = 2.0 * math.pi / 3.0
        self.assertEqual(of.toggle_color(left.faces, 0.0, False), "yellow")
        self.assertEqual(of.toggle_color(left.faces, step, False), "blue")
        self.assertEqual(of.toggle_color(left.faces, -step, False), "red")
        self.assertEqual(of.toggle_color(left.faces, 2 * step, False), "red")

    def test_not_set_while_touched_or_between_faces(self):
        faces = of.TOGGLES[0].faces
        step = 2.0 * math.pi / 3.0
        self.assertEqual(of.toggle_color(faces, step, True), "yellow")
        self.assertEqual(of.toggle_color(faces, step * 0.5, False), "yellow")


class TestScoring(unittest.TestCase):
    def quadrant_example(self):
        """The stack in the manual's three-Toggle example under <SC5>, from
        the Goal up. 3 red, 1 blue and 3 yellow halves show; the Pin in the
        Goal and the Pin between two opaque halves show none."""
        return [
            stack("neutral left", P("yellow", "yellow"), C(lower_opaque=True), P("yellow", "red"),
                  C(lower_opaque=False), P("blue", "blue"), C(lower_opaque=True), P("yellow", "yellow"),
                  C(lower_opaque=False), P("red", "blue"), C(lower_opaque=False), P("yellow", "red"),
                  C(lower_opaque=False), P("yellow", "red")),
        ]

    def test_manual_toggle_example(self):
        stacks = self.quadrant_example()
        for toggle, red, blue in (("yellow", 15, 5), ("blue", 15, 35), ("red", 45, 5)):
            result = of.score(stacks, {"left": toggle})
            self.assertEqual((result["red"].total, result["blue"].total), (red, blue), toggle)

    def test_cup_opaque_half_hides_the_pin_half_inside_it(self):
        halves = of.scored_halves(stack("red left", P("red", "yellow"), C(lower_opaque=True),
                                        P("blue", "red")))
        self.assertEqual(halves, [("red", False), ("yellow", False), ("blue", True), ("red", True)])
        halves = of.scored_halves(stack("red left", P("red", "yellow"), C(lower_opaque=False),
                                        P("blue", "red")))
        self.assertEqual(halves, [("red", False), ("yellow", True), ("blue", False), ("red", True)])

    def test_the_half_in_the_goal_does_not_score(self):
        stacks = [stack("neutral left", P("red", "yellow"))]
        self.assertEqual(of.score(stacks, {"left": "red"})["red"].total, 10)
        self.assertEqual(of.score([stack("red left", P("yellow", "red"))], {"left": "red"})["red"].total, 5)

    def test_midfield_yellow_goes_to_the_alliance_with_more_robots(self):
        stacks = [stack("center", P("yellow", "yellow"))]
        self.assertEqual(of.score(stacks, {}, {"red": 1, "blue": 0})["red"].total, 10 + 8)
        tie = of.score(stacks, {}, {"red": 1, "blue": 1})
        self.assertEqual((tie["red"].total, tie["blue"].total), (8, 8))

    def test_autonomous_leaves_out_the_midfield(self):
        stacks = [stack("center", P("yellow", "red"))]
        result = of.score(stacks, {}, {"red": 2}, autonomous=True)
        self.assertEqual(result["red"].total, 5)

    def test_autonomous_bonus(self):
        lead = of.score([stack("red left", P("red", "yellow"))], {"left": "red"}, autonomous=True)
        self.assertEqual(of.autonomous_bonus(lead), {"red": 12, "blue": 0})
        tie = of.score([], {}, autonomous=True)
        self.assertEqual(of.autonomous_bonus(tie), {"red": 6, "blue": 6})

    def test_autonomous_win_point(self):
        toggles = {"left": "red", "bottom": "red", "top": "yellow", "right": "yellow"}
        three_halves = (P("red", "yellow"), C(lower_opaque=False), P("red", "yellow"), C(lower_opaque=False),
                        P("red", "yellow"))
        stacks = [stack("red left", *three_halves), stack("red bottom", *three_halves)]
        self.assertEqual(of.win_point_progress("red", stacks, toggles), (6, 2))
        self.assertFalse(of.autonomous_win_point("red", [stacks[0], stack("red bottom", *three_halves[:3])],
                                                 toggles, touching_perimeter=False))
        self.assertTrue(of.autonomous_win_point("red", stacks, toggles, touching_perimeter=False))
        self.assertFalse(of.autonomous_win_point("red", stacks, toggles, touching_perimeter=True))
        all_red = (P("red", "red"), C(lower_opaque=False), P("red", "red"), C(lower_opaque=False), P("red", "red"))
        across = [stack("blue top", *all_red), stack("blue right", *all_red)]
        self.assertFalse(of.autonomous_win_point("red", across, toggles, touching_perimeter=False))

    def test_skills_needs_the_toggle_to_match_its_quadrant(self):
        stacks = [stack("neutral top", P("yellow", "yellow"))]
        self.assertEqual(of.score(stacks, {"top": "red"})["red"].total, 10)
        self.assertEqual(of.score(stacks, {"top": "red"}, skills=True)["red"].total, 0)


if __name__ == "__main__":
    unittest.main()

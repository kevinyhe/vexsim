"""VEX's Override field CAD (dated 26 April 2026) against manual 2.0
(3 September 2026), and the shapes made from it against where the CAD
itself puts pieces."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import override_cad as cad
from vexsim import override_field as of
from vexsim.units import INCH


def inches(metres):
    return metres / INCH


def widest_across(shape, from_s_in, to_s_in):
    return 2.0 * max(shape.radius_at((from_s_in + (to_s_in - from_s_in) * k / 40.0) * INCH) for k in range(41))


class TestPiecesMatchTheManual(unittest.TestCase):
    def test_pin(self):
        pin = cad.pin()
        self.assertAlmostEqual(inches(2.0 * pin.half_length), inches(of.PIN_LENGTH), delta=0.01)
        self.assertAlmostEqual(inches(widest_across(pin, 0.0, 0.1)), inches(of.PIN_FLANGE_ACROSS_CORNERS), delta=0.01)
        self.assertAlmostEqual(inches(2.0 * pin.radius_at(of.PIN_LENGTH / 2.0 - 0.3 * INCH)),
                               inches(of.PIN_TIP_DIAMETER), delta=0.02)
        past_the_fillet = of.PIN_FLANGE_THICKNESS / 2.0 + 0.05 * INCH
        self.assertAlmostEqual(inches(2.0 * pin.radius_at(past_the_fillet)), inches(of.PIN_CONE_DIAMETER), delta=0.02)

    def test_a_pins_cone_tapers_as_the_manuals(self):
        pin = cad.pin()
        wide = of.PIN_FLANGE_THICKNESS / 2.0 + 0.05 * INCH
        narrow = of.PIN_LENGTH / 2.0 - of.PIN_TIP_STRAIGHT
        taper = math.atan((pin.radius_at(wide) - pin.radius_at(narrow)) / (narrow - wide))
        self.assertAlmostEqual(math.degrees(taper), math.degrees(of.PIN_TAPER), delta=0.5)

    def test_a_pins_flange_squeezed_onto_a_flat_is_as_wide_as_the_manuals_flats(self):
        pin = cad.pin()
        self.assertAlmostEqual(inches(pin.squeezed_width_at(0.02 * INCH)), inches(of.PIN_FLANGE_ACROSS_FLATS),
                               delta=0.05)

    def test_cup(self):
        cup = cad.cup()
        self.assertAlmostEqual(inches(2.0 * cup.half_length), inches(of.CUP_LENGTH), delta=0.02)
        self.assertAlmostEqual(inches(2.0 * cup.radius_at(cup.half_length)), inches(of.CUP_RIM_DIAMETER), delta=0.01)
        self.assertAlmostEqual(inches(2.0 * cup.radius_at(0.04 * INCH)), inches(of.CUP_WAIST_DIAMETER), delta=0.02)

    def test_goal_taper(self):
        column = cad.goal_column()
        self.assertAlmostEqual(column.top - column.bottom, inches(of.GOAL_TAPER_HEIGHT), delta=0.01)

    def test_risers(self):
        for kind, height in (("neutral", of.GOAL_NEUTRAL_RISER), ("center", of.GOAL_CENTER_RISER)):
            with self.subTest(kind=kind):
                across_x, across_y, tall = cad.riser_size(kind)
                self.assertAlmostEqual(inches(tall), inches(height), delta=0.01)
                self.assertAlmostEqual(inches(across_x), inches(of.GOAL_BASE_ACROSS_FLATS), delta=0.01)
                self.assertAlmostEqual(inches(across_y), inches(of.GOAL_BASE_ACROSS_FLATS), delta=0.01)


class TestShapesMeetWhereTheCadPutsThem(unittest.TestCase):
    def test_pins_stand_in_cups_as_high_as_in_the_cad(self):
        field = cad.field()
        heights = []
        for pin in field["pins"]:
            px, py, pz = pin["middle_in"]
            for cup in field["cups"]:
                cx, cy, cz = cup["middle_in"]
                if math.hypot(px - cx, py - cy) < 0.05 and 0.0 < pz - cz < 4.0:
                    heights.append(pz - cz)
        self.assertEqual(len(heights), 16)
        for height in heights:
            self.assertAlmostEqual(height, inches(cad.seats().pin_in_cup), delta=0.002)

    def test_pins_stand_in_goals_as_high_as_in_the_cad(self):
        field = cad.field()
        heights = []
        for goal in field["goals"]:
            gx, gy, gz = goal["position_in"]
            for pin in field["pins"]:
                px, py, pz = pin["middle_in"]
                if math.hypot(px - gx, py - gy) < 0.05 and 0.0 < pz - gz < 4.0:
                    heights.append(pz - gz)
        self.assertEqual(len(heights), 5)
        for height in heights:
            self.assertAlmostEqual(height, inches(cad.seats().pin_in_goal), delta=0.002)


class TestTheStartingLayoutIsTheCads(unittest.TestCase):
    def test_every_piece_of_the_layout_is_where_the_cad_has_one(self):
        layout, field = of.starting_layout(), cad.field()
        for kind, starts in (("pins", layout.pins), ("cups", layout.cups)):
            placed = [piece["middle_in"] for piece in field[kind]]
            for start in starts:
                x, y, _ = (inches(v) for v in start.position)
                with self.subTest(kind=kind, x=round(x, 2), y=round(y, 2)):
                    self.assertLess(min(math.hypot(x - p[0], y - p[1]) for p in placed), 0.03)


if __name__ == "__main__":
    unittest.main()

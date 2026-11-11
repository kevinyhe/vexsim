"""The grey EVA foam field tiles.

Two things about the tiles change how a robot drives: how much grip they give
each kind of wheel, and how far a wheel sinks into them.  The grip numbers are
the forum's measurements of VEX wheels on VEX tiles.  Nobody has published
rolling resistance or turning scrub on foam, so those come from the sink: each
bit of tile under a wheel acts as its own spring, the wheel sits in the dent it
makes, and rolling forward keeps pressing in new foam.  EVA gives back only
part of the work that pressed it in.  The rest is the rolling resistance, and
the size of the dent is the size of the patch that scrubs when a wheel twists.

Numbers that no source gives are in `GUESSES`.
"""

from __future__ import annotations

import math
from typing import Dict

from .units import INCH

#: https://www.vexrobotics.com/antistatic-tiles.html: 0.63 in thick, 2 ft by
#: 2 ft, 1.1 lb (499 g) each.
TILE_THICKNESS = 0.63 * INCH
TILE_SIDE = 24.0 * INCH
TILE_MASS = 0.499
TILE_DENSITY = TILE_MASS / (TILE_SIDE ** 2 * TILE_THICKNESS)

#: Compressive modulus of EVA foam at 80 kg/m^3, the measured density nearest
#: a tile's 84 kg/m^3.  Chen et al., Polymers 16(1):34 (2023),
#: doi:10.3390/polym16010034, https://pmc.ncbi.nlm.nih.gov/articles/PMC10780418/
FOAM_MODULUS = 2.77e6
#: Pressure per metre of dent: the tile's modulus over its thickness
FOAM_STIFFNESS = FOAM_MODULUS / TILE_THICKNESS
#: EVA gives back about 62% of the work that compresses it at 30% strain
#: (https://ctyeh.com/articles/14568, which does not name its own source).
FOAM_ENERGY_LOSS = 1.0 - 0.62

#: Static pull test of VEX wheels on foam, a fish scale pulling a loaded base:
#: 4 in omni 0.90, 2.75 in double omni 0.84, 2.75 in single omni 0.59.
#: https://www.vexforum.com/t/pull-tests-for-vex-wheels/20141
OMNI_GRIP_MEASURED = {2.75: 0.84, 4.0: 0.90}
#: Tilt test, a tile raised until the robot slid: the two 4 in wheels slipped
#: at 0.93 and 0.81, and the thread's posts swap which one was the omni.  The
#: traction wheel gets 0.81 because the pull test above and
#: https://www.vexforum.com/t/geared-4-motor-drivetrain/95509/52 both have
#: omnis gripping foam better than traction wheels.
#: https://www.vexforum.com/t/anyone-have-friction-coefficient-of-vex-wheels/19261
TRACTION_GRIP = 0.81
#: Locked mecanum wheels dragged across foam measured 0.85 at best.
#: https://www.vexforum.com/t/omni-wheel-vs-mecanum/48151/13
MECANUM_GRIP = 0.85
#: The only sideways omni measurement: AndyMark omnis on FRC carpet, 0.2 to
#: 0.3.  The VEX foam test above only says "almost no resistance" crosstrack.
#: https://www.chiefdelphi.com/t/friction-coefficients-for-omni-wheels-and-mecanum-wheels-from-andymark/74918
OMNI_SIDEWAYS_CARPET = (0.2, 0.3)

GUESSES: Dict[str, str] = {
    "3.25 in omni grip": "0.87, halfway between the measured 2.75 in (0.84) and 4 in (0.90) omnis.",
    "Sliding grip": "A sliding wheel keeps 0.81 of its static grip. No test separates the two on foam.",
    "Omni sideways grip": "0.26 static, 0.22 sliding: inside the 0.2-0.3 measured on carpet. No one has "
                          "measured it on foam. The foam model above, run on one roller 0.2 to 0.3 in in "
                          "radius and 0.2 to 0.4 in wide under a quarter of a 16.5 lb robot, gives 0.05 to "
                          "0.08 from the foam alone. The rollers' axles and the rubber on them add to that by "
                          "an amount no one has measured, so the carpet figure stays.",
    "Mecanum sideways grip": "0.30 static, 0.25 sliding, a little above an omni for the angled rollers.",
    "Flex wheel grip": "1.32 (60A) to 1.75 (30A). Untested; the forum only says softer grips a bit more.",
    "Contact width": "Omni 0.4 in (one roller from each row), traction 1.0 in, flex 1.0 in, mecanum "
                     "0.5 in: the width of wheel that presses into the tile.",
    "Rolling loss": "Only the foam's loss counts. The wheel's own rubber is ignored, and so is the work "
                    "of an omni roller passing over a seam.",
    "Pin and Cup on foam": "Sliding 0.5, bounce 0.15 and spin damping 3/s, in pieces.GUESSES.",
}


def sink(load: float, radius: float, width: float) -> float:
    """How far a wheel sinks into a tile under this load, in metres.

    A wheel pressed into foam that acts as a bed of springs carries
    load = 4/3 x FOAM_STIFFNESS x width x sink x sqrt(2 x radius x sink).
    A wheel off the ground (no load) does not sink.
    """
    if load <= 0.0:
        return 0.0
    return (0.75 * load / (FOAM_STIFFNESS * width * math.sqrt(2.0 * radius))) ** (2.0 / 3.0)


def patch_half_length(radius: float, depth: float) -> float:
    """Half the length of the dent along the way the wheel rolls."""
    return math.sqrt(2.0 * radius * depth)


def rolling_resistance(load: float, radius: float, width: float) -> float:
    """Rolling resistance coefficient: resisting force over load.

    Rolling one metre presses width x 1 m of foam down by the sink, which takes
    FOAM_STIFFNESS x width x sink^2 / 2 of work; the foam keeps FOAM_ENERGY_LOSS
    of it.  Over the load, that is 3/8 x loss x sqrt(sink / (2 x radius)).
    """
    depth = sink(load, radius, width)
    return 0.375 * FOAM_ENERGY_LOSS * math.sqrt(depth / (2.0 * radius))


def scrub_arm(load: float, radius: float, width: float, mu_along: float, mu_across: float) -> float:
    """Twisting torque per newton of load when a wheel turns in place on its patch, in metres.

    A twist drags the sides of the patch along the rolling direction and its
    ends across it.  Across the patch the pressure is even, so the sides sit
    width/4 out on average; along it the pressure falls off to the edges of
    the dent, so the ends sit 3/8 of the half length out.
    """
    half_length = patch_half_length(radius, sink(load, radius, width))
    return mu_along * width / 4.0 + mu_across * 0.375 * half_length

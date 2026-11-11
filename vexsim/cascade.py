"""A cascade-lift robot like the ones teams built for Override.

On a four-motor drivetrain:

* A bite: a round notch cut into the front of the frame, reaching back
  past the front wheels' axles, with the drivetrain on both sides of it.
  Its lips flare out so a Goal driven at it slides in, and its round back
  holds the Goal's base so the Goal's middle sits right under the claw.
* A five-stage cascade lift behind the bite, standing on the frame.  Three
  11 W motors turn, through gears, a drum that winds in the string pulling
  the first stage up; string rigging makes each stage move on the one
  under it as far as the first moves on the frame, so the carriage rises
  five times as far.  Each string stretches like a damped spring.  Rubber
  bands help it up.  The stages are bodies on slides with hard stops at
  both ends.  Each slide rubs with friction from what it carries: its
  bearings hold the stages above it, and the twist of the claw hanging
  ahead of the rails presses them on the rails.
* A pneumatic claw on the carriage, reaching forward over the middle of the
  bite.  A piston drives both jaws through a rack and pinion, with the force
  the tank's pressure gives it.  Each jaw is a V seen from above, so what it
  closes on is pushed onto the claw's middle, and it is built in three
  layers: a band at the bottom, a recess cut back from it, and a band at the
  top that reaches further in.  On a stack (a Pin standing in a Cup) the
  bottom band presses on the Cup below its flared rim and the top band on
  the Pin above its flange, and the ridge between, where the Pin's flange
  sits on the Cup's rim, is in the recess and touches nothing.  So each jaw
  holds the stack at two heights.  The jaws hold by friction.  A lone Cup is
  held by the bottom bands; a lone Pin hung in the claw rests its flange on
  the bottom bands and is squeezed by the top ones.  With the lift down the
  claw's middle is level with the flange of a stack standing on the floor.
  A Pin standing on its own on the floor is too thin where the bands reach,
  so they close past it.  A plate behind the jaws reaches down into the
  bite and stops a stack the robot drives onto with its middle under the
  claw's, top and bottom together, so it does not tip.
* A roller across the back for the Toggles: one axle on two posts, turned
  by one 11 W motor through gears, with a roller at each end.  Each roller
  is a metal channel fixed at its middle with a rubber flap on each end.
  With the robot's back against the wall at a Toggle, the flaps reach over
  the wall and turn the Toggle's bar like a gear, its top toward the wall,
  one face for each flap.  Run the other way they press the bar onto its
  brackets and it does not turn.  Each flap bends on its root like rubber.
  Stopped, the flaps are fixed arms.
* Bumpers around the frame's outline, drawn only: they are the outside of
  the frame, not extra collision.

Every piece is a `Component` that names the pieces it is fixed to or slides
on.  `CascadeRobot.build` makes them MuJoCo bodies and shapes, and the
planner's 3D view draws these same pieces.

All lengths here are inches, angles degrees and masses kilograms.  Where the
numbers come from is in `SOURCES`; the numbers no source gives are in
`GUESSES`; the ones the robot's owner chose are in `CHOSEN`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import override_cad as cad
from . import override_field as of
from . import pieces
from .arena import Part, outline_part, profile_part
from .chassis import Chassis
from .motor import Motor
from .mujoco_drive import Lump, add_drivetrain
from .units import DEG, INCH

CUP_HALF_IN = cad.cup().half_length / INCH
PIN_HALF_IN = cad.pin().half_length / INCH
#: From a Pin's middle to either face of its flange (the CAD's first convex part)
PIN_FLANGE_HALF_IN = float(np.max(cad.pin().hulls[0][:, 2])) / INCH
PASCALS_PER_PSI = 6894.757
ROBOT = "robot"
DRUM = f"{ROBOT} lift drum"
DRUM_MASS_KG = 0.02
#: Where the drum sits on the frame, inches: forward, up
DRUM_AT = (0.85, 3.0)

SOURCES: Dict[str, str] = {
    "Three-stage cascade on two 11 W motors, an 18-tooth sprocket pulling 18 x 0.385 in of chain per turn, "
    "the top stage moving three times the first, 43.5 in tall fully up":
        "Team 55885X's Override code, https://github.com/Jim-Ox/55885X-Inquisitors-2026-2027-VEX-Override-Code"
        "/blob/HEAD/src/lift.cpp",
    "Cascade lifts on two 11 W green motors":
        "6104M https://github.com/k1nguy3n/6104M-Monsoon-VEX-Override-2026-2027/blob/HEAD/src/subsystems/"
        "intake.cpp and 569C https://github.com/daydayparker/VexOverride569C/blob/HEAD/src/subsystem_files/"
        "globals.cpp",
    "A three-stage cascade turns 12 in of first-stage travel into 36 in at the top; 22 W for the two "
    "cascade motors inside the 88 W limit":
        "https://www.vexforum.com/t/override-robot-with-a-cascade-lift/148629",
    "Rubber bands pull the lift up, so going down needs more power":
        "Team 1862X's Override code, https://github.com/ShadowKingYT444/Goofy_Goobers_code_OVERRIDE",
    "A pneumatic claw about 4 in wide, with a rounded cut-out that lets Goals and stacks sit deeper":
        "Team 1862X's Override code, https://github.com/ShadowKingYT444/Goofy_Goobers_code_OVERRIDE",
    "A pneumatic claw on a cascade":
        "Cosmic Electrons' Override code, https://github.com/CyrusKuhlmann/CE_OVERRIDE",
    "A passive Toggle pusher on the front, pressed into the wall and then backed off, because a robot "
    "touching the Toggle stops it scoring":
        "https://github.com/BraedynMendonca/override-autonomous and the manual's SC4, "
        "https://content.vexrobotics.com/docs/2026-2027/override/files/override-2.0.pdf",
    "A drive of four 11 W blue motors geared 36:48 on 3.25 in omnis (450 rpm)":
        "https://github.com/dafestin79/VEX-Override-LEMLIB2026",
    "Start within 18 x 18 x 18 in (SG1), never wider or longer than 24 in (SG2) or taller than 50 in (SG3), "
    "at most 88 W of motors (R10) and 55 W on the drive (R11), at most two air tanks at 100 psi (R25), one "
    "Pin and one Cup held at a time (SG6); holding the bottom of a stack holds all of it (Q&A 3166)":
        "https://content.vexrobotics.com/docs/2026-2027/override/files/override-2.0.pdf and "
        "https://events.vex.com/faqs/51/pdf",
    "An 11 W motor weighs 155 g": "https://kb.vex.com/hc/en-us/articles/360044325872",
    "The V5 battery weighs 350 g":
        "https://kb.vex.com/hc/en-us/articles/360060529912-Understanding-the-V5-Robot-Battery",
    "The V5 brain weighs 285 g":
        "https://kb.vex.com/hc/en-us/articles/360060662352-Understanding-the-V5-Robot-Brain",
    "A 200 mL air tank in the V5 Pneumatics Kit":
        "https://kb.vex.com/hc/en-us/articles/17984679227156-Legal-Pneumatics-for-Use-in-the-VEX-Robotics-Competition-VRC",
    "VEX cylinders have a 10 mm bore; the older kit's 150 mL tank, 2 in double-acting cylinders and 4 mm "
    "tubing give 45 strokes from 100 to 25 psi, and tubing between the valve and the cylinder loses more "
    "pressure each stroke":
        "VEX Pneumatics Kit sheet, pages 1, 4 and 8, https://www.vexforum.com/uploads/default/original/2X/b/"
        "bd03203edc92eec453fb915524d36209cd47f575.pdf",
}

CHOSEN: Dict[str, str] = {
    "Lift": "The robot's owner, 2026-11-04/05: a string-driven cascade with rubber bands on three 11 W motors, "
            "the robot's top at most 50 in fully up, and the string stretching like a spring with damping. "
            "On 2026-11-06 the owner put a full raise at about 1.2 s.",
    "Aligner": "A bite cut into the front of the frame that the Goal sits in, not guides sticking out.",
    "Claw over the bite": "The claw hangs from the carriage right over the middle of the bite, so a seated "
                          "Goal is under the held Cup and the cascade lowers it straight down.",
    "What the claw holds": "The robot's owner, 2026-11-04/05: on a stack each jaw touches at two heights, the Pin "
                           "above the ridge where its flange sits on the Cup's rim and the Cup below it, with "
                           "the ridge in a recess between. A lone Cup is held by the bottom bands. A lone Pin "
                           "hangs with its flange on, or up to 0.02 in over, the bottom bands and is squeezed by "
                           "the top bands. The jaws close past every Pin's flange, so a Pin's three flats cannot "
                           "let it through.",
    "Claw": "The robot's owner, 2026-11-04/05: one pneumatic piston, V-shaped jaws (35 degrees kept), rubber pads "
            "with friction 0.9 that give about 0.01 in, gripping slightly harder than the 48.5 N a jaw the "
            "sim had before.",
    "Toggles": "The robot's owner, 2026-11-04/05: a powered roller on one 11 W motor turns them. One "
               "axle across the back with a roller at each end, each a rigid 3 in metal channel fixed at its "
               "middle with a VEX rubber flap on each end, 180 degrees apart. Each flap is about 0.5 by 1 in "
               "with its tip 2.5 in from the axle, bends, and is not very grippy. A 200 rpm cartridge geared "
               "12:36 turns it at 67 rpm, about the owner's 60. Stopped, the flaps act as fixed arms.",
    "Look": "Drawn like an FRC robot: a boxy frame with bumpers around it and the wheels hidden inside.",
}

GUESSES: Dict[str, str] = {
    "Frame": "The robot's owner, 2026-11-04/05: 15 x 12.5 in. It is 0.4 to 5.0 in off the floor (a guess).",
    "Robot mass": "The robot's owner, 2026-11-04/05: about 14 lb (6.35 kg) with the battery, its CG about 3.5 in up "
                  "with the lift down.",
    "Drive": "The robot's owner, 2026-11-04/05: four 2.75 in omni wheels at 450 rpm on four 11 W motors, and one "
             "sprung, unpowered tracking wheel at the centre of rotation that measures sideways "
             "(`presets.override_cascade`).",
    "Bite": "The middle of a seated Goal is 1.87 in behind the frame front, which is also the middle of a "
            "Loader, so the claw is over a Loader's stack when the robot is against the wall. The round back "
            "is 3.36 in from that point (a Goal's base reaches 3.22 in), drawn as four straight pieces a side, "
            "and the lips flare to 4.5 in either side of the middle at the frame front. The bite reaches 5.23 "
            "in into the frame, past the front wheels' axles 2.25 in from the front.",
    "Cascade rails": "Five rails a side, the fixed stage's and four moving stages', each 12.9 in long and "
                     "0.3 in wide, from 3.4 to 4.9 in either side of the middle with the fixed stage outside, "
                     "1.5 in deep, 0.2 in behind the bite, and the carriage between the top stage's rails. The "
                     "fixed stage stands on the frame's top, so the robot starts 17.9 in tall, under SG1's 18 in. "
                     "Each stage travels 8.02 in on the one below and keeps 4.88 in on it, more than the "
                     "carriage's 3.0 in. Fully up the top stage's top is 17.9 + 4 x 8.02 = 49.98 in up, the "
                     "owner's 50 in, and the claw rises 5 x 8.02 = 40.1 in. Five stages because with four each "
                     "would travel 10.7 in and keep only 2.2 in on the one below.",
    "Mass layout": "Laid out in `MASSES`. VEX gives the masses of the eight 11 W motors, the battery and the "
                   "brain. Where each part sits, and the masses of the gears, rails, air parts and radio, are "
                   "guesses. The lift motors sit low on the frame, 3.0 in up, beside the drum they turn, and the "
                   "roller's motor low on the frame's back, 3.2 in up. The brain stands on its edge beside the "
                   "battery, 2.5 in up. The frame takes the 2.49 kg the rest leave of 6.35 kg, split between its "
                   "rails, 0.4 to 2.4 in up, and its deck, 4.8 to 5.0 in up, so the robot's CG is the owner's "
                   "3.5 in up: 89% on the rails. With the roller's motor up by its axle and the brain lying on "
                   "the deck the frame would have to sit 0.9 in up, under its own rails.",
    "Stage masses": "0.12 kg for each of the four moving stages and for the fixed one: two 12.9 in aluminium "
                    "rails of about 0.04 kg each, bearings and a cross brace. With 0.28 kg stages no split of "
                    "the frame put the CG as low as 3.5 in. 0.50 kg for the carriage with the claw, its piston "
                    "and its plate, of which each jaw is 0.08 kg.",
    "Drive layout": "Wheels 11.5 in apart side to side and 10.5 in front to back, as on the 3.25 in drive "
                    "before; the blue cartridges turn a 36-tooth gear driving a 48-tooth one on each wheel.",
    "Tracking wheel": "2 in across, pressed down with 4 N. Its rollers grip like a 2.75 in omni's.",
    "Slide friction": "0.2 between each stage and the one it slides in. Each slide's bearings carry the "
                      "sideways force and the twist the stages and claw above it put on them; the twist "
                      "presses the two ends of the slide, 12.9 in minus the travel apart (3 in apart on the "
                      "carriage), against the rails.",
    "Drum and gears": "The three motors' green (200 rpm) cartridges turn a 48-tooth gear driving a 72-tooth "
                      "gear on the drum (1.5:1), and the drum is 1.0 in across, so it winds in 3.14 in of "
                      "string a turn. Worked out from the owner's 1.2 s: at 12 V the lift goes from fully down "
                      "to fully up in 1.17 s empty. 48:60 (1.25:1) took 0.98 s and 36:60 (1.67:1) 1.30 s. 90% "
                      "of the motors' torque reaches the first stage. The drum weighs 20 g.",
    "String": "200 lb braided Dyneema: it breaks at 890 N after stretching 3.5%. Each run, the drum's and each "
              "stage's rigging, counts as long as a stage, 12.9 in, which makes 77,500 N per metre of "
              "stretch. The rigging runs both ways, up and down, so a stage that runs ahead is pulled back as "
              "hard. Each string is damped at 10% of what would stop what it moves from bouncing on it: the "
              "stages over it, each counted by the square of how many times as far it moves.",
    "Rubber bands": "8.4 N up on the first stage with the lift down, falling to 5.6 N fully up: 23% and 15% of "
                    "the 36.3 N the moving stages weigh at the first stage. Through the 48:72 gears, coasting motors "
                    "let it creep down at about 0.4 in a second, and shorted motors at 0 V at about 0.13.",
    "Hard stops": "Each stage stops at either end of its travel, and each jaw at open and closed, as hard as "
                  "a contact between field pieces (pieces.CONTACT_TIME).",
    "Claw": "V-shaped jaws from the back plate to 0.9 in ahead of the claw's middle; a held Cup touches the "
            "front side 0.81 in ahead. Each side of the V is 35 degrees off the jaw's length. Each jaw is three "
            "layers: a bottom band 0.6 in tall whose top is 0.54 in under the claw's middle, a recess cut back "
            "0.25 in, and a top band 0.6 in tall from 0.43 in over the claw's middle. Sized from the CAD: on a "
            "floor stack the Cup starts to flare 0.49 in under the Pin's middle (its flange's middle), and the "
            "Pin's flange and the fillet over it end 0.38 in over it, so each band meets only a straight taper, "
            "0.05 in clear of the ridge. There the Cup is 1.405 in round and the Pin 1.162 in, so the top "
            "band's V is 0.297 in further in than the bottom band's at its deepest point and both touch a "
            "stack at once. The recess's face is 0.07 in clear of the ridge, which is 1.58 in round. The bottom "
            "band's V is 4.4 in apart at its deepest point open and 3.1 in closed. Open, its front ends are "
            "3.14 in apart, 0.17 in wider than a Cup on each side, and the top band's 2.54 in, 0.11 in wider "
            "than a Pin. Closed, the top band's V is 2.51 in apart, narrower than a Pin's flat side there. "
            "With the lift down the claw's middle is 6.62 in up in the robot's frame; the robot sits 0.036 in "
            "lower on its tires and the foam, so the claw's middle is 6.58 in up, level with a floor stack's "
            "flange. A back plate 0.25 in thick and 5.2 in wide, 1.63 in behind the claw's middle (the ridge "
            "is 1.58 in round), reaches from 0.5 in above the jaws' top to 0.25 in off the floor with the lift "
            "down. The carriage is 3.0 in tall from the jaws' bottom, as wide as the gap between the top "
            "stage's rails, and reaches from the back of the rails to the plate.",
    "Claw piston": "A 12 mm bore with a 4 mm rod and 25 mm of travel, pushing out to close the jaws through a "
                   "rack and pinion that moves each jaw 0.65 in: 59.0 N on each jaw at 100 psi closing, 52.5 N "
                   "opening, less as the tank empties. VEX's older kit sheet gives its cylinders a 10 mm bore, "
                   "which here gives 41.0 N a jaw; the owner wants the claw to grip slightly harder than the "
                   "48.5 N a jaw the sim had, so the bore is raised to give 22% more. Its seals rub with 3 N. "
                   "Air flows from the tank into one side and out of the other to the room through the valve's "
                   "0.4 mm port, which lets the piston reach 0.43 m/s closing: the claw closes in 130 ms and "
                   "opens in 160 ms. The pads' contacts settle in 5 ms, which on a held stack lets them sink 4 "
                   "to 15 thou, the owner's 0.01 in. A pad twisting on what it holds rubs over 1 mm.",
    "Roller placement": "The axle is 13.5 in up and 1.75 in in from the frame's back, on two posts 4.5 in either "
                        "side of the middle that stand on the frame. Each roller's middle is 5.5 in from the "
                        "robot's middle. With the robot's back on the wall at a Toggle, the axle is 3.0 in from "
                        "the Toggle's (1.75 + 1.25 in back, 0.3 in up), so a flap's tip, 2.5 in out, reaches 0.68 "
                        "in inside the bar's corners (1.19 in out) and the channel's corners, 1.52 in out, clear "
                        "them by 0.30 in. The tips reach 0.75 in behind the frame, and over the wall's top (11.54 "
                        "in up) they stay 0.17 in above it.",
    "Flaps": "1/8 in thick rubber, 1 in out past the channel's end and 0.5 in wide, 1.2 g. A flap bends at its "
             "root as a sprung, damped hinge: 3 N on its tip bends it 30 degrees, damped at 30% of what would "
             "stop it ringing, and it stops at 90 degrees. Friction 0.55 on the Toggle.",
    "Roller parts": "The channels are 1 in wide and 0.5 in deep, 20 g each; the axle 1/8 in square, 30 g; each "
                    "post 30 g. The motor turns the axle through a chain from a 12-tooth sprocket to a 36-tooth "
                    "one (40 g with the sprockets); 95% of its torque reaches the axle.",
    "Bumpers": "Drawn only, 1.0 in thick inside the frame's outline and as tall as the frame, so the frame "
               "and its bumpers fill the same space the frame collides with.",
    "Air": "One 200 mL tank at 100 psi, inside the two R25 allows. 8 in of tubing, 2.5 mm inside, from the "
           "valve to each side of the piston: 1.0 mL. With these tubes the older kit's numbers give 46 strokes "
           "from 100 to 25 psi against VEX's 45; without them 58. The cascade's claw gets 77 strokes from "
           "100 to 25 psi. No leaks, and the air is the same temperature after a stroke as before it.",
}


@dataclass(frozen=True)
class CascadeGeometry:
    """Inches, degrees and kilograms.  Every number is in `GUESSES`, `CHOSEN` or `SOURCES`."""
    frame_bottom: float = 0.4
    frame_top: float = 5.0
    bite_seat_back: float = of.LOADER_DEPTH / INCH / 2.0
    bite_radius: float = 3.36
    bite_mouth_half_width: float = 4.5
    bite_arc_pieces: int = 4
    rail_length: float = 12.9
    rail_inner: float = 3.4
    rail_outer: float = 4.9
    rail_depth: float = 1.5
    rail_gap: float = 0.2
    stages: int = 5
    stage_travel: float = 8.02
    lift_motor_count: int = 3
    lift_cartridge: str = "green"
    lift_gear_ratio: float = 1.5
    drum_diameter: float = 1.0
    string_break_n: float = 890.0
    string_stretch_at_break: float = 0.035
    string_damping_ratio: float = 0.1
    stage_mass_kg: float = 0.12
    carriage_mass_kg: float = 0.50
    back_plate_mass_kg: float = 0.08
    jaw_mass_kg: float = 0.08
    slide_mu: float = 0.2
    carriage_pad_spacing: float = 3.0
    lift_efficiency: float = 0.9
    bands_down_n: float = 8.4
    bands_up_n: float = 5.6
    claw_lowest: float = 6.62
    jaw_ahead: float = 0.9
    jaw_thickness: float = 0.35
    cup_band_top: float = 0.54
    pin_band_bottom: float = 0.43
    band_height: float = 0.6
    recess_depth: float = 0.25
    jaw_face_deg: float = 35.0
    jaw_open_gap: float = 4.4
    jaw_closed_gap: float = 3.1
    jaw_mu: float = 0.9
    jaw_pad_contact_ms: float = 5.0
    jaw_twist_reach_mm: float = 1.0
    air_tank_ml: float = 200.0
    start_psi: float = 100.0
    piston_bore_mm: float = 12.0
    piston_rod_mm: float = 4.0
    piston_travel_mm: float = 25.0
    piston_seal_friction_n: float = 3.0
    valve_tube_in: float = 8.0
    valve_tube_bore_mm: float = 2.5
    valve_port_mm: float = 0.4
    back_plate_behind_claw: float = 1.63
    back_plate_thickness: float = 0.25
    back_plate_half_width: float = 2.6
    back_plate_floor_gap: float = 0.25
    back_plate_above_jaws: float = 0.5
    roller_axle_height: float = 13.5
    roller_axle_in_from_back: float = 1.75
    roller_middle_from_center: float = 5.5
    roller_cartridge: str = "green"
    roller_ratio: float = 3.0
    roller_gear_efficiency: float = 0.95
    roller_channel_length: float = 3.0
    roller_channel_width: float = 1.0
    roller_channel_depth: float = 0.5
    roller_channel_mass_kg: float = 0.02
    roller_axle_size: float = 0.125
    roller_axle_mass_kg: float = 0.03
    roller_post_from_center: float = 4.5
    roller_post_size: float = 0.5
    flap_length: float = 1.0
    flap_width: float = 0.5
    flap_thickness: float = 0.125
    flap_mass_kg: float = 0.0012
    flap_mu: float = 0.55
    flap_bend_force_n: float = 3.0
    flap_bend_deg: float = 30.0
    flap_damping_ratio: float = 0.3
    bumper_thickness: float = 1.0

    @property
    def grip_lowest(self) -> float:
        """The claw's middle with the lift down: level with the flange of a stack standing on the floor."""
        return self.claw_lowest

    @property
    def cup_middle_under_claw(self) -> float:
        """How far the middle of a Cup the claw takes from the floor hangs below the claw's middle."""
        return self.claw_lowest - CUP_HALF_IN

    @property
    def jaw_bottom(self) -> float:
        """How far below the claw's middle the jaws reach."""
        return self.cup_band_top + self.band_height

    @property
    def jaw_top(self) -> float:
        """How far above the claw's middle the jaws reach."""
        return self.pin_band_bottom + self.band_height

    @property
    def cup_band_radius(self) -> float:
        """A held stack's Cup's radius level with the top edge of the bottom band, where the band presses on it."""
        return cad.cup().radius_at((self.cup_middle_under_claw - self.cup_band_top) * INCH) / INCH

    @property
    def pin_band_radius(self) -> float:
        """A held stack's Pin's radius level with the bottom edge of the top band, where the band presses on it."""
        return cad.pin().radius_at(self.pin_band_bottom * INCH) / INCH

    @property
    def gap_per_radius(self) -> float:
        """How much wider apart the V's deepest points are than a round piece they both just touch."""
        return 1.0 / math.cos(math.radians(self.jaw_face_deg))

    @property
    def pin_band_reach(self) -> float:
        """How much nearer the middle the top band's V is than the bottom band's, at the V's deepest point,
        so on a stack both bands touch at once."""
        return (self.cup_band_radius - self.pin_band_radius) * self.gap_per_radius

    @property
    def recess_reach(self) -> float:
        """How much further from the middle the recess's V is than the bottom band's, at the V's deepest point."""
        return self.recess_depth * self.gap_per_radius

    @property
    def lift_travel(self) -> float:
        """How far the claw rises from fully down to fully up."""
        return self.stages * self.stage_travel

    @property
    def grip_highest(self) -> float:
        return self.grip_lowest + self.lift_travel

    @property
    def flap_stiffness(self) -> float:
        """N m per radian a flap's root resists bending with."""
        return self.flap_bend_force_n * self.flap_length * INCH / math.radians(self.flap_bend_deg)

    @property
    def flap_damping(self) -> float:
        """N m s per radian at a flap's root."""
        swing_inertia = self.flap_mass_kg * (self.flap_length * INCH) ** 2 / 3.0
        return 2.0 * self.flap_damping_ratio * math.sqrt(self.flap_stiffness * swing_inertia)

    @property
    def flap_tip_reach(self) -> float:
        """How far a flap's tip is from the roller's axle."""
        return self.roller_channel_length / 2.0 + self.flap_length

    @property
    def string_per_motor_rad(self) -> float:
        """Metres of string the drum winds in for each radian the lift motors turn."""
        return self.drum_diameter / 2.0 * INCH / self.lift_gear_ratio

    @property
    def string_stiffness(self) -> float:
        """N per metre a run of string as long as a stage pulls when stretched."""
        return self.string_break_n / self.string_stretch_at_break / (self.rail_length * INCH)

    @property
    def lift_full_deg(self) -> float:
        """Lift motor degrees from fully down to fully up."""
        return self.stage_travel * INCH / self.string_per_motor_rad / DEG

    @property
    def bite_back(self) -> float:
        """How far the bite reaches into the frame from its front."""
        return self.bite_seat_back + self.bite_radius

    @property
    def rail_front_back(self) -> float:
        """How far behind the frame front the rails' front faces are."""
        return self.bite_back + self.rail_gap

    @property
    def rail_top(self) -> float:
        """The top of the fixed stage."""
        return self.frame_top + self.rail_length

    @property
    def rail_width(self) -> float:
        """How wide each stage's rail is."""
        return (self.rail_outer - self.rail_inner) / self.stages

    @property
    def claw_lever(self) -> float:
        """How far ahead of the rails' middle the claw hangs."""
        return self.rail_front_back + self.rail_depth / 2.0 - self.bite_seat_back

    @property
    def jaw_travel(self) -> float:
        """How far each jaw moves from open to closed."""
        return (self.jaw_open_gap - self.jaw_closed_gap) / 2.0

    @property
    def jaw_per_piston(self) -> float:
        """The force on each jaw per newton on the piston: the piston's
        travel over both jaws' travel."""
        return self.piston_travel_mm / (2.0 * self.jaw_travel * INCH * 1000.0)

    def lift_deg_for_grip(self, grip_height: float) -> float:
        """Lift motor degrees that put the claw's middle `grip_height` inches up."""
        rise = grip_height - self.grip_lowest
        if not 0.0 <= rise <= self.lift_travel:
            raise ValueError(f"the claw reaches {self.grip_lowest:.2f} to {self.grip_highest:.2f} in, "
                             f"not {grip_height:.2f} in")
        return rise * INCH / self.stages / self.string_per_motor_rad / DEG


GEOMETRY = CascadeGeometry()

LIFT_CLEARANCE_IN = 0.25
#: Where Pins stand, from where the CAD's shapes meet (`override_cad.seats`)
NEUTRAL_PIN_MIDDLE_IN = (of.GOAL_NEUTRAL_RISER + cad.seats().pin_in_goal) / INCH
CENTER_PIN_MIDDLE_IN = (of.GOAL_CENTER_RISER + cad.seats().pin_in_goal) / INCH
#: From a Pin's middle to the middle of the Pin in the Cup set over it
STACK_LAYER_IN = (cad.seats().cup_on_pin + cad.seats().pin_in_cup) / INCH


def claw_on(pin_middle: float) -> float:
    """The claw's middle, inches up, that sets a held Cup down over a Pin whose middle is `pin_middle` up."""
    return pin_middle + cad.seats().cup_on_pin / INCH + GEOMETRY.cup_middle_under_claw


def claw_over(pin_middle: float) -> float:
    """The claw's middle that carries a held Cup just over the tip of a Pin whose middle is `pin_middle` up."""
    return pin_middle + PIN_HALF_IN + LIFT_CLEARANCE_IN + CUP_HALF_IN + GEOMETRY.cup_middle_under_claw


#: To 0.01 in, the precision the planner's presets carry
LIFT_HEIGHTS: Dict[str, float] = {name: round(height, 2) for name, height in {
    "down": GEOMETRY.grip_lowest,
    "over a neutral goal": claw_over(NEUTRAL_PIN_MIDDLE_IN),
    "on a neutral goal": claw_on(NEUTRAL_PIN_MIDDLE_IN),
    "over a stack on a neutral goal": claw_over(NEUTRAL_PIN_MIDDLE_IN + STACK_LAYER_IN),
    "on a stack on a neutral goal": claw_on(NEUTRAL_PIN_MIDDLE_IN + STACK_LAYER_IN),
    "over the center goal": claw_over(CENTER_PIN_MIDDLE_IN),
    "on the center goal": claw_on(CENTER_PIN_MIDDLE_IN),
    "top": GEOMETRY.grip_highest,
}.items()}


# ----------------------------------------------------------------- mass

@dataclass(frozen=True)
class Mass:
    """A part of the robot that does not move on it: kg, and a box from its middle (x forward, y left, z up,
    inches from the middle of the drivetrain on the floor) out `half` either way."""
    name: str
    kg: float
    center: Tuple[float, float, float]
    half: Tuple[float, float, float]


def _sides(name: str, kg: float, x: float, y: float, z: float, half) -> List[Mass]:
    """The same part on the left and on the right."""
    return [Mass(f"{name} {side_name}", kg, (x, side * y, z), half) for side_name, side in (("left", 1.0), ("right", -1.0))]


def _corners(name: str, kg: float, x: float, y: float, z: float, half) -> List[Mass]:
    """The same part at each corner."""
    return [mass for end_name, end in (("front", 1.0), ("back", -1.0))
            for mass in _sides(f"{name} {end_name}", kg, end * x, y, z, half)]


#: Where the parts that do not ride the lift sit.  The frame is whatever
#: mass the rest leave of the robot's, split between its rails and its deck.
MASSES: Tuple[Mass, ...] = tuple(
    _corners("drive motor", 0.155, 4.0, 4.3, 1.8, (1.1, 0.65, 1.1))
    + _corners("drive gears", 0.06, 5.25, 4.9, 1.6, (1.0, 0.3, 1.0))
    + _sides("lift motor", 0.155, -0.3, 2.3, DRUM_AT[1], (0.65, 1.1, 1.1))
    + [Mass("lift motor middle", 0.155, (-0.3, 0.0, DRUM_AT[1]), (0.65, 1.1, 1.1))]
    + [Mass("battery", 0.35, (-5.0, 0.0, 2.3), (2.75, 1.3, 0.8)),
       Mass("brain", 0.285, (-5.0, 2.6, 2.5), (2.25, 0.5, 1.5)),
       Mass("air tank and valve", 0.15, (-3.0, -2.8, 1.4), (2.75, 0.8, 0.8)),
       Mass("radio", 0.05, (-6.5, 4.0, 5.4), (0.6, 0.6, 0.4))]
    + _sides("fixed stage", 0.06, 1.32, 4.75, 11.45, (0.75, 0.15, 6.45))
    + _sides("roller post", 0.03, -5.75, 4.5, 9.22, (0.25, 0.25, 4.22))
    + [Mass("roller motor", 0.155, (-6.3, -3.3, 3.2), (1.1, 0.65, 1.1)),
       Mass("roller chain and sprockets", 0.04, (-5.75, -3.95, 8.35), (0.8, 0.15, 5.15))])
#: How the frame's rails share what of its mass is not on the deck: the
#: sides, the back, and the front beside the bite, 0.4 to 2.4 in up
FRAME_RAILS: Tuple[Tuple[str, float, Tuple[float, float, float], Tuple[float, float, float]], ...] = (
    ("frame side left", 0.3, (0.0, 5.75, 1.4), (7.5, 0.5, 1.0)),
    ("frame side right", 0.3, (0.0, -5.75, 1.4), (7.5, 0.5, 1.0)),
    ("frame back", 0.2, (-7.0, 0.0, 1.4), (0.5, 5.25, 1.0)),
    ("frame front left", 0.1, (5.0, 5.0, 1.4), (2.5, 1.25, 1.0)),
    ("frame front right", 0.1, (5.0, -5.0, 1.4), (2.5, 1.25, 1.0)),
)
FRAME_DECK = ("frame deck", (-2.0, 0.0, 4.9), (5.5, 5.25, 0.1))


def moving_mass(g: CascadeGeometry) -> float:
    return (g.stages - 1) * g.stage_mass_kg + g.carriage_mass_kg


def stage_masses(g: CascadeGeometry) -> List[float]:
    """kg of each moving stage, the carriage last."""
    return [g.stage_mass_kg] * (g.stages - 1) + [g.carriage_mass_kg]


def mass_on_string(g: CascadeGeometry, above: int) -> float:
    """The mass a string pulling stage `above` + 1 up on stage `above` moves
    as it stretches: each stage over it moves as many times as far as it is
    stages higher, so it counts that many times squared."""
    return sum((k - above) ** 2 * kg for k, kg in enumerate(stage_masses(g), start=1) if k > above)


def lumps(chassis: Chassis, riding: List[Tuple[float, float]]) -> List[Lump]:
    """The robot's mass that does not ride the lift, for `mujoco_drive.add_drivetrain`.
    `riding` is (kg, metres up) of each piece on the lift, with the lift down.
    The frame takes the mass the rest leave of the robot's, split between its
    rails and its deck so the robot's CG is the chassis's `cg_height` up."""
    to_m = lambda values: tuple(v * INCH for v in values)  # noqa: E731
    out = [Lump(m.name, m.kg, to_m(m.center), to_m(m.half)) for m in MASSES]
    placed = ([(m.kg, m.center[2] * INCH) for m in MASSES] + riding + [(DRUM_MASS_KG, DRUM_AT[1] * INCH)]
              + [(u.wheel.mass, u.radius) for u in chassis.wheel_units])
    frame = chassis.mass - sum(kg for kg, _ in placed)
    if frame <= 0.0:
        raise ValueError(f"the parts weigh more than the robot's {chassis.mass:.2f} kg")
    rails_z, deck_z = FRAME_RAILS[0][2][2] * INCH, FRAME_DECK[1][2] * INCH
    frame_z = (chassis.mass * chassis.cg_height - sum(kg * z for kg, z in placed)) / frame
    on_deck = (frame_z - rails_z) / (deck_z - rails_z)
    if not 0.0 <= on_deck <= 1.0:
        raise ValueError(f"the frame's {frame:.2f} kg would have to sit {frame_z / INCH:.2f} in up to put the "
                         f"robot's CG {chassis.cg_height / INCH:.2f} in up, outside its rails and deck "
                         f"({rails_z / INCH:.1f} to {deck_z / INCH:.1f} in)")
    out += [Lump(name, frame * (1.0 - on_deck) * share, to_m(center), to_m(half))
            for name, share, center, half in FRAME_RAILS]
    name, center, half = FRAME_DECK
    out.append(Lump(name, frame * on_deck, to_m(center), to_m(half)))
    return out


# ----------------------------------------------------------------- the claw's air

def circle_mm2(diameter_mm: float) -> float:
    return math.pi * diameter_mm ** 2 / 4.0


def psi_after_stroke(psi: float, tank_ml: float, filled_ml: float) -> float:
    """The tank's gauge pressure after it fills `filled_ml` of air at room pressure up to its own."""
    return psi * tank_ml / (tank_ml + filled_ml)


ROOM_PA = 101325.0
AIR_J_PER_KG_K = 287.05
ROOM_K = 293.15
AIR_HEAT_RATIO = 1.4
#: Below this ratio of downstream to upstream pressure the flow through a port is choked
CHOKED_RATIO = (2.0 / (AIR_HEAT_RATIO + 1.0)) ** (AIR_HEAT_RATIO / (AIR_HEAT_RATIO - 1.0))
FLOW_STEPS_PER_STEP = 10


def port_flow(upstream_pa: float, downstream_pa: float, port_m2: float) -> float:
    """kg/s of room-temperature air through a port from `upstream_pa` to `downstream_pa` (negative: back)."""
    if upstream_pa < downstream_pa:
        return -port_flow(downstream_pa, upstream_pa, port_m2)
    k, r = AIR_HEAT_RATIO, downstream_pa / upstream_pa
    if r <= CHOKED_RATIO:
        return port_m2 * upstream_pa * math.sqrt(k / (AIR_J_PER_KG_K * ROOM_K)
                                                 * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0)))
    return port_m2 * upstream_pa * math.sqrt(2.0 * k / ((k - 1.0) * AIR_J_PER_KG_K * ROOM_K)
                                             * (r ** (2.0 / k) - r ** ((k + 1.0) / k)))


def chamber_after(pa: float, volume: float, growth: float, source_pa: float, port_m2: float, seconds: float) -> float:
    """A cylinder chamber's pressure after `seconds` joined to `source_pa`
    through the valve's port while it grows `growth` m^3/s, the air staying
    at room temperature.  The air through the port takes it no further than
    the source's pressure."""
    filled = AIR_J_PER_KG_K * ROOM_K * port_flow(source_pa, pa, port_m2) / volume * seconds
    after = pa + filled if (source_pa - pa - filled) * (source_pa - pa) > 0.0 else source_pa
    return after - pa * growth / volume * seconds


class PneumaticClaw:
    """The valve and the air: which side of the piston the tank feeds, the
    tank's pressure, which drops each time the valve switches, and the
    pressure on each side of the piston as air flows through the valve's port."""

    def __init__(self, g: CascadeGeometry):
        self.g = g
        self.air_psi = g.start_psi
        self.hold(False)

    def hold(self, closed: bool) -> None:
        """Set the valve to `closed` with the piston's air already settled, using none from the tank."""
        self.closed = closed
        self.closing_pa, self.opening_pa = (self.tank_pa, ROOM_PA) if closed else (ROOM_PA, self.tank_pa)

    @property
    def tank_pa(self) -> float:
        return ROOM_PA + self.air_psi * PASCALS_PER_PSI

    @property
    def tube_ml(self) -> float:
        g = self.g
        return circle_mm2(g.valve_tube_bore_mm) * g.valve_tube_in * 25.4 / 1000.0

    @property
    def closing_ml(self) -> float:
        g = self.g
        return circle_mm2(g.piston_bore_mm) * g.piston_travel_mm / 1000.0 + self.tube_ml

    @property
    def opening_ml(self) -> float:
        g = self.g
        return self.ring_mm2 * g.piston_travel_mm / 1000.0 + self.tube_ml

    @property
    def ring_mm2(self) -> float:
        return circle_mm2(self.g.piston_bore_mm) - circle_mm2(self.g.piston_rod_mm)

    def command(self, closed: bool) -> None:
        if closed != self.closed:
            filled = self.closing_ml if closed else self.opening_ml
            self.air_psi = psi_after_stroke(self.air_psi, self.g.air_tank_ml, filled)
        self.closed = closed

    def flow(self, seconds: float, travel: float, speed: float) -> None:
        """Let air through the valve for `seconds`, the piston `travel`
        metres out from open and moving out at `speed`: the tank feeds one
        side and the other empties to the room."""
        g = self.g
        bore, ring = circle_mm2(g.piston_bore_mm) * 1e-6, self.ring_mm2 * 1e-6
        tube = self.tube_ml * 1e-6
        port = circle_mm2(g.valve_port_mm) * 1e-6
        closing_volume = tube + bore * travel
        opening_volume = tube + ring * (g.piston_travel_mm / 1000.0 - travel)
        closing_source, opening_source = (self.tank_pa, ROOM_PA) if self.closed else (ROOM_PA, self.tank_pa)
        step = seconds / FLOW_STEPS_PER_STEP
        for _ in range(FLOW_STEPS_PER_STEP):
            self.closing_pa = chamber_after(self.closing_pa, closing_volume, bore * speed, closing_source, port, step)
            self.opening_pa = chamber_after(self.opening_pa, opening_volume, -ring * speed, opening_source, port, step)

    def piston_force(self) -> float:
        """Newtons on the piston toward closing the jaws (negative: opening them)."""
        g = self.g
        return ((self.closing_pa - ROOM_PA) * circle_mm2(g.piston_bore_mm)
                - (self.opening_pa - ROOM_PA) * self.ring_mm2) * 1e-6


# ----------------------------------------------------------------- shapes

LIFT_PARTS = ("claw", "carriage")
FRAME_BLOCK = "frame behind the bite"
#: Each jaw's layers from the bottom up, and the two sides of the V in each
JAW_LAYERS = ("cup band", "recess", "pin band")
JAW_PIECES = ("back", "front")
ROLLER = "roller"
FLAP_ENDS = ("back", "front")


@dataclass(frozen=True)
class Hinge:
    """An axis across the robot (along y) that pieces turn on: its name, and
    where it passes through the robot's side view with everything at rest."""
    name: str
    forward: float
    up: float


@dataclass(frozen=True)
class Component:
    """One solid piece of the robot, posed with the lift down, the claw open
    and the roller and its flaps at rest.  It is fixed to, or slides on, the
    pieces keyed in `parents`.  It rises `rise_share` of what the claw rises,
    a jaw (`jaw_side` 1 on the left, -1 on the right) slides toward the
    middle as the claw closes, and a roller piece turns on each of its
    `hinges`, the axle's first.  A `padded` piece is faced with the claw's
    rubber pads."""
    key: str
    part: Part
    parents: Tuple[str, ...] = ()
    rise_share: float = 0.0
    jaw_side: int = 0
    padded: bool = False
    hinges: Tuple[Hinge, ...] = ()

    def stage(self, stages: int) -> int:
        """Which of `stages` moving stages the piece rides on: 0 for the frame, `stages` for the carriage."""
        return round(self.rise_share * stages)

    def posed(self, rise: float, closing: float, rise_speed: float, turns: Dict[str, float]) -> Part:
        """The piece with the stage it is on `rise` up, rising at `rise_speed`,
        each jaw `closing` in from open (metres), and each hinge it is on
        turned as `turns` names (radians; positive takes what is behind the
        hinge up)."""
        part = self.part.shifted(rise, -self.jaw_side * closing, rise_speed)
        if not self.hinges:
            return part
        side_view = list(part.profile)
        for hinge in reversed(self.hinges):
            side_view = _turned(side_view, hinge, turns[hinge.name])
        return profile_part(part.name, side_view, part.right, part.left)


def _turned(side_view: List[Tuple[float, float]], hinge: Hinge, angle: float) -> List[Tuple[float, float]]:
    """Side-view points turned `angle` about `hinge`: positive takes what is behind it up."""
    c, s = math.cos(angle), math.sin(angle)
    return [(hinge.forward + (x - hinge.forward) * c + (z - hinge.up) * s,
             hinge.up - (x - hinge.forward) * s + (z - hinge.up) * c) for x, z in side_view]


def _inset(outline, distance: float) -> List[Tuple[float, float]]:
    """A counter-clockwise outline with every edge moved `distance` inward."""
    lines = []
    for (xa, ya), (xb, yb) in zip(outline, outline[1:] + outline[:1]):
        length = math.hypot(xb - xa, yb - ya)
        nx, ny = -(yb - ya) / length, (xb - xa) / length
        lines.append(((xa + nx * distance, ya + ny * distance), ((xb - xa) / length, (yb - ya) / length)))
    corners = []
    for (pa, da), (pb, db) in zip(lines[-1:] + lines[:-1], lines):
        cross = da[0] * db[1] - da[1] * db[0]
        t = ((pb[0] - pa[0]) * db[1] - (pb[1] - pa[1]) * db[0]) / cross
        corners.append((pa[0] + da[0] * t, pa[1] + da[1] * t))
    return corners


def _mirror(outline) -> List[Tuple[float, float]]:
    """A counter-clockwise outline mirrored across the middle, still counter-clockwise."""
    return [(x, -y) for x, y in reversed(outline)]


def body_name(component: Component, g: CascadeGeometry) -> str:
    """The MuJoCo body a component is on."""
    if component.hinges:
        return f"{ROBOT} {component.hinges[-1].name}"
    if component.jaw_side:
        return f"{ROBOT} jaw {'left' if component.jaw_side > 0 else 'right'}"
    stage = component.stage(g.stages)
    if stage == 0:
        return ROBOT
    return f"{ROBOT} carriage" if stage == g.stages else f"{ROBOT} stage {stage}"


def geom_name(component: Component) -> str:
    """A robot geom's name: the robot, what kind of part it is and which, between "|"s."""
    return f"{ROBOT}|{component.part.name}|{component.key}"


def part_of(geom_name_: str) -> str:
    """What kind of part a robot geom is, from its name."""
    return geom_name_.split("|")[1]


class CascadeRobot:
    """The bite, the cascade lift, the claw and the Toggle roller on a drivetrain."""

    def __init__(self, geometry: CascadeGeometry = GEOMETRY):
        self.geometry = geometry
        self.claw = PneumaticClaw(geometry)
        self.lift_motors = [Motor(cartridge=geometry.lift_cartridge, name=f"lift {k + 1}")
                            for k in range(geometry.lift_motor_count)]
        self.roller_motor = Motor(cartridge=geometry.roller_cartridge, name="roller")
        self.mechanism: Optional[CascadeMechanism] = None
        self._components: Dict[Tuple[float, float], List[Component]] = {}

    # ------------------------------------------------------------- MuJoCo

    def model_key(self, chassis: Chassis) -> tuple:
        """What the robot's bodies are built from, to tell one build from another."""
        units = tuple((u.name, u.x, u.y, u.radius, u.wheel.mass, u.wheel.inertia, u.steer_angle, u.preload,
                       u.gear_ratio, tuple(m.output_inertia for m in u.motors)) for u in chassis.wheel_units)
        return (self.geometry, chassis.mass, chassis.length, chassis.width, units)

    def build(self, spec: mujoco.MjSpec, chassis: Chassis) -> None:
        """The robot's bodies, joints and shapes, with the lift down, the claw
        open and the roller at rest."""
        g = self.geometry
        masses = self._component_masses(chassis)
        shaped = [(masses[c.key], (c.part.bottom + c.part.top) / 2.0) for c in self.components(chassis)
                  if c.key in masses]
        bodies = {ROBOT: add_drivetrain(spec, chassis, lumps(chassis, shaped), ROBOT)}
        drum = bodies[ROBOT].add_body(name=DRUM, pos=[DRUM_AT[0] * INCH, 0.0, DRUM_AT[1] * INCH])
        drum.explicitinertial = True
        drum.mass = DRUM_MASS_KG
        drum.inertia = [DRUM_MASS_KG * (g.drum_diameter * INCH / 2.0) ** 2 / 2.0] * 3
        drum_turn = drum.add_joint(name=f"{DRUM} turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0])
        drum_turn.armature = sum(m.output_inertia for m in self.lift_motors) * g.lift_gear_ratio ** 2
        parent = bodies[ROBOT]
        slides = []
        for stage in range(1, g.stages + 1):
            name = f"{ROBOT} carriage" if stage == g.stages else f"{ROBOT} stage {stage}"
            body = parent.add_body(name=name)
            slide = body.add_joint(name=f"{name} slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0],
                                   range=[0.0, g.stage_travel * INCH])
            _stiff_stops(slide)
            slides.append(f"{name} slide")
            bodies[name] = parent = body
        drum_rise_per_rad = g.string_per_motor_rad * g.lift_gear_ratio
        self._string(spec, f"{DRUM} string", [(slides[0], 1.0), (drum_turn.name, -drum_rise_per_rad)],
                     mass_on_string(g, 0))
        for k, (below, above) in enumerate(zip(slides, slides[1:]), start=1):
            self._string(spec, f"{above} rigging", [(above, 1.0), (below, -1.0)], mass_on_string(g, k))
        for side_name, side in (("left", 1.0), ("right", -1.0)):
            name = f"{ROBOT} jaw {side_name}"
            jaw = parent.add_body(name=name)
            slide = jaw.add_joint(name=f"{name} slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, -side, 0.0],
                                  range=[0.0, g.jaw_travel * INCH])
            _stiff_stops(slide)
            slide.frictionloss = g.piston_seal_friction_n * g.jaw_per_piston
            bodies[name] = jaw
        _same_travel(spec, f"{ROBOT} jaw right slide", f"{ROBOT} jaw left slide")
        self._add_hinges(bodies, self.components(chassis))
        for component in self.components(chassis):
            self._add_shape(spec, bodies[body_name(component, g)], component, masses.get(component.key, 0.0))

    def _add_hinges(self, bodies: dict, components: List[Component]) -> None:
        """A body for each hinge the roller's pieces turn on: the axle, on
        the robot, turned by the motor through its gears, and each flap's
        root, on the axle, sprung and damped like the rubber it is."""
        g = self.geometry
        for component in components:
            parent = bodies[ROBOT]
            for hinge in component.hinges:
                name = f"{ROBOT} {hinge.name}"
                if name not in bodies:
                    body = parent.add_body(name=name)
                    joint = body.add_joint(name=f"{name} hinge", type=mujoco.mjtJoint.mjJNT_HINGE,
                                           axis=[0.0, 1.0, 0.0], pos=[hinge.forward, 0.0, hinge.up])
                    if hinge.name == ROLLER:
                        joint.armature = self.roller_motor.output_inertia * g.roller_ratio ** 2
                    else:
                        joint.stiffness = [g.flap_stiffness, 0.0, 0.0]
                        joint.damping = [g.flap_damping, 0.0, 0.0]
                        joint.range = [-math.pi / 2.0, math.pi / 2.0]
                        _stiff_stops(joint)
                    bodies[name] = body
                parent = bodies[name]

    def _string(self, spec: mujoco.MjSpec, name: str, joints: List[Tuple[str, float]], carried_kg: float) -> None:
        """A string, rigged both ways, whose stretch is the weighted sum of the
        joints' positions: a spring with `string_damping_ratio` of the damping
        that would stop the `carried_kg` hanging on it from bouncing."""
        g = self.geometry
        tendon = spec.add_tendon(name=name)
        tendon.stiffness = [g.string_stiffness, 0.0, 0.0]
        tendon.damping = [2.0 * g.string_damping_ratio * math.sqrt(g.string_stiffness * carried_kg), 0.0, 0.0]
        tendon.springlength = [0.0, 0.0]
        for joint, coefficient in joints:
            tendon.wrap_joint(joint, coefficient)

    def _component_masses(self, chassis: Chassis) -> Dict[str, float]:
        """kg carried by each moving piece's shape; the drivetrain's are in `MASSES`."""
        g = self.geometry
        out: Dict[str, float] = {}
        for component in self.components(chassis):
            stage = component.stage(g.stages)
            if component.jaw_side:
                out[component.key] = g.jaw_mass_kg / (len(JAW_LAYERS) * len(JAW_PIECES))
            elif component.part.name == "roller flap":
                out[component.key] = g.flap_mass_kg
            elif component.part.name == "roller":
                out[component.key] = g.roller_channel_mass_kg
            elif component.part.name == "roller axle":
                out[component.key] = g.roller_axle_mass_kg
            elif component.key == "back plate":
                out[component.key] = g.back_plate_mass_kg
            elif component.key == "carriage":
                out[component.key] = g.carriage_mass_kg - g.back_plate_mass_kg - 2.0 * g.jaw_mass_kg
            elif 0 < stage < g.stages:
                out[component.key] = g.stage_mass_kg / 2.0
        return out

    def _add_shape(self, spec: mujoco.MjSpec, body, component: Component, kg: float) -> None:
        part = component.part
        if part.outline or part.profile:
            name = f"{ROBOT} shape {component.key}"
            spec.add_mesh(name=name, uservert=np.array(part.vertices()).flatten().tolist())
            geom = body.add_geom(name=geom_name(component), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
        else:
            geom = body.add_geom(name=geom_name(component), type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=[(part.front - part.back) / 2.0, (part.left - part.right) / 2.0,
                                       (part.top - part.bottom) / 2.0],
                                 pos=[(part.front + part.back) / 2.0, (part.left + part.right) / 2.0,
                                      (part.top + part.bottom) / 2.0])
        if kg > 0.0:
            geom.mass = kg
        else:
            geom.density = 0.0
        geom.contype = pieces.ROBOT_BIT
        geom.conaffinity = pieces.STATIC_BIT | pieces.FLAT_BIT | pieces.PIECE_BIT
        g = self.geometry
        if component.padded:
            pieces.contact_settings(geom, g.jaw_mu, condim=4, priority=2, spin=g.jaw_mu * g.jaw_twist_reach_mm / 1000.0,
                                    contact_time=g.jaw_pad_contact_ms / 1000.0)
        elif part.name == "roller flap":
            pieces.contact_settings(geom, g.flap_mu, priority=2)
        else:
            pieces.contact_settings(geom, pieces.ROBOT_MU, priority=1)

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, chassis: Chassis) -> "CascadeMechanism":
        """The lift, claw and roller moving the bodies `build` made in `model`."""
        back, front = self._rail_x(chassis)
        self.mechanism = CascadeMechanism(self, model, data, (back + front) / 2.0)
        return self.mechanism

    def _attached(self) -> "CascadeMechanism":
        if self.mechanism is None:
            raise ValueError("the robot is not in a MuJoCo world yet: make an OverrideWorld with it")
        return self.mechanism

    # ------------------------------------------------------------- reports

    def grip_height(self) -> float:
        return self.geometry.grip_lowest * INCH + self._attached().lift.rise

    def telemetry(self) -> dict:
        mech = self._attached()
        return dict(lift_deg=round(mech.lift.angle / DEG, 1), lift_height_in=round(self.grip_height() / INCH, 2),
                    lift_amps=round(mech.lift.current, 2), claw_closed=self.claw.closed,
                    claw_gap_in=round(mech.gap / INCH, 2), grip_force_n=round(mech.squeeze, 1),
                    air_psi=round(self.claw.air_psi, 1),
                    stage_rises_in=[round(r / INCH, 2) for r in mech.lift.stage_rises()],
                    roller_deg=round(mech.roller.angle / DEG, 1),
                    roller_rpm=round(mech.roller.speed * 60.0 / (2.0 * math.pi), 1),
                    roller_amps=round(abs(self.roller_motor.current), 2),
                    flap_bends_deg={name: round(angle / DEG, 1) for name, angle in mech.roller.turns().items()
                                    if name != ROLLER})

    def motor_watts(self) -> Dict[str, float]:
        return {"lift": sum(m.spec.peak_power_rating for m in self.lift_motors),
                "roller": self.roller_motor.spec.peak_power_rating}

    def extent(self, chassis: Chassis) -> Tuple[float, float, float]:
        """Length, width and height of everything on the robot, metres."""
        parts = self.parts(chassis)
        xs = [x for part in parts for x, _ in part.corners()]
        ys = [y for part in parts for _, y in part.corners()]
        return max(xs) - min(xs), max(ys) - min(ys), max(part.top for part in parts)

    def claw_forward(self, chassis: Chassis) -> float:
        """How far ahead of the middle of the drivetrain the claw's middle is, metres."""
        return chassis.length / 2.0 - self.geometry.bite_seat_back * INCH

    def preload_middle_under_claw(self) -> float:
        """How far below the claw's middle the preload's middle is, inches: its flange rests on the bottom bands."""
        return self.geometry.cup_band_top - PIN_FLANGE_HALF_IN

    def preload_pose(self, chassis: Chassis, preload: str, colors):
        """The preload hangs in the closed claw by its flange with the lift
        down, its `preload` colour's half up.  (middle, axis from `colors[0]`'s
        half to `colors[1]`'s, in the claw) in the robot frame."""
        if preload not in colors:
            raise ValueError(f"the preload is a {colors[0]}/{colors[1]} Pin; it has no {preload!r} half to point up")
        axis = of.UP if colors[1] == preload else of.DOWN
        middle = (self.geometry.grip_lowest - self.preload_middle_under_claw()) * INCH
        return (self.claw_forward(chassis), 0.0, middle), axis, True

    def preload_gap(self) -> float:
        """The jaws' gap, metres, at which the top bands just touch the preload where `preload_pose` has it."""
        g = self.geometry
        radius = cad.pin().radius_at((g.pin_band_bottom + self.preload_middle_under_claw()) * INCH) / INCH
        return (2.0 * radius * g.gap_per_radius + 2.0 * g.pin_band_reach) * INCH

    # ------------------------------------------------------------- shapes

    def _bite_inner_edge(self, chassis) -> List[Tuple[float, float]]:
        """The left side of the bite from its back to the frame front, robot frame, metres."""
        g = self.geometry
        hl = chassis.length / 2.0
        seat_x = hl - g.bite_seat_back * INCH
        radius = g.bite_radius * INCH
        points = []
        for k in range(g.bite_arc_pieces + 1):
            angle = math.pi - 0.5 * math.pi * k / g.bite_arc_pieces
            points.append((seat_x + radius * math.cos(angle), radius * math.sin(angle)))
        points.append((hl, g.bite_mouth_half_width * INCH))
        return points

    def _frame_outlines(self, chassis) -> List[List[Tuple[float, float]]]:
        """The frame as convex pieces: the block behind the bite and the
        pieces on each side of it, counter-clockwise."""
        hl, hw = chassis.length / 2.0, chassis.width / 2.0
        edge = self._bite_inner_edge(chassis)
        back = edge[0][0]
        outlines = [[(-hl, -hw), (back, -hw), (back, hw), (-hl, hw)]]
        for side in (1.0, -1.0):
            for (x0, y0), (x1, y1) in zip(edge, edge[1:]):
                quad = [(x0, side * y0), (x1, side * y1), (x1, side * hw), (x0, side * hw)]
                outlines.append(quad if side > 0.0 else [quad[1], quad[0], quad[3], quad[2]])
        return outlines

    def frame_outline(self, chassis) -> List[Tuple[float, float]]:
        """The frame's outside, bite and all, counter-clockwise, robot frame, metres."""
        hl, hw = chassis.length / 2.0, chassis.width / 2.0
        edge = self._bite_inner_edge(chassis)
        right_side = [(x, -y) for x, y in reversed(edge)]
        return [(-hl, -hw), (hl, -hw)] + right_side + edge[1:] + [(hl, hw), (-hl, hw)]

    def bumper_inside(self, chassis) -> List[Tuple[float, float]]:
        """The inside edge of the bumpers, which follow the frame's outline."""
        return _inset(self.frame_outline(chassis), self.geometry.bumper_thickness * INCH)

    def components(self, chassis) -> List[Component]:
        """Every solid piece of the robot with the lift down and the claw open."""
        size = (chassis.length, chassis.width)
        if size not in self._components:
            self._components[size] = (self._frame_components(chassis) + self._stage_components(chassis)
                                      + self._claw_components(chassis) + self._roller_components(chassis))
        return self._components[size]

    def _frame_components(self, chassis) -> List[Component]:
        g = self.geometry
        bottom, top = g.frame_bottom * INCH, g.frame_top * INCH
        block, *sides = self._frame_outlines(chassis)
        out = [Component(FRAME_BLOCK, outline_part("drivetrain", block, bottom, top))]
        per_side = len(sides) // 2
        for side_name, outlines in (("left", sides[:per_side]), ("right", sides[per_side:])):
            parent = FRAME_BLOCK
            for k, outline in enumerate(outlines, start=1):
                key = f"frame {side_name} of the bite {k}"
                out.append(Component(key, outline_part("drivetrain", outline, bottom, top), (parent,)))
                parent = key
        return out

    def _rail_x(self, chassis) -> Tuple[float, float]:
        g = self.geometry
        front = chassis.length / 2.0 - g.rail_front_back * INCH
        return front - g.rail_depth * INCH, front

    def _stage_key(self, stage: int, side_name: str) -> str:
        return f"fixed stage {side_name}" if stage == 0 else f"stage {stage} {side_name}"

    def _stage_components(self, chassis) -> List[Component]:
        """Each stage's rail on each side, the fixed one outside, every one standing on the frame."""
        g = self.geometry
        back, front = self._rail_x(chassis)
        bottom, top = g.frame_top * INCH, g.rail_top * INCH
        out = []
        for side_name, side in (("left", 1.0), ("right", -1.0)):
            for stage in range(g.stages):
                outer = (g.rail_outer - stage * g.rail_width) * INCH
                inner = outer - g.rail_width * INCH
                y0, y1 = sorted((side * inner, side * outer))
                parents = (FRAME_BLOCK,) if stage == 0 else (self._stage_key(stage - 1, side_name),)
                out.append(Component(self._stage_key(stage, side_name), Part("lift", back, front, y0, y1, bottom, top),
                                     parents, stage / g.stages))
        return out

    def _claw_components(self, chassis) -> List[Component]:
        """The carriage between the top stage's rails, the plate on its front, and the jaws on the plate."""
        g = self.geometry
        claw_x, claw_z = self.claw_forward(chassis), g.grip_lowest * INCH
        rail_back, _ = self._rail_x(chassis)
        plate_front = claw_x - g.back_plate_behind_claw * INCH
        plate_back = plate_front - g.back_plate_thickness * INCH
        jaw_bottom, jaw_top = claw_z - g.jaw_bottom * INCH, claw_z + g.jaw_top * INCH
        inner, plate_half = g.rail_inner * INCH, g.back_plate_half_width * INCH
        top_stage = g.stages - 1
        carriage = Part("carriage", rail_back, plate_back, -inner, inner, jaw_bottom,
                        jaw_bottom + g.carriage_pad_spacing * INCH)
        plate = Part("claw", plate_back, plate_front, -plate_half, plate_half, g.back_plate_floor_gap * INCH,
                     jaw_top + g.back_plate_above_jaws * INCH)
        out = [Component("carriage", carriage, (self._stage_key(top_stage, "right"), self._stage_key(top_stage, "left")),
                         1.0),
               Component("back plate", plate, ("carriage",), 1.0)]
        for side_name, side in (("left", 1), ("right", -1)):
            below = None
            for layer, (bottom, top, inner_reach) in zip(JAW_LAYERS, self._jaw_layers(claw_z)):
                keys = {piece: f"jaw {side_name} {layer} {piece}" for piece in JAW_PIECES}
                outlines = self._jaw_outlines(claw_x, plate_front, inner_reach)
                for piece, outline in zip(JAW_PIECES, outlines):
                    parents = ("back plate",) if piece == "back" else (keys["back"],)
                    if below is not None:
                        parents += (below[piece],)
                    out.append(Component(keys[piece], outline_part("claw", outline if side > 0 else _mirror(outline),
                                                                   bottom, top),
                                         parents, 1.0, side, padded=layer != "recess"))
                below = keys
        return out

    def _jaw_layers(self, claw_z: float) -> List[Tuple[float, float, float]]:
        """Each jaw layer from the bottom up: (bottom, top, how much further in than the bottom band its V is), metres."""
        g = self.geometry
        cup_band_top, pin_band_bottom = claw_z - g.cup_band_top * INCH, claw_z + g.pin_band_bottom * INCH
        return [(claw_z - g.jaw_bottom * INCH, cup_band_top, 0.0),
                (cup_band_top, pin_band_bottom, -g.recess_reach * INCH),
                (pin_band_bottom, claw_z + g.jaw_top * INCH, g.pin_band_reach * INCH)]

    def _jaw_outlines(self, claw_x: float, plate_front: float, inner_reach: float) -> List[List[Tuple[float, float]]]:
        """One layer of the left jaw open, seen from above: the side of the V
        from the plate to its deepest point, and the side from there to the
        jaw's front.  Every layer's outside is the recess's outside."""
        g = self.geometry
        deepest = g.jaw_open_gap * INCH / 2.0 - inner_reach
        outside = g.jaw_open_gap * INCH / 2.0 + (g.recess_reach + g.jaw_thickness) * INCH
        slope = math.tan(math.radians(g.jaw_face_deg))
        front = claw_x + g.jaw_ahead * INCH
        behind = (claw_x - plate_front) * slope
        ahead = (front - claw_x) * slope
        return [[(plate_front, deepest - behind), (claw_x, deepest), (claw_x, outside), (plate_front, outside - behind)],
                [(claw_x, deepest), (front, deepest - ahead), (front, outside - ahead), (claw_x, outside)]]

    def roller_axle(self, chassis) -> Hinge:
        g = self.geometry
        return Hinge(ROLLER, -chassis.length / 2.0 + g.roller_axle_in_from_back * INCH, g.roller_axle_height * INCH)

    def _roller_components(self, chassis) -> List[Component]:
        """The posts on the frame, the axle on the posts, a channel on each
        end of the axle, and a flap on each end of each channel."""
        g = self.geometry
        axle = self.roller_axle(chassis)
        half_axle = g.roller_axle_size * INCH / 2.0
        post_half = g.roller_post_size * INCH / 2.0
        out = []
        for side_name, side in (("left", 1.0), ("right", -1.0)):
            y0, y1 = sorted((side * (g.roller_post_from_center * INCH - post_half),
                             side * (g.roller_post_from_center * INCH + post_half)))
            out.append(Component(f"roller post {side_name}",
                                 Part("roller post", axle.forward - post_half, axle.forward + post_half, y0, y1,
                                      g.frame_top * INCH, axle.up - half_axle), (FRAME_BLOCK,)))
        channel_inner = (g.roller_middle_from_center - g.roller_channel_width / 2.0) * INCH
        out.append(Component("roller axle", profile_part("roller axle", _square(axle.forward, axle.up, half_axle),
                                                         -channel_inner, channel_inner),
                             ("roller post left", "roller post right"), hinges=(axle,)))
        reach = g.roller_channel_length * INCH / 2.0
        depth = g.roller_channel_depth * INCH / 2.0
        flap_half = g.flap_thickness * INCH / 2.0
        for side_name, side in (("left", 1.0), ("right", -1.0)):
            y0, y1 = sorted((side * channel_inner, side * (channel_inner + g.roller_channel_width * INCH)))
            channel = f"roller {side_name}"
            side_view = [(axle.forward - reach, axle.up - depth), (axle.forward + reach, axle.up - depth),
                         (axle.forward + reach, axle.up + depth), (axle.forward - reach, axle.up + depth)]
            out.append(Component(channel, profile_part("roller", side_view, y0, y1), ("roller axle",), hinges=(axle,)))
            middle = side * g.roller_middle_from_center * INCH
            fy0, fy1 = middle - g.flap_width * INCH / 2.0, middle + g.flap_width * INCH / 2.0
            for end_name, end in zip(FLAP_ENDS, (-1.0, 1.0)):
                root = Hinge(f"roller flap {side_name} {end_name}", axle.forward + end * reach, axle.up)
                tip = root.forward + end * g.flap_length * INCH
                near, far = min(root.forward, tip), max(root.forward, tip)
                side_view = [(near, axle.up - flap_half), (far, axle.up - flap_half),
                             (far, axle.up + flap_half), (near, axle.up + flap_half)]
                out.append(Component(root.name, profile_part("roller flap", side_view, fy0, fy1), (channel,),
                                     hinges=(axle, root)))
        return out

    def parts(self, chassis) -> List[Part]:
        """Every piece where the lift, claw and roller have it now."""
        mech = self._attached()
        g = self.geometry
        closing = (g.jaw_open_gap * INCH - mech.gap) / 2.0
        rises, speeds = [0.0] + mech.lift.stage_rises(), [0.0] + mech.lift.stage_speeds()
        turns = mech.roller.turns()
        return [component.posed(rises[component.stage(g.stages)], closing, speeds[component.stage(g.stages)], turns)
                for component in self.components(chassis)]


def _square(forward: float, up: float, half: float) -> List[Tuple[float, float]]:
    return [(forward - half, up - half), (forward + half, up - half), (forward + half, up + half),
            (forward - half, up + half)]


def _stiff_stops(joint) -> None:
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.solref_limit = [pieces.CONTACT_TIME, pieces.DAMPING_RATIO]


def _same_travel(spec: mujoco.MjSpec, follower: str, leader: str) -> None:
    """`follower` moves on its joint exactly as far as `leader` on its own."""
    equality = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=follower, name2=leader)
    equality.solref = [pieces.CONTACT_TIME, 1.0]


# ----------------------------------------------------------------- the moving parts

class CascadeLift:
    """The lift's motors on one shaft, geared to the drum that winds in the
    string pulling the first stage up."""

    def __init__(self, g: CascadeGeometry, motors: List[Motor], model: mujoco.MjModel, data: mujoco.MjData,
                 rail_middle: float):
        self.g = g
        self.motors = motors
        self.rail_middle = rail_middle
        self.model, self.data = model, data
        names = [f"{ROBOT} stage {k}" for k in range(1, g.stages)] + [f"{ROBOT} carriage"]
        self.bodies = [model.body(name).id for name in names]
        joints = [model.joint(f"{name} slide").id for name in names]
        self.at = [int(model.jnt_qposadr[j]) for j in joints]
        self.dofs = [int(model.jnt_dofadr[j]) for j in joints]
        drum = model.joint(f"{DRUM} turn").id
        self.drum_at, self.drum_dof = int(model.jnt_qposadr[drum]), int(model.jnt_dofadr[drum])

    @property
    def angle(self) -> float:
        return self.motors[0].position

    @property
    def extension(self) -> float:
        """How far the first stage has risen, metres."""
        return float(self.data.qpos[self.at[0]])

    def stage_rises(self) -> List[float]:
        """How far each moving stage has risen (the last one is the carriage), metres."""
        return np.cumsum(self.data.qpos[self.at]).tolist()

    def stage_speeds(self) -> List[float]:
        """How fast each moving stage rises, m/s."""
        return np.cumsum(self.data.qvel[self.dofs]).tolist()

    @property
    def wound(self) -> float:
        """How much string the drum has wound in, metres."""
        return float(self.data.qpos[self.drum_at]) * self.g.lift_gear_ratio * self.g.string_per_motor_rad

    @property
    def rise(self) -> float:
        """How far the carriage has risen, metres."""
        return float(np.sum(self.data.qpos[self.at]))

    @property
    def rise_speed(self) -> float:
        return float(np.sum(self.data.qvel[self.dofs]))

    @property
    def current(self) -> float:
        return sum(abs(m.current) for m in self.motors)

    def bands(self) -> float:
        g = self.g
        share = self.extension / (g.stage_travel * INCH)
        return g.bands_down_n + share * (g.bands_up_n - g.bands_down_n)

    def apply(self, dt: float) -> None:
        """The motors' torque through the gears on the drum, and the bands' pull on the first stage."""
        g = self.g
        torque = sum(motor.output_torque() for motor in self.motors) * g.lift_gear_ratio
        speed = float(self.data.qvel[self.drum_dof])
        delivered = torque * g.lift_efficiency if torque * speed >= 0.0 else torque / g.lift_efficiency
        self.data.qfrc_applied[self.drum_dof] += delivered
        self.data.qfrc_applied[self.dofs[0]] += self.bands()

    def after(self, dt: float) -> None:
        """Turn the motors with the drum, and set each slide's friction for
        the next step from what its bearings carried in this one."""
        ratio = self.g.lift_gear_ratio
        shaft_angle = float(self.data.qpos[self.drum_at]) * ratio
        shaft_speed = float(self.data.qvel[self.drum_dof]) * ratio
        for motor in self.motors:
            motor.position, motor.velocity = shaft_angle, shaft_speed
            motor.thermal_step(dt)
        self._slide_friction()

    def _slide_friction(self) -> None:
        g, m, d = self.g, self.model, self.data
        mujoco.mj_rnePostConstraint(m, d)
        rises = [0.0] + self.stage_rises()
        for k, (body, dof) in enumerate(zip(self.bodies, self.dofs)):
            if k == len(self.bodies) - 1:
                span = g.carriage_pad_spacing * INCH
                low = (g.grip_lowest - g.jaw_bottom) * INCH + rises[k + 1]
            else:
                span = (g.rail_length * INCH) - (rises[k + 1] - rises[k])
                low = g.frame_top * INCH + rises[k + 1]
            turn = d.xmat[body].reshape(3, 3)
            middle = d.xpos[body] + turn @ (self.rail_middle, 0.0, low + span / 2.0 - rises[k + 1])
            t0, t1, t2, f0, f1, f2 = d.cfrc_int[body].tolist()
            a0, a1, a2 = (d.subtree_com[m.body_rootid[body]] - middle).tolist()
            about_middle = (t0 + a1 * f2 - a2 * f1, t1 + a2 * f0 - a0 * f2, t2 + a0 * f1 - a1 * f0)
            moment = turn.T @ about_middle
            sideways = turn.T @ (f0, f1, f2)
            pressing = math.hypot(sideways[0], sideways[1]) + 2.0 * math.hypot(moment[0], moment[1]) / span
            m.dof_frictionloss[dof] = g.slide_mu * pressing


class CascadeRoller:
    """The roller's motor turning the axle through its gears.  Positive
    volts turn it so the flaps behind the axle go up."""

    def __init__(self, g: CascadeGeometry, motor: Motor, model: mujoco.MjModel, data: mujoco.MjData):
        self.g = g
        self.motor = motor
        self.data = data
        axle = model.joint(f"{ROBOT} {ROLLER} hinge").id
        self.at, self.dof = int(model.jnt_qposadr[axle]), int(model.jnt_dofadr[axle])
        self.flaps = {}
        for side_name in ("left", "right"):
            for end_name in FLAP_ENDS:
                name = f"roller flap {side_name} {end_name}"
                self.flaps[name] = int(model.jnt_qposadr[model.joint(f"{ROBOT} {name} hinge").id])

    @property
    def angle(self) -> float:
        return float(self.data.qpos[self.at])

    @property
    def speed(self) -> float:
        return float(self.data.qvel[self.dof])

    def turns(self) -> Dict[str, float]:
        """Radians the axle has turned, and each flap is bent on its root."""
        out = {ROLLER: self.angle}
        out.update({name: float(self.data.qpos[at]) for name, at in self.flaps.items()})
        return out

    def apply(self, dt: float) -> None:
        torque = self.motor.output_torque() * self.g.roller_ratio
        efficiency = self.g.roller_gear_efficiency
        self.data.qfrc_applied[self.dof] += torque * efficiency if torque * self.speed >= 0.0 else torque / efficiency

    def after(self, dt: float) -> None:
        self.motor.position = self.angle * self.g.roller_ratio
        self.motor.velocity = self.speed * self.g.roller_ratio
        self.motor.thermal_step(dt)


class CascadeMechanism:
    """The lift, the claw and the roller in a MuJoCo world."""

    def __init__(self, robot: CascadeRobot, model: mujoco.MjModel, data: mujoco.MjData, rail_middle: float):
        g = robot.geometry
        self.g = g
        self.model, self.data = model, data
        self.lift = CascadeLift(g, robot.lift_motors, model, data, rail_middle)
        self.roller = CascadeRoller(g, robot.roller_motor, model, data)
        self.claw = robot.claw
        jaws = [model.joint(f"{ROBOT} jaw {side} slide").id for side in ("left", "right")]
        self.jaw_at = [int(model.jnt_qposadr[j]) for j in jaws]
        self.jaw_dofs = [int(model.jnt_dofadr[j]) for j in jaws]
        self.jaw_bodies = [model.body(f"{ROBOT} jaw {side}").id for side in ("left", "right")]
        self.squeeze = 0.0

    @property
    def gap(self) -> float:
        """How far apart the jaws are at their V's deepest points, metres."""
        return self.g.jaw_open_gap * INCH - float(np.sum(self.data.qpos[self.jaw_at]))

    def close_on(self, gap: float) -> None:
        """Put the jaws `gap` apart, standing still."""
        each = (self.g.jaw_open_gap * INCH - gap) / 2.0
        self.data.qpos[self.jaw_at] = each
        self.data.qvel[self.jaw_dofs] = 0.0

    def motors(self) -> Dict[str, List[Motor]]:
        return {"lift": self.lift.motors, "roller": [self.roller.motor]}

    def apply(self, dt: float) -> None:
        self.lift.apply(dt)
        self.roller.apply(dt)
        per_piston = self.g.jaw_per_piston
        travel = float(np.sum(self.data.qpos[self.jaw_at])) * per_piston
        speed = float(np.sum(self.data.qvel[self.jaw_dofs])) * per_piston
        self.claw.flow(dt, travel, speed)
        self.data.qfrc_applied[self.jaw_dofs] += self.claw.piston_force() * per_piston

    def after(self, dt: float) -> None:
        self.lift.after(dt)
        self.roller.after(dt)

"""The V5RC Override field (2026-27), to the numbers in the game manual.

Source: "VEX V5 Robotics Competition Override - Game Manual", version 2.0,
3 September 2026.  Each number cites the Appendix A drawing it was read from
(A5 Pin, A6 Cup, A7 Goals, A8 Toggle, A9 Loader, A10 Field Element Locations,
A11 Tape Locations, A12 Game Object Starting Locations, A13 Field Perimeter,
A15 Toggle Starting Orientation), the rule (<SC1>..<SC8>, <SG1>, <SG5>), or
the glossary.  Numbers the manual does not give are in `GUESSES`, with the
reason for the value.

Frame: the same one `field.py` uses.  Metres, origin at the field centre, +x
toward the blue Alliance Station, +y away from the audience, +z up.  The
drawings measure in inches from the audience-left corner; `from_corner()`
converts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

from . import override_cad as cad
from .units import INCH

MANUAL = "Override game manual 2.0 (3 September 2026)"

# ----------------------------------------------------------------- perimeter

FIELD_SIZE = 140.41 * INCH                  # A13, inside of a portable perimeter
HALF = FIELD_SIZE / 2.0
WALL_HEIGHT = 11.54 * INCH                  # A13, portable perimeter
WALL_THICKNESS = 2.00 * INCH                # A13, portable perimeter
TILE = 24.0 * INCH
ORIGIN_IN = 140.41 / 2.0

#: A10: the tile seams every element is dimensioned from, in drawing inches
GRID_IN = (23.11, 46.66, 70.20, 93.75, 117.30)


def from_corner(x_in: float, y_in: float) -> Tuple[float, float]:
    return ((x_in - ORIGIN_IN) * INCH, (y_in - ORIGIN_IN) * INCH)


# ----------------------------------------------------------------- Pin (A5)

PIN_LENGTH = 6.50 * INCH
PIN_HALF_LENGTH = 2.93 * INCH               # tip to flange face
PIN_FLANGE_THICKNESS = PIN_LENGTH - 2.0 * PIN_HALF_LENGTH    # 0.64 in; VEX's field CAD has 0.636 in
PIN_FLANGE_ACROSS_FLATS = 3.02 * INCH
PIN_FLANGE_ACROSS_CORNERS = 3.16 * INCH
PIN_TIP_DIAMETER = 1.40 * INCH
PIN_TIP_STRAIGHT = 0.64 * INCH              # straight section at the tip
PIN_CONE_DIAMETER = 2.35 * INCH             # cone, next to the flange
#: the cone's half-angle; a gripped Pin wedges or slides by it
PIN_TAPER = math.atan(((PIN_CONE_DIAMETER - PIN_TIP_DIAMETER) / 2.0)
                      / (PIN_HALF_LENGTH - PIN_TIP_STRAIGHT))


# ----------------------------------------------------------------- Cup (A6)

CUP_LENGTH = 6.48 * INCH
CUP_RIM_DIAMETER = 3.16 * INCH              # both ends
CUP_WAIST_DIAMETER = 2.32 * INCH            # the middle


# ----------------------------------------------------------------- Goals (A7)

GOAL_BASE_ACROSS_FLATS = 5.61 * INCH        # octagonal base
GOAL_BASE_CORNER_RADIUS = 3.22 * INCH       # octagon, centre to corner
GOAL_BASE_PLATE = 0.39 * INCH
GOAL_TOP_ACROSS = 3.49 * INCH
GOAL_HOLE_DIAMETER = 2.37 * INCH
GOAL_TAPER_HEIGHT = 3.25 * INCH             # the alliance Goal is only this part
GOAL_NEUTRAL_HEIGHT = 5.77 * INCH
GOAL_CENTER_HEIGHT = 8.77 * INCH
GOAL_NEUTRAL_RISER = 2.52 * INCH            # box under the tapered part
GOAL_CENTER_RISER = 5.52 * INCH


def goal_octagon() -> List[Tuple[float, float]]:
    """A7: the base outline, counter-clockwise, metres from the Goal's centre."""
    flat = GOAL_BASE_ACROSS_FLATS / 2.0
    corner = math.sqrt(GOAL_BASE_CORNER_RADIUS ** 2 - flat ** 2)
    return [(flat, -corner), (flat, corner), (corner, flat), (-corner, flat),
            (-flat, corner), (-flat, -corner), (-corner, -flat), (corner, -flat)]


@dataclass(frozen=True)
class Goal:
    name: str
    kind: str               # "red", "blue", "neutral" or "center"
    x: float
    y: float
    height: float
    riser: float

    @property
    def quadrant(self) -> Optional[str]:
        return quadrant_of(self.x, self.y) if self.kind != "center" else None

    def radius_at(self, z: float) -> float:
        """Outer radius of the Goal at height `z`, 0 above its top."""
        if z < 0.0 or z > self.height:
            return 0.0
        base = (GOAL_BASE_ACROSS_FLATS + 2.0 * GOAL_BASE_CORNER_RADIUS) / 4.0
        if z <= self.riser + GOAL_BASE_PLATE:
            return base
        t = (z - self.riser - GOAL_BASE_PLATE) / (GOAL_TAPER_HEIGHT - GOAL_BASE_PLATE)
        return base + t * (GOAL_TOP_ACROSS / 2.0 - base)


def _goal(name: str, kind: str, x_in: float, y_in: float) -> Goal:
    x, y = from_corner(x_in, y_in)
    if kind in ("red", "blue"):
        return Goal(name, kind, x, y, GOAL_TAPER_HEIGHT, 0.0)
    if kind == "neutral":
        return Goal(name, kind, x, y, GOAL_NEUTRAL_HEIGHT, GOAL_NEUTRAL_RISER)
    return Goal(name, kind, x, y, GOAL_CENTER_HEIGHT, GOAL_CENTER_RISER)


#: A10.  Names say which wall the Goal's Quadrant is on.
GOALS: Tuple[Goal, ...] = (
    _goal("red left", "red", 23.11, 46.66),
    _goal("red bottom", "red", 46.66, 23.11),
    _goal("blue top", "blue", 93.75, 117.30),
    _goal("blue right", "blue", 117.30, 93.75),
    _goal("neutral left", "neutral", 23.11, 93.75),
    _goal("neutral bottom", "neutral", 93.75, 23.11),
    _goal("neutral top", "neutral", 46.66, 117.30),
    _goal("neutral right", "neutral", 117.30, 46.66),
    _goal("center", "center", 70.20, 70.20),
)


def goal(name: str) -> Goal:
    for candidate in GOALS:
        if candidate.name == name:
            return candidate
    raise KeyError(name)


# ----------------------------------------------------------------- Quadrants (A11)
#
# The two diagonals cut the field into four triangles.  The Autonomous Line
# runs top-left to bottom-right (x + y = 0); the red side is x + y < 0.

QUADRANT_COLOR = {"left": "red", "bottom": "red", "top": "blue", "right": "blue"}


def quadrant_of(x: float, y: float) -> str:
    if abs(x) >= abs(y):
        return "right" if x > 0.0 else "left"
    return "top" if y > 0.0 else "bottom"


def side_of(x: float, y: float) -> str:
    return "red" if x + y < 0.0 else "blue"


AUTON_LINE_WIDTH = 2.50 * INCH              # A11, both white lines together
#: A11: the Midfield is a square turned 45 degrees with its corners on the
#: tile seams one tile out from the centre.
MIDFIELD_HALF_DIAGONAL = (93.75 - 70.20) * INCH


def in_midfield(x: float, y: float) -> bool:
    return abs(x) + abs(y) <= MIDFIELD_HALF_DIAGONAL


# ----------------------------------------------------------------- Toggles (A8, A15)

#: VEX's field CAD (`override_cad.toggle_shape`): the bar is 25.835 in long
#: and its end caps make it 26.00 in (A8: 25.99 in).  Its corners are
#: rounded 0.5 in; from the axle its faces are 0.844 in and its corners
#: 1.187 in, so it is 2.031 in from a corner to the face opposite and
#: 2.191 in across two corners.
TOGGLE_LENGTH = 26.00 * INCH
TOGGLE_FACE_REACH = 0.844 * INCH
TOGGLE_CORNER_REACH = 1.187 * INCH
TOGGLE_AXLE_HEIGHT = 13.202 * INCH          # CAD; A15 gives 13.20 in
TOGGLE_AXLE_FROM_CENTER = 71.451 * INCH     # CAD: 1.246 in behind the wall's inner face
TOGGLE_MASS = 0.40
#: <SC4> needs a face flat on the mounts; how far off flat still counts
TOGGLE_SEATED_TOLERANCE = math.radians(3.0)


@dataclass(frozen=True)
class ToggleMount:
    """A Toggle on its wall.  `faces` lists the colours on the face that
    points into the field, out of the field, and down onto the mounts at the
    start (A15: yellow faces in, the Quadrant's colour faces out)."""
    quadrant: str
    x: float
    y: float
    inward_x: float
    inward_y: float
    faces: Tuple[str, str, str]

    @property
    def along_x(self) -> float:
        return -self.inward_y

    @property
    def along_y(self) -> float:
        return self.inward_x


def _toggle(quadrant: str) -> ToggleMount:
    own = QUADRANT_COLOR[quadrant]
    other = "blue" if own == "red" else "red"
    inward = {"left": (1.0, 0.0), "right": (-1.0, 0.0),
              "bottom": (0.0, 1.0), "top": (0.0, -1.0)}[quadrant]
    return ToggleMount(quadrant, -inward[0] * TOGGLE_AXLE_FROM_CENTER,
                       -inward[1] * TOGGLE_AXLE_FROM_CENTER,
                       inward[0], inward[1], ("yellow", own, other))


TOGGLES: Tuple[ToggleMount, ...] = tuple(_toggle(q) for q in ("left", "bottom", "top", "right"))

#: Face normals at the start, as angles in the wall's cross-section (0 points
#: into the field, 90 points up): the face into the field, the one toward the
#: wall, and the one on the brackets' plates, which slope up 30 degrees toward
#: the field.  A rotation of +120 degrees tips the top toward the wall.
TOGGLE_FACE_ANGLES = (math.radians(60.0), math.radians(180.0), math.radians(300.0))


def toggle_inner_face(faces: Sequence[str], angle: float) -> str:
    """The colour facing into the field once turned by `angle`, rounded to
    the nearest seated position."""
    turns = round(angle / (2.0 * math.pi / 3.0)) % 3
    return faces[(-turns) % 3]


def toggle_color(faces: Sequence[str], angle: float, robot_contact: bool) -> str:
    """<SC4>: set to a colour only when seated on a face and untouched."""
    step = 2.0 * math.pi / 3.0
    off_seat = abs(angle - step * round(angle / step))
    if robot_contact or off_seat > TOGGLE_SEATED_TOLERANCE:
        return "yellow"
    return toggle_inner_face(faces, angle)


# ----------------------------------------------------------------- Loaders (A9, A10)

LOADER_BORE = 3.39 * INCH
LOADER_DEPTH = 3.74 * INCH                  # out from the wall
LOADER_WIDTH = 4.02 * INCH                  # along the wall
LOADER_TOP_LOWERED = 14.37 * INCH
LOADER_TOP_RAISED = 22.97 * INCH
LOADER_OPENING = 3.25 * INCH                # gap under the tube
LOADER_TUBE_HEIGHT = LOADER_TOP_LOWERED - LOADER_OPENING


def loader_span(raised: bool) -> Tuple[float, float]:
    """Bottom and top of a Loader's tube, lowered or lifted by its handle."""
    top = LOADER_TOP_RAISED if raised else LOADER_TOP_LOWERED
    return top - LOADER_TUBE_HEIGHT, top


@dataclass(frozen=True)
class Loader:
    """`name` is the alliance and the side the Loader is on as its Drive
    Team sees it from the Alliance Station."""
    alliance: str
    x: float
    y: float
    inward_x: float
    name: str


def _loader(alliance: str, y_in: float) -> Loader:
    inward = 1.0 if alliance == "red" else -1.0
    x = -inward * (HALF - LOADER_DEPTH / 2.0)
    y = from_corner(0.0, y_in)[1]
    side = "left" if y * inward > 0.0 else "right"
    return Loader(alliance, x, y, inward, f"{alliance} {side}")


def loader(name: str) -> "Loader":
    for candidate in LOADERS:
        if candidate.name == name:
            return candidate
    raise ValueError(f"there is no {name} Loader; the Loaders are {', '.join(l.name for l in LOADERS)}")


def loader_box(loader: "Loader", raised: bool) -> Tuple[float, float, float, float, float, float]:
    """x0, x1, y0, y1, bottom, top of the Loader's tube."""
    wall = -loader.inward_x * HALF
    front = wall + loader.inward_x * LOADER_DEPTH
    bottom, top = loader_span(raised)
    return (min(wall, front), max(wall, front), loader.y - LOADER_WIDTH / 2.0, loader.y + LOADER_WIDTH / 2.0,
            bottom, top)


#: A10: red on the red Alliance Station wall, blue on the blue one
LOADERS: Tuple[Loader, ...] = (
    _loader("red", 11.44), _loader("red", 128.97),
    _loader("blue", 11.44), _loader("blue", 128.97),
)

#: A11: from the wall to the middle of the first tile, one tile along it
LOAD_ZONE_DEPTH = 23.11 / 2.0 * INCH
LOAD_ZONE_LENGTH = 23.11 * INCH


def load_zone(loader: Loader) -> Tuple[float, float, float, float]:
    """xmin, xmax, ymin, ymax."""
    wall = -loader.inward_x * HALF
    inner = wall + loader.inward_x * LOAD_ZONE_DEPTH
    ys = (HALF - LOAD_ZONE_LENGTH, HALF) if loader.y > 0.0 else (-HALF, -HALF + LOAD_ZONE_LENGTH)
    return (min(wall, inner), max(wall, inner), ys[0], ys[1])


# ----------------------------------------------------------------- starting layout (A12)

@dataclass(frozen=True)
class PinStart:
    """`colors` are the two halves.  `axis` points from the first half to the
    second.  `in_goal` / `in_cup` name what the first half is nested in."""
    colors: Tuple[str, str]
    position: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    in_goal: Optional[str] = None
    in_cup: Optional[int] = None


@dataclass(frozen=True)
class CupStart:
    """`axis` points from the opaque (gray) half to the clear half."""
    position: Tuple[float, float, float]
    axis: Tuple[float, float, float]


@dataclass
class Layout:
    pins: List[PinStart] = dc_field(default_factory=list)
    cups: List[CupStart] = dc_field(default_factory=list)


UP = (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)
#: A12: Pins lying around a Cup, middle of Pin to middle of Cup
CROSS_ARM = 4.65 * INCH
#: A12: a Cup against the wall, centre to the wall
WALL_CUP_INSET = 1.58 * INCH


def starting_layout() -> Layout:
    """Where A12 puts everything.  How high a Pin stands in a Cup or a Goal
    is where their shapes in VEX's field CAD meet (`override_cad.seats`)."""
    layout = Layout()
    seats = cad.seats()
    cup_middle = cad.cup().half_length

    def cup(x: float, y: float, clear_up: bool) -> int:
        layout.cups.append(CupStart((x, y, cup_middle), UP if clear_up else DOWN))
        return len(layout.cups) - 1

    def standing_pin(colors: Tuple[str, str], x: float, y: float, cup_index: int) -> None:
        layout.pins.append(PinStart(colors, (x, y, cup_middle + seats.pin_in_cup), UP, in_cup=cup_index))

    for x_in, y_in in ((23.11, 117.30), (46.66, 93.75), (93.75, 46.66), (117.30, 23.11)):
        x, y = from_corner(x_in, y_in)
        cup(x, y, clear_up=True)
        lying_z = PIN_FLANGE_ACROSS_FLATS / 2.0
        for dx, dy, colors in ((1, 0, ("yellow", "blue")), (0, 1, ("yellow", "blue")),
                               (-1, 0, ("yellow", "red")), (0, -1, ("yellow", "red"))):
            layout.pins.append(PinStart(colors, (x + dx * CROSS_ARM, y + dy * CROSS_ARM, lying_z),
                                        (float(dx), float(dy), 0.0)))

    for x_in, y_in, colors in ((46.66, 70.20, ("blue", "red")), (70.20, 46.66, ("blue", "red")),
                               (70.20, 93.75, ("red", "blue")), (93.75, 70.20, ("red", "blue"))):
        x, y = from_corner(x_in, y_in)
        standing_pin(colors, x, y, cup(x, y, clear_up=True))

    for x_in in (23.11, 46.66, 93.75, 117.30):
        x, y = from_corner(x_in, x_in)
        standing_pin(("yellow", "yellow"), x, y, cup(x, y, clear_up=True))

    for g in GOALS:
        if g.kind in ("neutral", "center"):
            layout.pins.append(PinStart(("yellow", "yellow"), (g.x, g.y, g.riser + seats.pin_in_goal),
                                        UP, in_goal=g.name))

    near, far = WALL_CUP_INSET, 140.41 - 1.58
    for along in ((43.50, 46.66, 49.82), (90.59, 93.75, 96.91)):
        for wall in ("bottom", "top", "left", "right"):
            for i, a in enumerate(along):
                x_in, y_in = {"bottom": (a, near / INCH), "top": (a, far),
                              "left": (near / INCH, a), "right": (far, a)}[wall]
                x, y = from_corner(x_in, y_in)
                index = cup(x, y, clear_up=False)
                if i == 1:
                    standing_pin(("yellow", "yellow"), x, y, index)
    return layout


#: <SG5>
PRELOAD_COLORS = {"red": ("red", "yellow"), "blue": ("blue", "yellow")}

#: Field Overview: each alliance's Match Loads in a Head-to-Head Match.  The
#: two yellow/yellow Match Loads are one per alliance (VEX U, <VUR1>, gives
#: each alliance two more, for three).
MATCH_LOADS = {
    "red": {"cup": 10, ("red", "yellow"): 10, ("yellow", "yellow"): 1},
    "blue": {"cup": 10, ("blue", "yellow"): 10, ("yellow", "yellow"): 1},
}

# ----------------------------------------------------------------- scoring (<SC1>..<SC8>)

AUTONOMOUS_BONUS = 12
ALLIANCE_PIN_POINTS = 5
YELLOW_PIN_POINTS = 10
MIDFIELD_ROBOT_POINTS = 8
AUTON_SECONDS = 15.0
DRIVER_SECONDS = 105.0


@dataclass(frozen=True)
class StackPin:
    lower: str
    upper: str


@dataclass(frozen=True)
class StackCup:
    """`lower_opaque`: the gray half is the one facing down the stack."""
    lower_opaque: bool


@dataclass
class GoalStack:
    """What is nested in one Goal, from the Goal up.  Pins and Cups alternate,
    starting with the Pin in the Goal.  Anything past the first break in the
    nesting is not part of the stack."""
    goal: str
    layers: List[object] = dc_field(default_factory=list)


@dataclass
class ScoredHalf:
    goal: str
    color: str
    owner: Optional[str]


def scored_halves(stack: GoalStack) -> List[Tuple[str, bool]]:
    """(colour, visible) for every half of every Placed Pin in the stack.
    <SC3>: a half nested in the Goal or in a Cup's opaque half is not
    visible, so the lower half of the Pin in the Goal never scores."""
    halves: List[Tuple[str, bool]] = []
    layers = stack.layers
    for i, layer in enumerate(layers):
        if not isinstance(layer, StackPin):
            continue
        in_the_goal = i == 0
        below = layers[i - 1] if i > 0 else None
        above = layers[i + 1] if i + 1 < len(layers) else None
        lower_hidden = in_the_goal or (isinstance(below, StackCup) and not below.lower_opaque)
        upper_hidden = isinstance(above, StackCup) and above.lower_opaque
        halves.append((layer.lower, not lower_hidden))
        halves.append((layer.upper, not upper_hidden))
    return halves


def yellow_owner(goal_name: str, toggles: Dict[str, str], midfield_robots: Dict[str, int],
                 skills: bool = False) -> Optional[str]:
    g = goal(goal_name)
    if g.kind == "center":
        red, blue = midfield_robots.get("red", 0), midfield_robots.get("blue", 0)
        if red == blue:
            return None
        return "red" if red > blue else "blue"
    setting = toggles[g.quadrant]
    if setting == "yellow":
        return None
    if skills and setting != QUADRANT_COLOR[g.quadrant]:
        return None
    return setting


def owned_halves(stacks: Sequence[GoalStack], toggles: Dict[str, str],
                 midfield_robots: Dict[str, int], autonomous: bool = False,
                 skills: bool = False) -> List[ScoredHalf]:
    result: List[ScoredHalf] = []
    for stack in stacks:
        for color, visible in scored_halves(stack):
            if not visible:
                continue
            if color != "yellow":
                result.append(ScoredHalf(stack.goal, color, color))
                continue
            if autonomous and goal(stack.goal).kind == "center":
                result.append(ScoredHalf(stack.goal, color, None))
                continue
            result.append(ScoredHalf(stack.goal, color,
                                     yellow_owner(stack.goal, toggles, midfield_robots, skills)))
    return result


@dataclass
class AllianceScore:
    alliance_pins: int = 0
    yellow_pins: int = 0
    midfield_robots: int = 0
    autonomous_bonus: int = 0

    @property
    def total(self) -> int:
        return (self.alliance_pins * ALLIANCE_PIN_POINTS + self.yellow_pins * YELLOW_PIN_POINTS
                + self.midfield_robots * MIDFIELD_ROBOT_POINTS + self.autonomous_bonus)


def score(stacks: Sequence[GoalStack], toggles: Dict[str, str],
          midfield_robots: Optional[Dict[str, int]] = None, autonomous: bool = False,
          skills: bool = False) -> Dict[str, AllianceScore]:
    """<SC1>..<SC6>.  With `autonomous` the Midfield parts are left out, as
    <SC7a> says the Autonomous Bonus is judged."""
    robots = {} if autonomous or midfield_robots is None else midfield_robots
    result = {"red": AllianceScore(), "blue": AllianceScore()}
    for half in owned_halves(stacks, toggles, robots, autonomous, skills):
        if half.owner is None:
            continue
        if half.color == "yellow":
            result[half.owner].yellow_pins += 1
        else:
            result[half.owner].alliance_pins += 1
    for alliance in result:
        result[alliance].midfield_robots = robots.get(alliance, 0)
    return result


def autonomous_bonus(at_auton_end: Dict[str, AllianceScore]) -> Dict[str, int]:
    """<SC7>: 12 to the leader, 6 each on a tie."""
    red, blue = at_auton_end["red"].total, at_auton_end["blue"].total
    if red == blue:
        return {"red": AUTONOMOUS_BONUS // 2, "blue": AUTONOMOUS_BONUS // 2}
    return {"red": AUTONOMOUS_BONUS if red > blue else 0,
            "blue": AUTONOMOUS_BONUS if blue > red else 0}


def win_point_progress(alliance: str, stacks: Sequence[GoalStack],
                       toggles: Dict[str, str]) -> Tuple[int, int]:
    """<SC8>: (Pin halves the alliance owns in Goals on its side of the
    Autonomous Line, how many of those Goals hold at least two)."""
    per_goal: Dict[str, int] = {}
    for half in owned_halves(stacks, toggles, {}, autonomous=True):
        g = goal(half.goal)
        if half.owner != alliance:
            continue
        if g.quadrant is not None and QUADRANT_COLOR[g.quadrant] != alliance:
            continue
        per_goal[half.goal] = per_goal.get(half.goal, 0) + 1
    return sum(per_goal.values()), sum(1 for count in per_goal.values() if count >= 2)


def autonomous_win_point(alliance: str, stacks: Sequence[GoalStack], toggles: Dict[str, str],
                         touching_perimeter: bool, worlds_qualifying: bool = False) -> bool:
    """<SC8>.  Goals in Quadrants across the Autonomous Line do not count."""
    pins_needed, goals_needed = win_point_needs(worlds_qualifying)
    pins, goals_with_two = win_point_progress(alliance, stacks, toggles)
    return pins >= pins_needed and goals_with_two >= goals_needed and not touching_perimeter


def win_point_needs(worlds_qualifying: bool = False) -> Tuple[int, int]:
    return (7, 3) if worlds_qualifying else (6, 2)


# ----------------------------------------------------------------- guesses

#: Every number above that the manual, the Q&A or VEX's own documents do not
#: give, and why it is what it is.
GUESSES: Dict[str, str] = {
    "TOGGLE_MASS": "0.40 kg. No mass is published. The CAD's bar, end caps and washers are 21.5 in^3, so "
                   "this is 1.13 g/cm^3, a plastic's; its inertia is the CAD solid's at that density.",
    "Goal profile": "The robot sees a Goal's base as the octagon and its taper as a cone of the octagon's mean "
                    "radius. Pins and Cups touch the CAD's shape.",
    "TOGGLE_SEATED_TOLERANCE": "3 degrees off a face still counts as seated.",
}

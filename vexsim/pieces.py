"""Pins and Cups as rigid bodies with the shapes VEX's field CAD gives them.

The contact solver is MuJoCo (https://mujoco.org).  A Pin is its flange and
its two tapers; a Cup is its shell, inside and outside, cut into slices
(`override_cad`); the Goals are their funnels.  Nothing is scripted: a Pin
dropped into a Cup goes in because the Cup's mouth is wider than the Pin's
tip, and stops where the Pin's flange meets the Cup's rim; a Pin started
tilted slides down the inside wall until it leans on it; a stack stands as
high as its pieces' shapes put it.  `stacks()` reads what is nested in what
off where the pieces are.

The same MuJoCo model can hold a robot (`RobotBuild`), whose bodies then
touch the pieces and the field the way they touch each other.  A world with
a robot is stepped by the robot's dynamics (`mujoco_drive.MujocoDrive`),
which calls `after_step` after each step.

Pieces still for a moment fall asleep and cost nothing until something
awake touches them.  Every Pin and Cup there can be in a Match is made with
the model and kept out of the field, asleep and touching nothing, until
`add_pin` or `add_cup` brings it in.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from . import foam
from . import override_cad as cad
from . import override_field as of
from . import setup_cache
from .mujoco_checks import raise_on_warnings
from .out_of_reach import OutOfReach
from .toggles import Toggles, add_toggles
from .units import G, INCH

#: The world's step, s
DT = 0.00025
#: How long a contact takes to push an overlap out, and how much of a
#: landing it gives back: MuJoCo's contact spring settles in CONTACT_TIME
#: with a damping ratio that returns RESTITUTION of the speed.
CONTACT_TIME = 8.0 * DT
RESTITUTION = 0.15
DAMPING_RATIO = -math.log(RESTITUTION) / math.sqrt(math.pi ** 2 + math.log(RESTITUTION) ** 2)
#: Passes of MuJoCo's noslip solver after each step.  Without them a Pin
#: held by friction in the claw creeps down the jaws at 0.016 in/s; with
#: them, 0.00002 in/s.
NOSLIP_ITERATIONS = 10
FLOOR_MU = 0.5
PIECE_MU = 0.3
STATIC_MU = 0.3
ROBOT_MU = 0.35
#: kg, weighed by the user on 2026-11-04.  Each piece's inertia is its CAD
#: solid's at one even density that gives this mass: 1.06 g/cm^3 for a Pin,
#: 1.18 g/cm^3 for a Cup.
PIN_MASS = 0.072
CUP_MASS = 0.077
#: The width of foam an edge of a lying piece presses into
EDGE_WIDTH = 0.10 * INCH

#: Corners kept on a piece's outline: its two ends' rims, 64 corners round each
OUTLINE_CORNERS = 128
MAX_PINS = 64
MAX_CUPS = 60
PARKED_DEPTH = -5.0
WAKE_SPEED = 1e-6
#: How far the robot may reach into a piece or a field element where it
#: starts: less than the contacts' own give, so it starts touching, not inside
OVERLAP_ALLOWED = 0.001 * INCH

#: Nesting, read off the pieces' poses: the lower end of the one on top has
#: to be this far inside the one under it, lined up within NESTED_ANGLE.
NESTED_DEPTH = 1.0 * INCH
NESTED_ANGLE = math.radians(30.0)

#: What collides with what.  The tiles and walls are flat, and a flat face
#: only ever meets a piece's convex outline, so they collide with that one
#: part of a piece; the Goals, the Loaders, the robot and other pieces
#: collide with all the convex parts of its shape.
FLAT_BIT, STATIC_BIT, PIECE_BIT, ROBOT_BIT = 8, 1, 2, 4
FREE = (PIECE_BIT, STATIC_BIT | PIECE_BIT | ROBOT_BIT, FLAT_BIT)
PARKED = (0, 0, 0)

GUESSES: Dict[str, str] = {
    "PIN_MASS and CUP_MASS": "The robot's owner, 2026-11-04/05: a Pin 72 g and a Cup 77 g, weighed.",
    "FLOOR_MU": "0.5, plastic sliding on a foam tile. The nearest published figure is 0.63 for solid "
                "EVA against an unnamed surface, and foam's pores give it a little less grip than solid.",
    "PIECE_MU": "0.3, moulded plastic on moulded plastic.",
    "STATIC_MU": "0.3, plastic on the steel walls, the Goals and the Loaders.",
    "ROBOT_MU": "0.35, the robot's aluminium and polycarbonate on the pieces' plastic and the field's steel "
                "and polycarbonate.",
    "RESTITUTION": "0.15: a Pin dropped 10 in onto foam comes back up about 0.2 in. Foam keeps most of "
                   "a landing (foam.FOAM_ENERGY_LOSS is for slow presses, not impacts).",
    "EDGE_WIDTH": "0.10 in: how wide a strip of foam a lying Pin's flange or a Cup's rim presses into. With "
                  "foam.rolling_resistance and foam.scrub_arm it sets how fast a rolling or spinning piece stops.",
    "CONTACT_TIME": "2 ms, eight steps. A resting piece sinks into what holds it about 0.001 mm.",
}


# ----------------------------------------------------------------- shapes

@dataclass(frozen=True)
class Kind:
    shape: cad.PieceShape
    mass: float

    @property
    def inertia(self) -> Tuple[float, float, float]:
        axial, across = self.shape.gyration
        return (self.mass * across, self.mass * across, self.mass * axial)


@lru_cache(maxsize=None)
def kinds() -> Dict[str, Kind]:
    pin, cup = cad.pin(), cad.cup()
    return {"pin": Kind(pin, PIN_MASS), "cup": Kind(cup, CUP_MASS)}


def pin_mass() -> float:
    return kinds()["pin"].mass


def cup_mass() -> float:
    return kinds()["cup"].mass


def floor_friction() -> Tuple[float, float, float]:
    """(sliding, spinning, rolling) on the tiles.  A lying piece rests on
    two edges; each carries half its weight into a strip EDGE_WIDTH wide."""
    pin = kinds()["pin"]
    load = pin.mass * G / 2.0
    radius = of.PIN_FLANGE_ACROSS_CORNERS / 2.0
    rolling = foam.rolling_resistance(load, radius, EDGE_WIDTH) * radius
    spinning = foam.scrub_arm(load, radius, EDGE_WIDTH, FLOOR_MU, FLOOR_MU)
    return FLOOR_MU, spinning, rolling


def quat_from_axis(axis) -> Tuple[float, float, float, float]:
    """The rotation (w, x, y, z) that takes +z onto `axis`."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    if ax[2] < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    w = 1.0 + ax[2]
    q = np.array([w, -ax[1], ax[0], 0.0])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def turned_by(heading: float, quat) -> Tuple[float, float, float, float]:
    """The orientation `quat` turned `heading` about the vertical."""
    out = np.empty(4)
    mujoco.mju_mulQuat(out, _yaw(heading), np.asarray(quat, dtype=float))
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def _axis_of(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


# ----------------------------------------------------------------- the model

@dataclass(eq=False)
class Piece:
    """One Pin or Cup.  A Pin's axis runs from `colors[0]`'s half to
    `colors[1]`'s; a Cup's from its opaque (gray) half to its clear half."""
    id: int
    kind: str
    colors: Tuple[str, str]
    world: "Pieces"
    body: int

    @property
    def label(self) -> str:
        """"a Cup", "a yellow Pin" or "a red/yellow Pin"."""
        if self.kind == "cup":
            return "a Cup"
        first, second = self.colors
        return f"a {first} Pin" if first == second else f"a {first}/{second} Pin"

    @property
    def mass(self) -> float:
        return kinds()[self.kind].mass

    @property
    def half_length(self) -> float:
        return kinds()[self.kind].shape.half_length

    def radius_at(self, s: float) -> float:
        return kinds()[self.kind].shape.radius_at(s)

    def squeezed_width_at(self, s: float) -> float:
        return kinds()[self.kind].shape.squeezed_width_at(s)

    def center(self) -> Tuple[float, float, float]:
        p = self.world.data.qpos[self.world.qpos_at[self.body]:][:3]
        return (float(p[0]), float(p[1]), float(p[2]))

    def quat(self) -> np.ndarray:
        return self.world.data.qpos[self.world.qpos_at[self.body] + 3:][:4]

    def axis(self) -> Tuple[float, float, float]:
        a = _axis_of(self.quat())
        return (float(a[0]), float(a[1]), float(a[2]))

    def end(self, sign: float) -> np.ndarray:
        return np.array(self.center()) + np.array(self.axis()) * self.half_length * sign

    def velocity(self) -> np.ndarray:
        return self.world.data.qvel[self.world.qvel_at[self.body]:][:3].copy()

    def spheres(self, count: int = 9) -> List[Tuple[Tuple[float, float, float], float]]:
        """Balls along the axis, each as wide as the piece is there, that
        together hold it: (middle, radius)."""
        middle, axis = np.array(self.center()), np.array(self.axis())
        out = []
        for k in range(count):
            s = self.half_length * (2.0 * k / (count - 1) - 1.0)
            point = middle + axis * s
            out.append(((float(point[0]), float(point[1]), float(point[2])), self.radius_at(s)))
        return out

    @property
    def asleep(self) -> bool:
        return self.world.data.body_awake[self.body] == mujoco.mjtSleepState.mjS_ASLEEP

    @property
    def below(self):
        """The Cup or Goal this Pin is nested in, or the Pin this Cup is over."""
        return self.world.below(self)

    @property
    def above(self) -> Optional["Piece"]:
        return self.world.above(self)


def contact_settings(geom, mu, condim=3, priority=0, spin=0.0, roll=0.0, contact_time=CONTACT_TIME) -> None:
    geom.friction = [mu, spin, roll]
    geom.condim = condim
    geom.priority = priority
    geom.solref = [contact_time, DAMPING_RATIO]


def _add_meshes(spec, prefix: str, hulls, most_corners: int = -1) -> List[str]:
    names = []
    for i, hull in enumerate(hulls):
        name = f"{prefix}{i}"
        spec.add_mesh(name=name, uservert=np.asarray(hull, dtype=float).flatten().tolist(), maxhullvert=most_corners)
        names.append(name)
    return names


#: A field element's geoms are named ELEMENT_PREFIX, the element's name, and
#: after a "|" which part of it the geom is
ELEMENT_PREFIX = "element:"
WALLS = ("the blue Alliance Station wall", "the red Alliance Station wall", "the wall away from the audience",
         "the audience wall")


def _static_field(spec, goals: Sequence[of.Goal]) -> None:
    world = spec.worldbody
    sliding, spinning, rolling = floor_friction()
    floor = world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[of.HALF + 1.0, of.HALF + 1.0, 0.1],
                           contype=FLAT_BIT, conaffinity=0)
    contact_settings(floor, sliding, condim=6, priority=2, spin=spinning, roll=rolling)
    thickness = of.WALL_THICKNESS
    for k, (x, y, hx, hy) in enumerate(((of.HALF + thickness / 2.0, 0.0, thickness / 2.0, of.HALF + thickness),
                                        (-of.HALF - thickness / 2.0, 0.0, thickness / 2.0, of.HALF + thickness),
                                        (0.0, of.HALF + thickness / 2.0, of.HALF + thickness, thickness / 2.0),
                                        (0.0, -of.HALF - thickness / 2.0, of.HALF + thickness, thickness / 2.0))):
        wall = world.add_geom(name=f"{ELEMENT_PREFIX}{WALLS[k]}", type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=[hx, hy, of.WALL_HEIGHT / 2.0], pos=[x, y, of.WALL_HEIGHT / 2.0],
                              contype=FLAT_BIT, conaffinity=0)
        contact_settings(wall, STATIC_MU, priority=1)
    taper = _add_meshes(spec, "goal", cad.goal_taper())
    risers = {kind: _add_meshes(spec, f"{kind} riser", [cad.riser_hull(kind)])[0] for kind in ("neutral", "center")}
    for goal in goals:
        label = f"{ELEMENT_PREFIX}the {goal.name} Goal"
        body = world.add_body(name=f"goal {goal.name}", pos=[goal.x, goal.y, 0.0])
        if goal.riser > 0.0:
            riser = body.add_geom(name=f"{label}|riser", type=mujoco.mjtGeom.mjGEOM_MESH,
                                  meshname=risers["center" if goal.kind == "center" else "neutral"],
                                  contype=STATIC_BIT, conaffinity=0)
            contact_settings(riser, STATIC_MU, priority=1)
        for k, name in enumerate(taper):
            part = body.add_geom(name=f"{label}|taper {k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                 pos=[0.0, 0.0, goal.riser], contype=STATIC_BIT, conaffinity=0)
            contact_settings(part, STATIC_MU, priority=1)


def loader_foot(loader: of.Loader, raised: bool) -> Tuple[float, float, float]:
    """Where the Loader's frame is: the middle of its foot on the wall's inner face, lifted when raised."""
    lift = of.LOADER_TOP_RAISED - of.LOADER_TOP_LOWERED if raised else 0.0
    return -loader.inward_x * of.HALF, loader.y, lift


def _loaders(spec, loaders: Sequence[of.Loader]) -> None:
    """Each Loader as VEX's CAD shapes it, moved by its handle as the Drive Team lifts it."""
    parts = _add_meshes(spec, "loader", [points * INCH for points in cad.loader_shape().parts])
    for i, loader in enumerate(loaders):
        quat = np.zeros(4)
        along = loader.inward_x
        mujoco.mju_mat2Quat(quat, np.array([[loader.inward_x, 0.0, 0.0], [0.0, along, 0.0], [0.0, 0.0, 1.0]]).flatten())
        body = spec.worldbody.add_body(name=f"loader{i}", mocap=True, pos=list(loader_foot(loader, raised=False)),
                                       quat=quat.tolist())
        for k, name in enumerate(parts):
            part = body.add_geom(name=f"{ELEMENT_PREFIX}the {loader.name} Loader|part {k}",
                                 type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, contype=STATIC_BIT, conaffinity=0)
            contact_settings(part, STATIC_MU, priority=1)


def _piece_bodies(spec, kind: str, count: int) -> None:
    meshes = _add_meshes(spec, kind, kinds()[kind].shape.hulls)
    outline = _add_meshes(spec, f"{kind}_outline", [np.vstack(kinds()[kind].shape.hulls)], OUTLINE_CORNERS)[0]
    mass = kinds()[kind].mass
    inertia = kinds()[kind].inertia
    for i in range(count):
        body = spec.worldbody.add_body(name=f"{kind}{i}", pos=[(i % 32) * 0.3, (i // 32) * 0.3 + (2.0 if kind == "cup" else 0.0),
                                                                PARKED_DEPTH])
        body.add_freejoint()
        body.explicitinertial = True
        body.mass = mass
        body.inertia = list(inertia)
        body.ipos = [0.0, 0.0, 0.0]
        body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=outline, contype=0, conaffinity=0)
        contact_settings(geom, PIECE_MU)
        for name in meshes:
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, contype=0, conaffinity=0)
            contact_settings(geom, PIECE_MU)


@dataclass(frozen=True)
class RobotBuild:
    """A robot to put in the world: `build(spec)` adds its bodies, the root
    one named `root`, and `key` tells one build from another in this
    process.  `inputs` writes out everything `build` reads, so a model kept
    on disk by `setup_cache` is found only by the same."""
    root: str
    key: tuple
    build: Callable[[mujoco.MjSpec], None]
    inputs: str


#: 64 times the most any run has used: 3.99 MB, both for the full field with every piece placed and for
#: all 124 pieces dropped in a heap on the robot.  Pages the world never uses take no memory.
ARENA_BYTES = 256 * 2 ** 20
_TEMPLATES: Dict[tuple, mujoco.MjModel] = {}


def _template(goals: Tuple[of.Goal, ...], loaders: Tuple[of.Loader, ...], pins: int, cups: int,
              robot: Optional[RobotBuild], others: Tuple[RobotBuild, ...]) -> mujoco.MjModel:
    key = (goals, loaders, pins, cups, None if robot is None else robot.key, tuple(o.key for o in others))
    if key not in _TEMPLATES:
        inputs = (goals, loaders, pins, cups, None if robot is None else robot.inputs,
                  tuple(o.inputs for o in others))
        _TEMPLATES[key] = setup_cache.kept("model", inputs, lambda: _build(goals, loaders, pins, cups, robot, others))
    return _TEMPLATES[key]


def _build(goals: Tuple[of.Goal, ...], loaders: Tuple[of.Loader, ...], pins: int, cups: int,
           robot: Optional[RobotBuild], others: Tuple[RobotBuild, ...]) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.memory = ARENA_BYTES
    option = spec.option
    option.timestep = DT
    option.gravity = [0.0, 0.0, -G]
    option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    option.noslip_iterations = NOSLIP_ITERATIONS
    option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_AUTORESET
    _static_field(spec, goals)
    _loaders(spec, loaders)
    add_toggles(spec, lambda geom, mu: contact_settings(geom, mu, priority=1), [CONTACT_TIME, DAMPING_RATIO],
                ELEMENT_PREFIX, STATIC_BIT, ROBOT_BIT)
    _piece_bodies(spec, "pin", pins)
    _piece_bodies(spec, "cup", cups)
    if robot is not None:
        robot.build(spec)
    for other in others:
        other.build(spec)
    return spec.compile()


# ----------------------------------------------------------------- the world of pieces

LAYOUT_SETTLE_SECONDS = 6.0
#: Where each piece of a layout comes to rest, by the Goals, the Loaders and the layout
_AT_REST: Dict[tuple, List[tuple]] = {}


class Pieces:
    """Pins and Cups on the field, the robot when `robot` says how to build
    it, and the other robots `others` build (`other_robots`)."""

    def __init__(self, goals: Sequence[of.Goal] = of.GOALS, loaders: Sequence[of.Loader] = of.LOADERS,
                 pins: int = MAX_PINS, cups: int = MAX_CUPS, robot: Optional[RobotBuild] = None,
                 others: Sequence[RobotBuild] = ()):
        self.goals = tuple(goals)
        self.loaders = tuple(loaders)
        self.most = dict(pin=pins, cup=cups)
        self.model = copy.deepcopy(_template(self.goals, self.loaders, pins, cups, robot, tuple(others)))
        self.data = mujoco.MjData(self.model)
        self.qpos_at = self.model.jnt_qposadr[self.model.body_jntadr]
        self.qvel_at = self.model.jnt_dofadr[self.model.body_jntadr]
        self._spare = {kind: [self.model.body(f"{kind}{i}").id for i in range(count)]
                       for kind, count in self.most.items()}
        self.has_robot = robot is not None
        root = self.model.body(robot.root).id if robot is not None else -1
        self.robot_bodies = np.nonzero(self.model.body_rootid == root)[0]
        self.robot_geoms = np.nonzero(np.isin(self.model.geom_bodyid, self.robot_bodies)
                                      & (self.model.geom_contype != 0))[0]
        self.robot_geom_set = frozenset(self.robot_geoms.tolist())
        other_roots = [self.model.body(other.root).id for other in others]
        other_bodies = np.nonzero(np.isin(self.model.body_rootid, other_roots))[0]
        self._robot_body_set = frozenset(self.robot_bodies.tolist() + other_bodies.tolist())
        #: Geoms of the robot and of the other robots, for the Toggles' <SC4>
        self.every_robot_geom = frozenset(
            np.nonzero(np.isin(self.model.geom_bodyid, list(self._robot_body_set))
                       & (self.model.geom_contype != 0))[0].tolist())
        self.body_of_geom: List[int] = self.model.geom_bodyid.tolist()
        self.by_body: Dict[int, Piece] = {}
        #: qpos addresses of each Pin's and each Cup's centre, in the order of `pins` and `cups`
        self._centres_at = {kind: np.zeros((0, 3), dtype=int) for kind in ("pin", "cup")}
        #: The two geoms of each of this step's contacts
        self.contact_geoms: List[List[int]] = []
        self._loader_mocap = {loader: self.model.body(f"loader{i}").mocapid[0] for i, loader in enumerate(self.loaders)}
        self.elements = {geom: self.model.geom(geom).name[len(ELEMENT_PREFIX):].split("|")[0]
                         for geom in range(self.model.ngeom)
                         if self.model.geom(geom).name.startswith(ELEMENT_PREFIX)}
        self.toggles = Toggles(self.model, self.data)
        #: (contype, conaffinity) of each piece in play, by its body
        self.body_bits: Dict[int, Tuple[int, int]] = {}
        self.out_of_reach = OutOfReach(self.model, self.data, [b for bodies in self._spare.values() for b in bodies],
                                       self.body_bits)
        self.pins: List[Piece] = []
        self.cups: List[Piece] = []
        self.t = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------- building

    def _bring_in(self, kind: str, colors, position, axis) -> Piece:
        if not self._spare[kind]:
            raise ValueError(f"all {self.most[kind]} {kind}s are already on the field")
        body = self._spare[kind].pop(0)
        piece = Piece(len(self.pins) + len(self.cups), kind, tuple(colors), self, body)
        (self.pins if kind == "pin" else self.cups).append(piece)
        self.by_body[body] = piece
        self._centres_at[kind] = np.vstack([self._centres_at[kind], self.qpos_at[body] + np.arange(3)])
        self._set_collisions(piece, FREE)
        self.place(piece, position, axis)
        return piece

    def add_pin(self, colors, position, axis) -> Piece:
        return self._bring_in("pin", colors, position, axis)

    def add_cup(self, position, axis) -> Piece:
        return self._bring_in("cup", ("gray", "clear"), position, axis)

    def add_lying_pin(self, colors, x: float, y: float, heading: float) -> Piece:
        """A Pin at rest on the floor, its axis heading `heading` from +x toward +y."""
        rest = lying_pin_rest()
        pin = self.add_pin(colors, (x, y, rest.height), rest.axis(heading))
        self.place_turned(pin, (x, y, rest.height), rest.quat(heading))
        return pin

    def place(self, piece: Piece, position, axis, velocity=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)) -> None:
        self.place_turned(piece, position, quat_from_axis(axis), velocity, spin)

    def place_turned(self, piece: Piece, position, quat, velocity=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)) -> None:
        q, v = self.qpos_at[piece.body], self.qvel_at[piece.body]
        self.data.qpos[q:q + 3] = position
        self.data.qpos[q + 3:q + 7] = quat
        self.data.qvel[v:v + 3] = velocity
        self.data.qvel[v + 3:v + 6] = self._local_spin(piece, spin)
        self.wake(piece)

    def shift(self, piece: Piece, offset) -> None:
        """Move `piece` by `offset`, keeping how it is turned and how it moves."""
        q = self.qpos_at[piece.body]
        self.data.qpos[q:q + 3] += np.asarray(offset, dtype=float)
        self.wake(piece)

    def _local_spin(self, piece: Piece, spin) -> np.ndarray:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, piece.quat())
        return rotation.reshape(3, 3).T @ np.asarray(spin, dtype=float)

    def _set_collisions(self, piece: Piece, bits: Tuple[int, int, int]) -> None:
        """(contype, conaffinity) of the piece's convex parts, and the conaffinity of its outline."""
        contype, conaffinity, outline = bits
        start = self.model.body_geomadr[piece.body]
        count = self.model.body_geomnum[piece.body]
        self.model.geom_conaffinity[start] = outline
        self.model.geom_contype[start + 1:start + count] = contype
        self.model.geom_conaffinity[start + 1:start + count] = conaffinity
        self.model.body_contype[piece.body] = contype
        self.model.body_conaffinity[piece.body] = conaffinity | outline
        self.body_bits[piece.body] = (contype, conaffinity | outline)

    def wake(self, piece: Piece) -> None:
        v = self.qvel_at[piece.body]
        if not np.any(self.data.qvel[v:v + 6]):
            self.data.qvel[v + 2] = WAKE_SPEED

    def pieces(self) -> List[Piece]:
        """In play, in the order they came in, which is their ids' order."""
        return list(self.by_body.values())

    def place_layout(self, layout: of.Layout) -> None:
        """Everything where `layout` has it, at rest; a Pin it has lying flat
        lies the way a Pin comes to rest, on its flange and the tip it points
        to.  Where the pieces come to rest is worked out once, on a field
        with no robot, and kept for the next field with the same layout."""
        if self.pins or self.cups:
            raise ValueError("a layout goes on an empty field")
        key = (self.goals, self.loaders, tuple(layout.pins), tuple(layout.cups))
        if key not in _AT_REST:
            _AT_REST[key] = setup_cache.kept("layout", (key, self.most),
                                             lambda: _come_to_rest(self.goals, self.loaders, self.most, layout))
        for kind, colors, position, quat in _AT_REST[key]:
            piece = self._bring_in(kind, colors, position, of.UP)
            self.place_turned(piece, position, quat)

    @staticmethod
    def standing_height(kind: str) -> float:
        return kinds()[kind].shape.half_length

    # ------------------------------------------------------------- loaders

    def set_loader(self, loader: of.Loader, raised: bool) -> None:
        self._move_loader(loader, raised)
        x0, x1, y0, y1, _, _ = of.loader_box(loader, raised=False)
        for piece in self.pieces():
            x, y, _ = piece.center()
            if x0 - piece.half_length < x < x1 + piece.half_length and y0 - piece.half_length < y < y1 + piece.half_length:
                self.wake(piece)

    def _move_loader(self, loader: of.Loader, raised: bool) -> None:
        self.data.mocap_pos[self._loader_mocap[loader]] = loader_foot(loader, raised)

    # ------------------------------------------------------------- stepping

    def step(self, dt: float = DT) -> None:
        if abs(dt - DT) > 1e-12:
            raise ValueError(f"pieces step every {DT} s, not {dt}")
        if self.has_robot:
            raise ValueError("a world with a robot in it is stepped by the robot's dynamics")
        self.out_of_reach.before_step()
        mujoco.mj_step(self.model, self.data)
        raise_on_warnings(self.model, self.data)
        self.after_step(dt)

    def after_step(self, dt: float) -> None:
        """What follows each step of the model: the clock, keeping awake what
        rests on the robot, and which Toggles the robot touches."""
        self.t += dt
        self.contact_geoms = self.data.contact.geom[:self.data.ncon].tolist()
        self._keep_awake_what_the_robot_touches()
        self.toggles.note_touches(self.contact_geoms, self.every_robot_geom)

    def _keep_awake_what_the_robot_touches(self) -> None:
        """A piece resting on the robot, or on pieces resting on it, shares
        its island, which never sleeps, so it never sleeps on its own count.
        Its count would run out while it rests there, and when the robot
        moves away it would fall asleep at once, in the air.  So its count
        starts again every step it touches the robot."""
        if not self.has_robot:
            return
        body_of = self.body_of_geom
        body_pairs = [(body_of[first], body_of[second]) for first, second in self.contact_geoms]
        reached = set(self._robot_body_set)
        spreading = True
        while spreading:
            spreading = False
            for first, second in body_pairs:
                for near, far in ((first, second), (second, first)):
                    if near in reached and far not in reached and far in self.by_body:
                        reached.add(far)
                        spreading = True
        for body in sorted(reached - self._robot_body_set):
            piece = self.by_body[body]
            tree = self.model.body_treeid[piece.body]
            if self.data.tree_asleep[tree] < 0:
                self.data.tree_asleep[tree] = -(mujoco.mjMINAWAKE + 1)
            else:
                self.wake(piece)

    def robot_overlaps(self) -> List[Tuple[object, float, Tuple[float, float, float]]]:
        """What the robot reaches into where it stands, further than
        OVERLAP_ALLOWED: (the Piece, or the field element's name, how deep,
        the deepest point), the deepest first, each thing once."""
        return self.overlaps(self.robot_geoms)

    def overlaps(self, geoms: Sequence[int], own_element: Optional[str] = None
                 ) -> List[Tuple[object, float, Tuple[float, float, float]]]:
        """What `geoms` reach into, as `robot_overlaps` gives it, leaving out
        the field element `own_element` they make up."""
        mujoco.mj_kinematics(self.model, self.data)
        targets = [(geom, name) for geom, name in self.elements.items() if name != own_element]
        for piece in self.pieces():
            start = self.model.body_geomadr[piece.body]
            targets += [(geom, piece) for geom in range(start + 1, start + self.model.body_geomnum[piece.body])]
        deepest: Dict[object, Tuple[object, float, Tuple[float, float, float]]] = {}
        fromto = np.zeros(6)
        target_geoms = np.array([geom for geom, _ in targets], dtype=int)
        target_bounds = self.model.geom_rbound[target_geoms]
        target_middles = self.data.geom_xpos[target_geoms]
        for geom in geoms:
            apart = np.sum((target_middles - self.data.geom_xpos[geom]) ** 2, axis=1)
            reach = self.model.geom_rbound[geom] + target_bounds
            for index in np.nonzero(apart <= reach ** 2)[0].tolist():
                other, what = targets[index]
                distance = mujoco.mj_geomDistance(self.model, self.data, int(geom), int(other), 0.0, fromto)
                if distance >= -OVERLAP_ALLOWED:
                    continue
                point = tuple(float(v) for v in (fromto[:3] + fromto[3:]) / 2.0)
                if what not in deepest or -distance > deepest[what][1]:
                    deepest[what] = (what, -distance, point)
        return sorted(deepest.values(), key=lambda found: -found[1])

    def settle(self, seconds: float = 2.0) -> None:
        for piece in self.pieces():
            self.wake(piece)
        for _ in range(int(round(seconds / DT))):
            self.step(DT)
            if all(piece.asleep for piece in self.pieces()):
                return

    # ------------------------------------------------------------- nesting

    def _up(self, piece: Piece) -> Tuple[np.ndarray, float]:
        """The piece's axis pointed up, and which end (+1 / -1) is up."""
        a = np.array(piece.axis())
        sign = 1.0 if a[2] >= 0.0 else -1.0
        return a * sign, sign

    @staticmethod
    def _inside(tip: np.ndarray, mouth: np.ndarray, inward: np.ndarray, radius: float, deepest: float) -> bool:
        """Whether `tip` is in the opening at `mouth` that goes `deepest` along `inward`."""
        rel = tip - mouth
        depth = float(rel @ inward)
        lateral = rel - inward * depth
        return NESTED_DEPTH <= depth <= deepest and float(np.linalg.norm(lateral)) < radius

    def nested(self, upper: Piece, lower) -> bool:
        """Whether `upper` is nested on `lower` (a Piece or a Goal)."""
        up, sign = self._up(upper)
        if isinstance(lower, of.Goal):
            if upper.kind != "pin" or up[2] < math.cos(NESTED_ANGLE):
                return False
            mouth = np.array([lower.x, lower.y, lower.height])
            return self._inside(upper.end(-sign), mouth, np.array([0.0, 0.0, -1.0]), of.GOAL_HOLE_DIAMETER / 2.0,
                                lower.height)
        if lower is upper or upper.kind == lower.kind:
            return False
        lower_up, lower_sign = self._up(lower)
        if float(up @ lower_up) < math.cos(NESTED_ANGLE):
            return False
        rim = of.CUP_RIM_DIAMETER / 2.0
        if upper.kind == "pin":
            return self._inside(upper.end(-sign), lower.end(lower_sign), -lower_up, rim, 2.0 * lower.half_length)
        return self._inside(lower.end(lower_sign), upper.end(-sign), up, rim, 2.0 * upper.half_length)

    def _near(self, piece: Piece) -> List[Piece]:
        """Pieces of the other kind close enough to piece to be nested with it."""
        others = self.cups if piece.kind == "pin" else self.pins
        if not others:
            return []
        centres = self.data.qpos[self._centres_at["cup" if piece.kind == "pin" else "pin"]]
        reach = piece.half_length + others[0].half_length
        close = np.nonzero(np.sum((centres - np.array(piece.center())) ** 2, axis=1) < reach ** 2)[0]
        return [others[i] for i in close]

    def _pins_over(self, goal: of.Goal) -> List[Piece]:
        """Pins close enough to `goal`'s axis to be standing in it."""
        if not self.pins:
            return []
        centres = self.data.qpos[self._centres_at["pin"][:, :2]]
        reach = self.pins[0].half_length + of.GOAL_HOLE_DIAMETER / 2.0
        close = np.nonzero(np.sum((centres - np.array([goal.x, goal.y])) ** 2, axis=1) < reach ** 2)[0]
        return [self.pins[i] for i in close]

    def below(self, piece: Piece):
        if piece.kind == "pin":
            for goal in self.goals:
                if self.nested(piece, goal):
                    return goal
        return next((other for other in self._near(piece) if self.nested(piece, other)), None)

    def above(self, piece: Piece) -> Optional[Piece]:
        return next((other for other in self._near(piece) if self.nested(other, piece)), None)

    def carried(self, piece: Piece) -> List[Piece]:
        """What is nested on `piece`, one on another, from it up."""
        out = []
        above = self.above(piece)
        while above is not None and above not in out:
            out.append(above)
            above = self.above(above)
        return out

    def goal_under(self, piece: Piece) -> Optional[of.Goal]:
        """The Goal whose hole the piece's lower end is over, if any."""
        _, sign = self._up(piece)
        tip = piece.end(-sign)
        return next((goal for goal in self.goals
                     if math.hypot(tip[0] - goal.x, tip[1] - goal.y) < of.GOAL_HOLE_DIAMETER / 2.0), None)

    def stacks(self) -> List[of.GoalStack]:
        result = []
        for goal in self.goals:
            bottom = next((p for p in self._pins_over(goal) if self.nested(p, goal)), None)
            if bottom is None:
                continue
            layers = []
            for piece in [bottom] + self.carried(bottom):
                _, sign = self._up(piece)
                if piece.kind == "pin":
                    lower, upper = (piece.colors[0], piece.colors[1]) if sign > 0.0 else (piece.colors[1], piece.colors[0])
                    layers.append(of.StackPin(lower, upper))
                else:
                    layers.append(of.StackCup(lower_opaque=sign > 0.0))
            result.append(of.GoalStack(goal.name, layers))
        return result


def _come_to_rest(goals, loaders, most: Dict[str, int], layout: of.Layout) -> List[tuple]:
    """(kind, colours, position, orientation) of each piece of `layout`
    after it has come to rest on a field with no robot."""
    field = Pieces(goals, loaders, most["pin"], most["cup"])
    for start in layout.cups:
        field.add_cup(start.position, start.axis)
    for start in layout.pins:
        (x, y, _), axis = start.position, start.axis
        if start.in_cup is None and start.in_goal is None:
            field.add_lying_pin(start.colors, x, y, math.atan2(axis[1], axis[0]))
        else:
            field.add_pin(start.colors, start.position, axis)
    field.settle(LAYOUT_SETTLE_SECONDS)
    restless = [p.id for p in field.pieces() if not p.asleep]
    if restless:
        raise RuntimeError(f"pieces {restless} of the layout did not come to rest in {LAYOUT_SETTLE_SECONDS} s")
    return [(p.kind, p.colors, p.center(), tuple(float(v) for v in p.quat())) for p in field.pieces()]


@dataclass(frozen=True)
class LyingRest:
    """A Pin at rest on one of its flats, flange and tip on the floor: the
    height of its middle, the tilt of its axis below level, how it sits when
    its axis heads along +x, and how far it reaches out to each side."""
    height: float
    tilt: float
    quat_heading_x: Tuple[float, float, float, float]
    half_width: float

    def quat(self, heading: float) -> Tuple[float, float, float, float]:
        turned = np.empty(4)
        mujoco.mju_mulQuat(turned, _yaw(heading), np.array(self.quat_heading_x))
        return tuple(float(v) for v in turned)

    def axis(self, heading: float) -> Tuple[float, float, float]:
        level = math.cos(self.tilt)
        return (level * math.cos(heading), level * math.sin(heading), -math.sin(self.tilt))


def _yaw(heading: float) -> np.ndarray:
    return np.array([math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0)])


@lru_cache(maxsize=None)
def lying_pin_rest() -> LyingRest:
    """Worked out by dropping a Pin."""
    world = Pieces(goals=(), loaders=(), pins=1, cups=0)
    pin = world.add_pin(("yellow", "yellow"), (0.0, 0.0, 2.0 * INCH), (1.0, 0.0, -0.05))
    world.settle(3.0)
    if not pin.asleep:
        raise RuntimeError("a lying Pin did not come to rest in 3 s")
    a = pin.axis()
    heading_x = np.empty(4)
    mujoco.mju_mulQuat(heading_x, _yaw(-math.atan2(a[1], a[0])), np.array(pin.quat()))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, heading_x)
    corners = np.vstack(kinds()["pin"].shape.hulls) @ rotation.reshape(3, 3).T
    return LyingRest(pin.center()[2], math.asin(max(-1.0, min(1.0, -a[2]))), tuple(float(v) for v in heading_x),
                     float(np.max(np.abs(corners[:, 1]))))

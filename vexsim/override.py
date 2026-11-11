"""The Override game around a simulated robot, all in one MuJoCo world.

The field (walls, Goals, Loaders and Toggles, shaped as VEX's field CAD has
them), the Pins and Cups, and the robot are bodies in one MuJoCo model
(`pieces.Pieces`).  The robot drives on `mujoco_drive.MujocoDrive`, which
takes the place of the simulator's dynamics: each simulator step runs the
battery and motors and then steps the whole world, so the robot pushes, is
stopped by and carries everything around it through MuJoCo's contacts.

The claw holds what its jaws squeeze, by friction.  A piece both jaws press
on is held; what is nested on it is held with it (<SG6>, Q&A 3166).

`match_load` is the Drive Team putting a Match Load in a Loader (<SG11>).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from . import override_cad as cad
from . import override_field as of
from .arena import ContactLog, circle_overlap
from .cascade import ROBOT, CascadeRobot, part_of
from .mujoco_drive import MujocoDrive, settle
from .pieces import WALLS, Piece, Pieces, RobotBuild
from .units import INCH

#: A jaw presses on a piece it holds with at least this, newtons
GRIP_FORCE = 0.5
#: The jaws are off a piece this long before it counts as let go; closing,
#: they bounce off a Cup for about 1 ms before they settle on it
LET_GO_AFTER = 0.02
#: How long the robot settles where it starts, and the preload in its claw, before the Match
START_SETTLE_SECONDS = 0.5
#: How far open of the preload the jaws start, so it is not put inside them
PRELOAD_JAW_CLEARANCE = 0.02 * INCH

GUESSES: Dict[str, str] = {
    "GRIP_FORCE": "0.5 N: both jaws press on a piece at least this hard while they hold it. The weight of a "
                  "Pin is 0.71 N, and the piston squeezes with about 48 N.",
    "LET_GO_AFTER": "20 ms: longer than the jaws' 1 ms bounce as they close on a Cup, shorter than the 0.3 s a "
                    "dropped Cup takes to fall to the floor from the lowest claw height.",
}


def goal_label(g: of.Goal) -> str:
    return f"the {g.name} Goal"


def loader_label(loader: of.Loader) -> str:
    return f"the {loader.name} Loader"


def toggle_outline(toggle: of.ToggleMount) -> List[Tuple[float, float]]:
    """The Toggle's bar at rest from above, for drawing: from its face toward
    the wall to its corner pointing into the field."""
    half_length = of.TOGGLE_LENGTH / 2.0
    corners = []
    for along, out in ((-half_length, -of.TOGGLE_FACE_REACH), (half_length, -of.TOGGLE_FACE_REACH),
                       (half_length, of.TOGGLE_CORNER_REACH), (-half_length, of.TOGGLE_CORNER_REACH)):
        corners.append((toggle.x + along * toggle.along_x + out * toggle.inward_x,
                        toggle.y + along * toggle.along_y + out * toggle.inward_y))
    return corners


@dataclass(frozen=True)
class Overlap:
    """Something the robot reaches into: a Piece or a field element's name,
    how deep, and the deepest point (field frame, metres)."""
    what: object
    depth: float
    point: Tuple[float, float, float]

    @property
    def name(self) -> str:
        return self.what.label if isinstance(self.what, Piece) else self.what


class StartOverlapError(ValueError):
    """The robot was put inside Pins, Cups or field elements, which would throw it out."""

    def __init__(self, overlaps: List[Overlap]):
        self.overlaps = overlaps
        super().__init__("the robot starts inside "
                         + "; ".join(f"{overlap.name}, {overlap.depth / INCH:.2f} in deep" for overlap in overlaps))


@dataclass
class Grip:
    """The piece the jaws hold, the pieces nested on it, one on another, and
    the entry in `OverrideWorld.grips` that noted the take."""
    piece: Piece
    carried: Tuple[Piece, ...]
    record: dict

    @property
    def mass(self) -> float:
        return self.piece.mass + sum(p.mass for p in self.carried)


def _robot_build(robot: CascadeRobot, chassis) -> RobotBuild:
    return RobotBuild(ROBOT, robot.model_key(chassis), lambda spec: robot.build(spec, chassis),
                      repr((robot.geometry, chassis)))


_AT_REST: Dict[tuple, np.ndarray] = {}


def robot_at_rest(robot: CascadeRobot, chassis) -> np.ndarray:
    """The robot's joint positions standing still on the floor at the
    middle of the field heading +x, with the lift down and the claw open
    (`MujocoDrive.robot_qpos`).  Worked out once per build, on a field with
    nothing on it."""
    key = robot.model_key(chassis)
    if key not in _AT_REST:
        alone = type(robot)(robot.geometry)
        field = Pieces(goals=(), loaders=(), pins=0, cups=0, robot=_robot_build(alone, chassis))
        drive = MujocoDrive(chassis, field.model, field.data, ROBOT, -float(field.model.opt.gravity[2]))
        mechanism = alone.attach(field.model, field.data, chassis)
        drive.forces.append(mechanism.apply)
        drive.after.append(mechanism.after)
        mujoco.mj_forward(field.model, field.data)
        settle(drive, START_SETTLE_SECONDS)
        rest = drive.robot_qpos()
        heading = drive.state.theta
        unturned = np.empty(4)
        mujoco.mju_mulQuat(unturned, np.array([math.cos(heading / 2.0), 0.0, 0.0, -math.sin(heading / 2.0)]),
                           rest[3:7].copy())
        rest[0:2] = 0.0
        rest[3:7] = unturned
        _AT_REST[key] = rest
    return _AT_REST[key].copy()


class OverrideWorld:
    def __init__(self, sim, robot: CascadeRobot, objects: bool = True, preload: Optional[str] = None,
                 alliance: str = "red"):
        """`robot` drives where `sim`'s pose has it.  `preload` is the colour
        of the half of the preload Pin that points up in the closed claw, or
        None for no preload.  Raises StartOverlapError if the robot is put
        inside anything but its preload."""
        self.sim = sim
        self.robot = robot
        self.geometry = robot.geometry
        self.alliance = alliance
        chassis = sim.chassis
        start = sim.state
        self.pieces = Pieces(robot=_robot_build(robot, chassis))
        if objects:
            self.pieces.place_layout(of.starting_layout())
        model, data = self.pieces.model, self.pieces.data
        self.model, self.data = model, data
        self.drive = MujocoDrive(chassis, model, data, ROBOT, sim.config.gravity)
        self.mechanism = robot.attach(model, data, chassis)
        self.drive.before.append(self.pieces.out_of_reach.before_step)
        self.drive.forces.append(self.mechanism.apply)
        self.drive.after += [self.mechanism.after, self._after_step]
        self.drive.place(robot_at_rest(robot, chassis), start.x, start.y, start.theta)
        sim.dynamics = self.drive
        sim.config.max_physics_dt = model.opt.timestep
        sim.power_mechanisms([motor for motors in self.mechanism.motors().values() for motor in motors])
        self.log = ContactLog(sim)
        self.touching_perimeter = False
        self._element_names = sorted(set(self.pieces.elements.values()))
        self._element_of = {geom: self._element_names.index(name) for geom, name in self.pieces.elements.items()}
        self._walls = {self._element_names.index(name) for name in WALLS if name in self._element_names}
        self._jaw_geoms = frozenset(geom for body in self.mechanism.jaw_bodies
                                    for geom in range(model.body_geomadr[body],
                                                      model.body_geomadr[body] + model.body_geomnum[body]))
        self.raised_loaders: List[str] = []
        self.grip: Optional[Grip] = None
        self._letting_go: Optional[dict] = None
        self.grips: List[dict] = []
        self.drops: List[dict] = []
        self.match_loads: List[dict] = []
        self.match_loads_left = {kind: count for kind, count in of.MATCH_LOADS[alliance].items()}
        self.preload: Optional[Piece] = None
        self._reported: Dict[int, tuple] = {}
        if preload is not None:
            self._preload(preload)
        overlaps = [Overlap(what, depth, point) for what, depth, point in self.pieces.robot_overlaps()
                    if what is not self.preload]
        if overlaps:
            raise StartOverlapError(overlaps)
        self._settle_start()

    def _settle_start(self) -> None:
        """Let the robot, and the preload in its claw, come to rest before the Match starts."""
        settle(self.drive, START_SETTLE_SECONDS)
        self.pieces.t = 0.0
        self.log = ContactLog(self.sim)
        self._follow_grip()
        self.grips.clear()
        self.drops.clear()

    # ------------------------------------------------------------- reports

    @property
    def contacts(self):
        return self.log.contacts

    def reported_contacts(self):
        return self.log.reported()

    def telemetry(self) -> dict:
        frame = self.log.telemetry()
        frame.update(self.robot.telemetry())
        tilt = self.drive.tilt
        frame.update(holding=None if self.grip is None else self.grip.piece.id, toggles=self.toggles.telemetry(),
                     raised_loaders=list(self.raised_loaders), cg_height_in=round(self.center_of_mass()[2] / INCH, 2),
                     tilt_deg=round(math.degrees(tilt.angle), 2), fallen=tilt.fallen)
        frame["pieces"] = self.moved_pieces()
        return frame

    def center_of_mass(self) -> np.ndarray:
        return self.data.subtree_com[self.drive.body].copy()

    def moved_pieces(self) -> Dict[int, list]:
        """Pieces that moved since the last call: id -> [x, y, z, ax, ay, az],
        metres, axis from the first half to the second."""
        moved = {}
        for piece in self.pieces.pieces():
            c, a = piece.center(), piece.axis()
            pose = (round(c[0], 4), round(c[1], 4), round(c[2], 4), round(a[0], 3), round(a[1], 3), round(a[2], 3))
            if self._reported.get(piece.id) != pose:
                self._reported[piece.id] = pose
                moved[piece.id] = list(pose)
        return moved

    def piece_list(self) -> List[dict]:
        return [dict(id=p.id, kind=p.kind, colors=list(p.colors)) for p in self.pieces.pieces()]

    def stacks(self):
        return self.pieces.stacks()

    @property
    def toggles(self):
        return self.pieces.toggles

    @property
    def toggle_colors(self) -> Dict[str, str]:
        return self.toggles.colors()

    def score(self, autonomous: bool = True):
        return of.score(self.stacks(), self.toggle_colors, {}, autonomous=autonomous)

    # ------------------------------------------------------------- stepping

    def step(self, dt: float) -> None:
        self.sim.step(dt)
        self._follow_grip()
        self.sim.chassis.cg_height = float(self.center_of_mass()[2])

    def _after_step(self, dt: float) -> None:
        """After each MuJoCo step: the pieces' bookkeeping, and the robot's
        contacts with the field and with what it holds."""
        self.pieces.after_step(dt)
        d = self.data
        self.touching_perimeter = False
        self.mechanism.squeeze = 0.0
        geom_pairs = self.pieces.contact_geoms
        robot = self.pieces.robot_geom_set
        if not any(first in robot or second in robot for first, second in geom_pairs):
            return
        solved = d.contact.efc_address[:d.ncon].tolist()
        force = np.zeros(6)
        strongest: Dict[int, Tuple[float, int, int]] = {}
        for i, (first, second) in enumerate(geom_pairs):
            if solved[i] < 0:
                continue
            if first in robot and second in self._element_of:
                robot_geom, element = first, self._element_of[second]
            elif second in robot and first in self._element_of:
                robot_geom, element = second, self._element_of[first]
            else:
                continue
            mujoco.mj_contactForce(self.model, d, i, force)
            if element in self._walls:
                self.touching_perimeter = True
            total, _, _ = strongest.get(element, (0.0, 0, 0))
            strongest[element] = (total + force[0], i, robot_geom)
        for element, (total, i, geom) in strongest.items():
            point = d.contact.pos[i]
            self.log.record(self._element_names[element], part_of(self.model.geom(geom).name),
                            float(point[0]), float(point[1]), total * dt)
        pressed = 0.0
        for i, (first, second) in enumerate(geom_pairs):
            if solved[i] >= 0 and (first in self._jaw_geoms or second in self._jaw_geoms):
                mujoco.mj_contactForce(self.model, d, i, force)
                pressed += force[0]
        self.mechanism.squeeze = pressed / 2.0

    # ------------------------------------------------------------- the claw

    def _gripped(self) -> Optional[Piece]:
        """The lowest piece the jaws press on in a stack both jaws press on
        (each jaw pressing some piece of it).  A Pin both jaws press on in a
        Cup no jaw touches is held on its own."""
        d = self.data
        pressing: Dict[int, set] = {}
        force = np.zeros(6)
        jaws = set(self.mechanism.jaw_bodies)
        body_of = self.pieces.body_of_geom
        solved = d.contact.efc_address[:d.ncon].tolist()
        for i, (first, second) in enumerate(d.contact.geom[:d.ncon].tolist()):
            b0, b1 = body_of[first], body_of[second]
            jaw, other = (b0, b1) if b0 in jaws else (b1, b0) if b1 in jaws else (None, None)
            if jaw is None or solved[i] < 0:
                continue
            mujoco.mj_contactForce(self.model, d, i, force)
            if force[0] >= GRIP_FORCE:
                pressing.setdefault(other, set()).add(jaw)
        by_body = self.pieces.by_body
        stacks: Dict[int, Tuple[set, List[Piece]]] = {}
        for body, jaws in pressing.items():
            if body not in by_body:
                continue
            piece = by_body[body]
            jaws_on, pressed = stacks.setdefault(self._stack_bottom(piece).id, (set(), []))
            jaws_on |= jaws
            pressed.append(piece)
        held = [min(pressed, key=lambda piece: piece.center()[2]) for jaws_on, pressed in stacks.values()
                if len(jaws_on) == 2]
        return min(held, key=lambda piece: piece.center()[2]) if held else None

    def _stack_bottom(self, piece: Piece) -> Piece:
        """The piece at the bottom of the stack `piece` is nested in."""
        below = self.pieces.below(piece)
        while isinstance(below, Piece):
            piece, below = below, self.pieces.below(below)
        return piece

    def _follow_grip(self) -> None:
        """Note what the claw takes and lets go.  A piece counts as let go
        once the jaws have been off it for LET_GO_AFTER.  The jaws hold a
        stack by its Pin, its Cup or both, and which of them is the lowest
        they press can change while they hold it: that is neither a take
        nor a drop.  A piece that was with the claw and is no longer in the
        stack it holds has slipped out.  When the jaws, closing on a stack,
        reach its lower piece within LET_GO_AFTER of taking its upper one,
        the take is noted as of the lower piece.  With the valve set to
        open the claw only lets go: it takes nothing and does not move its
        hold to another piece of the stack, so jaws opening off a stack's
        Cup before its Pin, or brushing it as it settles, let go of the
        stack."""
        held = self._gripped()
        grip = self.grip
        if not self.robot.claw.closed and (grip is None or held is not grip.piece):
            held = None
        if grip is not None and held is not None and held is not grip.piece and self._same_stack(held, grip):
            carried = tuple(self.pieces.carried(held))
            self._note_slipped(grip, held, carried)
            if grip.piece in carried and self.sim.t - grip.record["t"] < LET_GO_AFTER:
                grip.record.update(piece=held.id, at=held.center(), stacked=[p.id for p in carried])
            grip.piece, grip.carried = held, carried
        if grip is not None and held is not grip.piece:
            if self._letting_go is None:
                reason = "the claw opened" if not self.robot.claw.closed else "it slipped out of the jaws"
                self._letting_go = dict(t=self.sim.t, piece=grip.piece.id, reason=reason, at=grip.piece.center(),
                                        stacked=[p.id for p in grip.carried])
            if self.sim.t - self._letting_go["t"] < LET_GO_AFTER:
                return
            self.drops.append(self._letting_go)
            self.grip = None
        self._letting_go = None
        if held is not None:
            carried = tuple(self.pieces.carried(held))
            if self.grip is None:
                record = dict(t=self.sim.t, piece=held.id, at=held.center(), stacked=[p.id for p in carried])
                self.grips.append(record)
                self.grip = Grip(held, carried, record)
            else:
                self._note_slipped(self.grip, held, carried)
                self.grip.carried = carried

    def _same_stack(self, held: Piece, grip: Grip) -> bool:
        """Whether `held` is the piece the claw holds or one nested on or under it."""
        return held in grip.carried or grip.piece in self.pieces.carried(held)

    def _note_slipped(self, grip: Grip, held: Piece, carried: Tuple[Piece, ...]) -> None:
        """Note as dropped each piece that was with the claw and is no longer in the stack of `held`."""
        stack = {held, *carried}
        below = self.pieces.below(held)
        while isinstance(below, Piece):
            stack.add(below)
            below = self.pieces.below(below)
        for piece in (grip.piece, *grip.carried):
            if piece not in stack:
                self.drops.append(dict(t=self.sim.t, piece=piece.id, reason="it slipped out of the jaws",
                                       at=piece.center(), stacked=[]))

    def _preload(self, preload: str) -> None:
        """The preload Pin hangs in the closed claw by its flange: put it
        there, on the robot as it stands, with the jaws just open of it,
        and let them close on it."""
        colors = of.PRELOAD_COLORS[self.alliance]
        (forward, left, up), axis, _ = self.robot.preload_pose(self.sim.chassis, preload, colors)
        body = self.drive.body
        turn = self.data.xmat[body].reshape(3, 3)
        rise = self.mechanism.lift.rise
        position = self.data.xpos[body] + turn @ np.array([forward, left, up + rise])
        self.preload = self.pieces.add_pin(colors, tuple(float(v) for v in position), tuple(turn @ np.array(axis)))
        self.mechanism.close_on(self.robot.preload_gap() + PRELOAD_JAW_CLEARANCE)
        self.robot.claw.hold(True)

    # ------------------------------------------------------------- the Drive Team

    def match_load(self, loader_name: str, pin_colors: Optional[Tuple[str, str]] = None,
                   upper_color: str = "yellow", cup_gray_up: bool = True) -> Piece:
        """The Drive Team lifts a Loader by its handle and stands a Cup with a
        Pin in it on the floor under it, through the Loader's back (<SG11b>).
        The Loader stays up.  The Pin is one of the alliance's Pins unless
        `pin_colors` says otherwise, with its `upper_color` half up.  Returns
        the Cup."""
        loader = of.loader(loader_name)
        colors = of.PRELOAD_COLORS[self.alliance] if pin_colors is None else tuple(pin_colors)
        if upper_color not in colors:
            raise ValueError(f"a {colors[0]}/{colors[1]} Pin has no {upper_color} half")
        for kind, name in (("cup", "Cups"), (colors, f"{colors[0]}/{colors[1]} Pins")):
            if self.match_loads_left.get(kind, 0) < 1:
                raise ValueError(f"the {self.alliance} alliance has no {name} left to load")
        if any(self._under_loader(loader, piece) for piece in self.pieces.pieces()):
            raise ValueError(f"something already stands under {loader_label(loader)}")
        if self._robot_reaches(loader.x, loader.y, of.CUP_RIM_DIAMETER / 2.0, of.CUP_LENGTH):
            raise ValueError(f"the robot is where the Cup would stand under {loader_label(loader)}")
        self.match_loads_left["cup"] -= 1
        self.match_loads_left[colors] -= 1
        self._set_loader(loader, raised=True)
        cup_middle = self.pieces.standing_height("cup")
        cup = self.pieces.add_cup((loader.x, loader.y, cup_middle), of.DOWN if cup_gray_up else of.UP)
        pin_axis = of.UP if colors[1] == upper_color else of.DOWN
        pin = self.pieces.add_pin(colors, (loader.x, loader.y, cup_middle + cad.seats().pin_in_cup), pin_axis)
        self.match_loads.append(dict(t=self.sim.t, loader=loader.name, cup=cup.id, pin=pin.id))
        return cup

    def _robot_reaches(self, x: float, y: float, radius: float, height: float) -> bool:
        """Whether any part of the robot lower than `height` is within `radius` of (x, y), seen from above."""
        st = self.drive.state
        c, s = math.cos(st.theta), math.sin(st.theta)
        for part in self.robot.parts(self.sim.chassis):
            if part.bottom >= height:
                continue
            outline = [(st.x + px * c - py * s, st.y + px * s + py * c) for px, py in part.corners()]
            if circle_overlap(outline, x, y, radius) is not None:
                return True
        return False

    def lower_loader(self, loader_name: str) -> None:
        loader = of.loader(loader_name)
        if any(self._under_loader(loader, piece) for piece in self.pieces.pieces()):
            raise ValueError(f"a Pin or Cup is under {loader_label(loader)}; lowering it would land on it")
        self._set_loader(loader, raised=False)

    def _set_loader(self, loader: of.Loader, raised: bool) -> None:
        self.raised_loaders = [name for name in self.raised_loaders if name != loader.name]
        if raised:
            self.raised_loaders.append(loader.name)
        self.pieces.set_loader(loader, raised)

    @staticmethod
    def _under_loader(loader: of.Loader, piece: Piece) -> bool:
        x0, x1, y0, y1, _, _ = of.loader_box(loader, raised=False)
        return any(x0 - r < c[0] < x1 + r and y0 - r < c[1] < y1 + r for c, r in piece.spheres())

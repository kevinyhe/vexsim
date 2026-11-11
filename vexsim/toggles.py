"""The four Toggles: triangular bars on hubs in slotted brackets over the walls.

Each Toggle is shaped as VEX's field CAD has it (`override_cad.toggle_shape`):
a bar with rounded corners, an end cap on each end whose hub rides in a slot
in a bracket bent over the wall's top, and a plate on each bracket sloping up
toward the field that the bar rests a face on.  The bar is a MuJoCo body
that slides up the slots and turns about the hubs; nothing about how it
turns is scripted.  Turned off its face, a corner presses on the plate and
lifts the hubs up the slots, so gravity pulls it back to the nearest face
until it is pushed past the corner, and then into the next one.

Angles are in the wall's cross-section, counter-clockwise from the
direction into the field, so a positive turn tips the top of the bar toward
the wall.  At the start the bar's faces point 60, 180 and 300 degrees
(`override_field.TOGGLE_FACE_ANGLES`).
"""

from __future__ import annotations

import math
from typing import AbstractSet, Callable, Dict, List, Sequence

import mujoco
import numpy as np

from . import override_cad as cad
from . import override_field as of
from .units import INCH

#: N m s at the hub
TOGGLE_DAMPING = 2.0e-3
#: The bar and its end caps on the brackets' plates
BRACKET_MU = 0.3

GUESSES: Dict[str, str] = {
    "TOGGLE_DAMPING": "2e-3 N m s at the hub, so a dropped bar stops rocking in about 0.1 s.",
    "BRACKET_MU": "The robot's owner, 2026-11-04/05: 0.3, the bar's plastic on the brackets' steel. A real bar "
                  "left at 90 degrees stays put.",
}


def mount_quat(mount: of.ToggleMount) -> List[float]:
    """The mount frame: x into the field, y along the wall, z up."""
    rotation = np.array([[mount.inward_x, mount.along_x, 0.0],
                         [mount.inward_y, mount.along_y, 0.0],
                         [0.0, 0.0, 1.0]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.flatten())
    return quat.tolist()


def bar_name(mount: of.ToggleMount) -> str:
    return f"toggle {mount.quadrant}"


def add_toggles(spec, contact: Callable, solref: Sequence[float], element_prefix: str, static_bit: int,
                robot_bit: int, mounts: Sequence[of.ToggleMount] = of.TOGGLES) -> None:
    """Each Toggle's brackets as part of the field and its bar as a body on
    a slide up the slots and a hinge about the hubs.  `contact(geom, mu)`
    sets a geom's contact settings, and the slots' ends are as stiff as
    `solref` makes a contact.  The bar collides with the brackets, the
    robot and the pieces; its hubs only with the robot and the pieces, since
    the slide is their slot."""
    shape = cad.toggle_shape()
    brackets = [_mesh(spec, f"toggle bracket {k}", points) for k, points in enumerate(shape.brackets)]
    bar_mesh = _mesh(spec, "toggle bar", shape.bar)
    hubs = [_mesh(spec, f"toggle hub {k}", points) for k, points in enumerate(shape.hubs)]
    inertia = shape.inertia_per_volume * INCH ** 2 * of.TOGGLE_MASS
    for mount in mounts:
        quat = mount_quat(mount)
        axle = [mount.x, mount.y, of.TOGGLE_AXLE_HEIGHT]
        label = f"{element_prefix}the {mount.quadrant} Toggle"
        for k, name in enumerate(brackets):
            geom = spec.worldbody.add_geom(name=f"{label}|bracket {k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                           pos=axle, quat=quat, contype=static_bit, conaffinity=0)
            contact(geom, BRACKET_MU)
        body = spec.worldbody.add_body(name=bar_name(mount), pos=axle, quat=quat)
        slide = body.add_joint(name=f"{bar_name(mount)} slot", type=mujoco.mjtJoint.mjJNT_SLIDE,
                               axis=[shape.slot_direction[0], 0.0, shape.slot_direction[1]],
                               range=[0.0, shape.slot_travel * INCH])
        slide.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        slide.solref_limit = list(solref)
        hinge = body.add_joint(name=f"{bar_name(mount)} hub", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, -1.0, 0.0])
        hinge.damping = [TOGGLE_DAMPING, 0.0, 0.0]
        body.explicitinertial = True
        body.mass = of.TOGGLE_MASS
        body.ipos = (shape.centroid * INCH).tolist()
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        geom = body.add_geom(name=f"{label}|bar", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=bar_mesh,
                             contype=static_bit, conaffinity=static_bit | robot_bit)
        contact(geom, BRACKET_MU)
        for k, name in enumerate(hubs):
            geom = body.add_geom(name=f"{label}|hub {k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                 contype=static_bit, conaffinity=robot_bit)
            contact(geom, BRACKET_MU)


def _mesh(spec, name: str, points_in: np.ndarray) -> str:
    spec.add_mesh(name=name, uservert=(np.asarray(points_in, dtype=float) * INCH).flatten().tolist())
    return name


class Toggle:
    """One Toggle's bar in a model `add_toggles` built."""

    def __init__(self, mount: of.ToggleMount, model: mujoco.MjModel, data: mujoco.MjData):
        self.mount = mount
        self.data = data
        self.body = model.body(bar_name(mount)).id
        slide, hinge = model.joint(f"{bar_name(mount)} slot").id, model.joint(f"{bar_name(mount)} hub").id
        self._slide = model.jnt_qposadr[slide]
        self._hinge = model.jnt_qposadr[hinge]
        self._hinge_speed = model.jnt_dofadr[hinge]
        start = model.body_geomadr[self.body]
        self.geoms = np.arange(start, start + model.body_geomnum[self.body])
        self.touched = False

    @property
    def name(self) -> str:
        return f"the {self.mount.quadrant} Toggle"

    @property
    def angle(self) -> float:
        return float(self.data.qpos[self._hinge])

    @property
    def rate(self) -> float:
        return float(self.data.qvel[self._hinge_speed])

    @property
    def lift(self) -> float:
        """How far the hubs have risen up the slots, metres."""
        return float(self.data.qpos[self._slide])

    def color(self) -> str:
        return of.toggle_color(self.mount.faces, self.angle, self.touched)

    def inner_face(self) -> str:
        return of.toggle_inner_face(self.mount.faces, self.angle)

    def seated(self) -> bool:
        step = 2.0 * math.pi / 3.0
        return abs(self.angle - step * round(self.angle / step)) <= of.TOGGLE_SEATED_TOLERANCE


class Toggles:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.bars = [Toggle(mount, model, data) for mount in of.TOGGLES]
        self._bar_of: Dict[int, Toggle] = {geom: bar for bar in self.bars for geom in bar.geoms.tolist()}

    def note_touches(self, geom_pairs: Sequence[Sequence[int]], robot_geoms: AbstractSet[int]) -> None:
        """Mark each bar the robot touches in this step's contacts (<SC4>), given as the geoms each touches."""
        for bar in self.bars:
            bar.touched = False
        for first, second in geom_pairs:
            for robot_side, other in ((first, second), (second, first)):
                if robot_side in robot_geoms and other in self._bar_of:
                    self._bar_of[other].touched = True

    def colors(self) -> Dict[str, str]:
        return {bar.mount.quadrant: bar.color() for bar in self.bars}

    def telemetry(self) -> Dict[str, dict]:
        return {bar.mount.quadrant: dict(inner=bar.inner_face(), set=bar.color(),
                                         angle_deg=round(math.degrees(bar.angle), 1), touched=bar.touched,
                                         lift_in=round(bar.lift / INCH, 3))
                for bar in self.bars}

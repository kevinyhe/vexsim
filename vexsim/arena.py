"""A robot against fixed field elements: the perimeter, Goals, Loaders.

An element is a circle or a convex polygon in plan with a height span.  The
robot is boxes in its own frame, each with its own height span, so a part
above an element passes over it.  A contact moves the robot out, removes its
velocity into the element and applies Coulomb friction along it.  The wheels
are not told: they keep turning, slip, and the motors draw stall current,
which is what happens on a real field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .units import INCH, clamp

GUESSES: Dict[str, str] = {
    "Contact friction": "0.35 between the robot and a field element: plastic and aluminium on polycarbonate and "
                        "steel.",
    "Hit force window": "Reported forces are the impulse over 20 ms, about how long a frame and bumper take to give.",
    "Instant stop": "A hit takes all the robot's speed into the element in one step and it does not bounce. No "
                    "one has published a measurement of a VEX robot hitting a wall. Tried in the simulator with "
                    "a spring and damper that stop the robot over 10 or 20 ms instead: after driving into a wall "
                    "from 25 in away, odometry was off by 2.30 or 2.26 in at 8 V where the instant stop gives "
                    "2.49 in, and 5.87 or 5.74 in at 12 V where it gives 5.79 in. The wheels spinning on "
                    "against the wall make that error, not how fast the frame stops.",
    "Hits do not rock the robot": "A hit below the CG would rock the robot up onto the wheels nearest the "
                                  "element, but its frame is against the element's face, so it can only rock "
                                  "with its wheels sliding back on the floor and its frame sliding up the face. "
                                  "That needs both contacts at once, which this flat model does not have. "
                                  "Rocking it about its wheels alone would make it fall more often than a real "
                                  "robot does.",
}

CONTACT_MU = 0.35
#: a contact that lets go for less than this is the same contact
CONTACT_GAP = 0.15
#: below this a touch is not reported
REPORT_FORCE = 5.0
#: Contacts here are rigid, so a robot hitting a wall loses its speed in one
#: step and the force of that one step is meaningless.  Reported forces are
#: the impulse over this long, about how long a real frame and bumper take to
#: give and spring back.
FORCE_WINDOW = 0.020


@dataclass(frozen=True)
class Part:
    """A box in the robot's frame: x forward, y left, z up, metres.  A part
    that moves on the robot moves `forward_speed` forward and `rise_speed`
    up relative to the drivetrain.  A part with an `outline` is that convex
    polygon (counter-clockwise) in plan instead of the box; its box is the
    outline's bounds.  A part with a `profile` is that convex polygon
    (counter-clockwise in forward and up) seen from the side, reaching from
    `right` to `left`; its box is the profile's bounds.  `passes_through`
    names field elements the part goes into rather than hits, such as the
    Goal a held Pin is going into."""
    name: str
    back: float
    front: float
    right: float
    left: float
    bottom: float
    top: float
    forward_speed: float = 0.0
    rise_speed: float = 0.0
    outline: Tuple[Tuple[float, float], ...] = ()
    passes_through: Tuple[str, ...] = ()
    profile: Tuple[Tuple[float, float], ...] = ()

    def corners(self) -> List[Tuple[float, float]]:
        if self.outline:
            return list(self.outline)
        return [(self.back, self.right), (self.front, self.right),
                (self.front, self.left), (self.back, self.left)]

    def vertices(self) -> List[Tuple[float, float, float]]:
        """The corners of the solid, robot frame."""
        if self.profile:
            return [(x, y, z) for x, z in self.profile for y in (self.right, self.left)]
        return [(x, y, z) for x, y in self.corners() for z in (self.bottom, self.top)]

    def shifted(self, up: float, left: float, rise_speed: float) -> "Part":
        """The same part moved `up` and `left`, rising at `rise_speed`."""
        return replace(self, right=self.right + left, left=self.left + left, bottom=self.bottom + up,
                       top=self.top + up, rise_speed=rise_speed,
                       outline=tuple((x, y + left) for x, y in self.outline),
                       profile=tuple((x, z + up) for x, z in self.profile))


def outline_part(name: str, points: Sequence[Tuple[float, float]], bottom: float, top: float,
                 forward_speed: float = 0.0, rise_speed: float = 0.0,
                 passes_through: Tuple[str, ...] = ()) -> Part:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Part(name, min(xs), max(xs), min(ys), max(ys), bottom, top, forward_speed, rise_speed,
                tuple(tuple(p) for p in points), passes_through)


def profile_part(name: str, points: Sequence[Tuple[float, float]], right: float, left: float,
                 forward_speed: float = 0.0, rise_speed: float = 0.0) -> Part:
    """A bar seen from the side: `points` is its convex outline in (forward, up), counter-clockwise."""
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    return Part(name, min(xs), max(xs), right, left, min(zs), max(zs), forward_speed, rise_speed,
                profile=tuple(tuple(p) for p in points))


@dataclass(frozen=True)
class Obstacle:
    name: str
    bottom: float
    top: float
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    points: Tuple[Tuple[float, float], ...] = ()
    #: distance from (x, y) to the furthest point of the element
    extent: float = 0.0
    #: a cone instead of a cylinder: the radius at `bottom` (`radius` is the one at `top`)
    bottom_radius: float = 0.0

    @property
    def is_circle(self) -> bool:
        return not self.points

    def radius_above(self, z: float) -> float:
        """The widest the element is at or above height `z`."""
        if not self.bottom_radius:
            return self.radius
        t = min(1.0, max(0.0, (z - self.bottom) / (self.top - self.bottom)))
        return self.bottom_radius + t * (self.radius - self.bottom_radius)


def circle(name: str, x: float, y: float, radius: float, bottom: float, top: float) -> Obstacle:
    return Obstacle(name, bottom, top, x=x, y=y, radius=radius, extent=radius)


def cone(name: str, x: float, y: float, bottom_radius: float, top_radius: float,
         bottom: float, top: float) -> Obstacle:
    return Obstacle(name, bottom, top, x=x, y=y, radius=top_radius, extent=max(bottom_radius, top_radius),
                    bottom_radius=bottom_radius)


def polygon(name: str, points: Sequence[Tuple[float, float]], bottom: float, top: float) -> Obstacle:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    extent = max(math.hypot(px - cx, py - cy) for px, py in points)
    return Obstacle(name, bottom, top, x=cx, y=cy, points=tuple(points), extent=extent)


def rectangle(name: str, x0: float, x1: float, y0: float, y1: float,
              bottom: float, top: float) -> Obstacle:
    return polygon(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], bottom, top)


@dataclass
class Contact:
    """One continuous touch between the robot and a field element."""
    element: str
    part: str
    x: float
    y: float
    start: float
    end: float
    peak_force: float = 0.0
    peak_current: float = 0.0
    peak_x: float = 0.0
    peak_y: float = 0.0
    force: float = 0.0
    window: list = dc_field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(element=self.element, part=self.part, x=self.x, y=self.y, start=self.start,
                    end=self.end, peak_force=self.peak_force, peak_current=self.peak_current)


def drivetrain_parts(chassis, bottom: float = 0.4 * INCH, top: float = 5.0 * INCH) -> List[Part]:
    hl, hw = chassis.length / 2.0, chassis.width / 2.0
    return [Part("drivetrain", -hl, hl, -hw, hw, bottom, top)]


def _project(points, ax, ay):
    values = [px * ax + py * ay for px, py in points]
    return min(values), max(values)


def _inside(point, poly) -> bool:
    px, py = point
    sign = 0.0
    for i, (ax, ay) in enumerate(poly):
        bx, by = poly[(i + 1) % len(poly)]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross * sign < 0.0:
            return False
        if cross != 0.0:
            sign = cross
    return True


def polygon_overlap(robot_poly, element_poly):
    """Separating axis test.  Returns (depth, nx, ny, px, py) with n pointing
    from the robot into the element, or None."""
    best = None
    for poly in (robot_poly, element_poly):
        for i, (ax, ay) in enumerate(poly):
            bx, by = poly[(i + 1) % len(poly)]
            ex, ey = bx - ax, by - ay
            length = math.hypot(ex, ey)
            if length < 1e-12:
                continue
            nx, ny = ey / length, -ex / length
            rmin, rmax = _project(robot_poly, nx, ny)
            emin, emax = _project(element_poly, nx, ny)
            overlap = min(rmax, emax) - max(rmin, emin)
            if overlap <= 0.0:
                return None
            if best is None or overlap < best[0]:
                into = 1.0 if (emin + emax) > (rmin + rmax) else -1.0
                best = (overlap, nx * into, ny * into)
    depth, nx, ny = best
    inside = [p for p in robot_poly if _inside(p, element_poly)]
    inside += [p for p in element_poly if _inside(p, robot_poly)]
    if inside:
        px = sum(p[0] for p in inside) / len(inside)
        py = sum(p[1] for p in inside) / len(inside)
    else:
        deepest_robot = max(robot_poly, key=lambda p: p[0] * nx + p[1] * ny)
        deepest_element = min(element_poly, key=lambda p: p[0] * nx + p[1] * ny)
        px = 0.5 * (deepest_robot[0] + deepest_element[0])
        py = 0.5 * (deepest_robot[1] + deepest_element[1])
    return depth, nx, ny, px, py


def circle_overlap(robot_poly, cx, cy, radius):
    """A circle against a convex polygon (counter-clockwise)."""
    best_d2, best = None, None
    for i, (ax, ay) in enumerate(robot_poly):
        bx, by = robot_poly[(i + 1) % len(robot_poly)]
        ex, ey = bx - ax, by - ay
        t = clamp(((cx - ax) * ex + (cy - ay) * ey) / max(ex * ex + ey * ey, 1e-18), 0.0, 1.0)
        qx, qy = ax + t * ex, ay + t * ey
        d2 = (cx - qx) ** 2 + (cy - qy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best = d2, (qx, qy)
    qx, qy = best
    distance = math.sqrt(best_d2)
    if _inside((cx, cy), robot_poly):
        if distance < 1e-12:
            return None
        nx, ny = (qx - cx) / distance, (qy - cy) / distance
        return radius + distance, -nx, -ny, qx, qy
    if distance >= radius:
        return None
    nx, ny = (cx - qx) / distance, (cy - qy) / distance
    return radius - distance, nx, ny, qx, qy


class ContactLog:
    """Each continuous touch between the robot and a field element, with the
    force over the last FORCE_WINDOW and the drive's current at its peak."""

    def __init__(self, sim):
        self.sim = sim
        self.contacts: List[Contact] = []
        self._open: dict = {}

    def drive_current(self) -> float:
        return sum(abs(m.current) for m in self.sim.motors)

    def record(self, element: str, part: str, x: float, y: float, impulse: float) -> None:
        t = self.sim.t
        current = self.drive_current()
        contact = self._open.get(element)
        if contact is None or t - contact.end > CONTACT_GAP:
            contact = Contact(element, part, x, y, t, t)
            self._open[element] = contact
            self.contacts.append(contact)
        contact.end = t
        contact.peak_current = max(contact.peak_current, current)
        contact.window.append((t, impulse))
        while contact.window[0][0] <= t - FORCE_WINDOW:
            contact.window.pop(0)
        force = sum(i for _, i in contact.window) / FORCE_WINDOW
        contact.force = force
        if force > contact.peak_force:
            contact.peak_force, contact.peak_x, contact.peak_y = force, x, y

    def reported(self) -> List[Contact]:
        return [c for c in self.contacts if c.peak_force >= REPORT_FORCE]

    def telemetry(self) -> dict:
        t = self.sim.t
        touching = [c for c in self._open.values() if t - c.end <= 0.002]
        if not touching:
            return dict(contact=None, contact_force_n=0.0)
        strongest = max(touching, key=lambda c: c.force)
        return dict(contact=strongest.element, contact_force_n=round(strongest.force, 1))


class Arena:
    """Steps a `Simulator` and keeps its robot out of the field elements."""

    def __init__(self, sim, obstacles: Sequence[Obstacle], half_size: Optional[float],
                 wall_height: float, parts: Callable[[], List[Part]]):
        self.sim = sim
        self.obstacles = list(obstacles)
        self.half_size = half_size
        self.wall_height = wall_height
        self.parts = parts
        self.log = ContactLog(sim)
        self.touching_perimeter = False

    @property
    def contacts(self) -> List[Contact]:
        return self.log.contacts

    def step(self, dt: float) -> None:
        self.sim.step(dt)
        self.resolve(dt)

    def velocity_at(self, x: float, y: float, forward_speed: float = 0.0) -> Tuple[float, float]:
        """World velocity of a point on the robot, plus `forward_speed` along
        the robot's heading for a part moving on it."""
        st = self.sim.state
        c, s = math.cos(st.theta), math.sin(st.theta)
        return (st.u * c - st.v * s - st.r * (y - st.y) + forward_speed * c,
                st.u * s + st.v * c + st.r * (x - st.x) + forward_speed * s)

    def robot_polygon(self, part: Part):
        return [self.to_world(px, py) for px, py in part.corners()]

    def to_world(self, x: float, y: float) -> Tuple[float, float]:
        """A point in the robot frame, on the field."""
        st = self.sim.state
        c, s = math.cos(st.theta), math.sin(st.theta)
        return st.x + x * c - y * s, st.y + x * s + y * c

    def resolve(self, dt: float) -> None:
        self.touching_perimeter = False
        for part in self.parts():
            if self.half_size is not None and part.bottom < self.wall_height:
                self._perimeter(part, dt)
            poly = self.robot_polygon(part)
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            extent = max(math.hypot(px - cx, py - cy) for px, py in poly)
            for element in self.obstacles:
                if part.top <= element.bottom or part.bottom >= element.top:
                    continue
                if element.name in part.passes_through:
                    continue
                if math.hypot(element.x - cx, element.y - cy) > element.extent + extent:
                    continue
                if element.is_circle:
                    hit = circle_overlap(poly, element.x, element.y, element.radius_above(part.bottom))
                else:
                    hit = polygon_overlap(poly, list(element.points))
                if hit is not None:
                    self.push_out(element.name, part.name, hit, dt)
                    poly = self.robot_polygon(part)
                    cx = sum(p[0] for p in poly) / len(poly)
                    cy = sum(p[1] for p in poly) / len(poly)

    def _perimeter(self, part: Part, dt: float) -> None:
        poly = self.robot_polygon(part)
        for axis in (0, 1):
            for sign in (1.0, -1.0):
                deepest = max(poly, key=lambda p: sign * p[axis])
                depth = sign * deepest[axis] - self.half_size
                if depth <= 0.0:
                    continue
                nx, ny = (sign, 0.0) if axis == 0 else (0.0, sign)
                touching = [p for p in poly if sign * p[axis] - self.half_size > -1e-4]
                px = sum(p[0] for p in touching) / len(touching)
                py = sum(p[1] for p in touching) / len(touching)
                self.touching_perimeter = True
                self.push_out("the wall", part.name, (depth, nx, ny, px, py), dt)
                poly = self.robot_polygon(part)

    def push_out(self, element: str, part: str, hit, dt: float, surface_speed: float = 0.0) -> None:
        """Move the robot `depth` out of the element along -n and take away its
        speed into it.  `surface_speed` is how fast the element's surface
        where they touch moves along n, away from the robot: a turning
        Toggle that moves away as fast as the robot takes none of its speed."""
        depth, nx, ny, px, py = hit
        st = self.sim.state
        st.x -= nx * depth
        st.y -= ny * depth
        px -= nx * depth
        py -= ny * depth
        c, s = math.cos(st.theta), math.sin(st.theta)
        rx, ry = px - st.x, py - st.y
        bx, by = rx * c + ry * s, -rx * s + ry * c
        bnx, bny = nx * c + ny * s, -nx * s + ny * c
        impulse = self.static_contact(bx, by, bnx, bny, surface_speed)
        self.log.record(element, part, px, py, impulse)

    def static_contact(self, px: float, py: float, nx: float, ny: float, surface_speed: float = 0.0) -> float:
        """Inelastic normal impulse then Coulomb friction, body frame, n into
        the element, whose surface moves along n at `surface_speed`.
        Returns the normal impulse."""
        st, ch = self.sim.state, self.sim.chassis
        arm_n = px * ny - py * nx
        into = (st.u - st.r * py) * nx + (st.v + st.r * px) * ny - surface_speed
        if into <= 0.0:
            return 0.0
        normal = into / (1.0 / ch.mass + arm_n * arm_n / ch.inertia_z)
        st.u -= normal * nx / ch.mass
        st.v -= normal * ny / ch.mass
        st.r -= normal * arm_n / ch.inertia_z
        tx, ty = -ny, nx
        arm_t = px * ty - py * tx
        slip = (st.u - st.r * py) * tx + (st.v + st.r * px) * ty
        tangent = clamp(-slip / (1.0 / ch.mass + arm_t * arm_t / ch.inertia_z),
                        -CONTACT_MU * normal, CONTACT_MU * normal)
        st.u += tangent * tx / ch.mass
        st.v += tangent * ty / ch.mass
        st.r += tangent * arm_t / ch.inertia_z
        return normal

    def drive_current(self) -> float:
        return self.log.drive_current()

    def reported_contacts(self) -> List[Contact]:
        return self.log.reported()

    def telemetry(self) -> dict:
        return self.log.telemetry()


def open_floor(sim) -> Arena:
    return Arena(sim, [], None, 0.0, lambda: drivetrain_parts(sim.chassis))

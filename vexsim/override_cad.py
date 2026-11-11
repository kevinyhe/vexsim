"""The Pins, Cups and Goals as VEX's field CAD draws them.

`tools/override_cad.py` cuts VEX's STEP model of the Override field and
writes the cuts to `cad/override/shapes.json`.  This module turns those cuts
into the convex parts `pieces` collides: MuJoCo, like every fast contact
solver, only collides convex shapes, so a hollow Cup or the Goal's funnel is
built from slices that fit together.

A slice covers a range of angles around the axis and a range along it.  Its
inside face is flat, set where it just touches the narrowest inner radius
the CAD has anywhere in that slice, so the cavity the slices leave is never
narrower than the real one and is exactly as narrow at the middle of each
face.  Each range of angles is split along the axis wherever a flat face
would cut more than `INNER_TOLERANCE` into the cavity, or a convex part
bulge more than `OUTER_TOLERANCE` out past the surface.

Piece frame: metres, the piece's axis is +z, its middle at the origin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .units import INCH

CAD_DIR = Path(__file__).resolve().parent.parent / "cad" / "override"
MAKE_IT = ('Override needs the shapes from VEX\'s field CAD, made once on this computer: download the Override '
           'field CAD (276-9250-000) from VEX and run\n    .venv/bin/python tools/override_cad.py '
           '"<path>/276-9250-000 (2026-04-26).STEP"\n(README, "Override field CAD").')

INNER_TOLERANCE = 0.004
OUTER_TOLERANCE = 0.008
CUP_SLICES = 18
GOAL_SLICES = 16
#: The Cup's three hollow ribs sit at 10-50, 130-170 and 250-290 degrees in
#: its half's frame; starting the slices at 10 degrees puts two whole slices
#: in each rib, so the channel inside it stays open.
CUP_FIRST_SLICE_DEG = 10.0
GOAL_FIRST_SLICE_DEG = 11.25

#: Half frame (x, y along the axis, z) -> piece frame (u, v, axis).
_PIN_AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#: A Cup half's y runs from its rim in to the middle, so its axis is -y.
_CUP_AXES = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class MissingCad(RuntimeError):
    pass


@lru_cache(maxsize=None)
def shapes() -> dict:
    path = CAD_DIR / "shapes.json"
    if not path.is_file():
        raise MissingCad(f"{path} is missing. {MAKE_IT}")
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def field() -> dict:
    path = CAD_DIR / "field.json"
    if not path.is_file():
        raise MissingCad(f"{path} is missing. {MAKE_IT}")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Profile:
    """Cuts across one CAD part, inches: `y` along its axis, and per cut the
    nearest (`inner`) and farthest (`outer`) material in each of `sectors`
    slices around it.  NaN where a slice has no material."""
    y: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    sectors: int

    @staticmethod
    def of(part: dict) -> "Profile":
        inner = np.array([[np.nan if v is None else v for v in row] for row in part["inner"]], dtype=float)
        outer = np.array([[np.nan if v is None else v for v in row] for row in part["outer"]], dtype=float)
        cut_material = ~np.all(np.isnan(outer), axis=1)
        return Profile(np.array(part["y"], dtype=float)[cut_material], inner[cut_material], outer[cut_material],
                       int(part["sectors"]))

    def row(self, y: float) -> int:
        return int(np.argmin(np.abs(self.y - y)))

    def angle(self, sector: int) -> float:
        return (sector + 0.5) * 2.0 * math.pi / self.sectors

    def sectors_from(self, start: float, span: float) -> List[int]:
        return [k for k in range(self.sectors) if ((self.angle(k) - start) % (2.0 * math.pi)) < span - 1e-9]


def _hull_gap(ys: np.ndarray, values: np.ndarray) -> float:
    """How far any point lies above the lower convex hull of (ys, values)."""
    hull: List[int] = []
    for i in range(len(ys)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (ys[b] - ys[a]) * (values[i] - values[a]) - (values[b] - values[a]) * (ys[i] - ys[a]) <= 0.0:
                hull.pop()
            else:
                break
        hull.append(i)
    return float(np.max(values - np.interp(ys, ys[hull], values[hull])))


def _hull_rows(ys: np.ndarray, values: np.ndarray) -> List[int]:
    """Rows on the lower convex hull of (ys, values), ends included."""
    keep = [i for i in range(len(ys)) if not np.isnan(values[i])]
    hull: List[int] = []
    for i in keep:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (ys[b] - ys[a]) * (values[i] - values[a]) - (values[b] - values[a]) * (ys[i] - ys[a]) <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def _bands(ys: np.ndarray, inner_curves: Sequence[np.ndarray], outer_curves: Sequence[np.ndarray],
           first: int, last: int) -> List[int]:
    """Row indices splitting first..last so no slice's flat inside face cuts
    into the cavity, and no slice bulges out, past the tolerances.  Each
    slice is as long as it can be; adding a row only ever widens a slice's
    gaps, so its end is found by doubling and then halving."""
    checks = ([(c, 1.0, INNER_TOLERANCE) for c in _distinct(inner_curves)]
              + [(c, -1.0, OUTER_TOLERANCE) for c in _distinct(outer_curves)])

    def within_tolerance(start: int, end: int) -> bool:
        for curve, sign, tolerance in checks:
            segment = curve[start:end + 1]
            keep = ~np.isnan(segment)
            if keep.sum() >= 3 and _hull_gap(ys[start:end + 1][keep], sign * segment[keep]) > tolerance:
                return False
        return True

    edges = [first]
    while edges[-1] < last:
        start = edges[-1]
        passing, step = start + 1, 1
        while passing + step <= last and within_tolerance(start, passing + step):
            passing += step
            step *= 2
        failing = min(passing + step, last + 1)
        while failing - passing > 1:
            middle = (passing + failing) // 2
            if within_tolerance(start, middle):
                passing = middle
            else:
                failing = middle
        edges.append(passing)
    return edges


def _distinct(curves: Sequence[np.ndarray]) -> List[np.ndarray]:
    return list({curve.tobytes(): curve for curve in curves}.values())


def _ring(points: List[Tuple[float, float]], tolerance: float = 0.002) -> List[Tuple[float, float]]:
    """Convex hull of 2D points, dropping corners within `tolerance` of the
    line through their neighbours."""
    ordered = sorted(set(points))

    def chain(sequence):
        out: List[Tuple[float, float]] = []
        for p in sequence:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0.0:
                out.pop()
            out.append(p)
        return out

    hull = chain(ordered)[:-1] + chain(list(reversed(ordered)))[:-1]
    dropped = True
    while dropped and len(hull) > 3:
        dropped = False
        for i in range(len(hull)):
            a, b, c = hull[i - 1], hull[i], hull[(i + 1) % len(hull)]
            ex, ez = c[0] - a[0], c[1] - a[1]
            if abs(ex * (b[1] - a[1]) - ez * (b[0] - a[0])) / math.hypot(ex, ez) < tolerance:
                hull.pop(i)
                dropped = True
                break
    return hull


def _solid_parts(profile: Profile, first: int, last: int) -> List[np.ndarray]:
    """A solid part (a Pin half's taper) as convex bands, half frame inches."""
    widest = np.nanmax(np.where(np.isnan(profile.outer), -np.inf, profile.outer), axis=1)
    curves = [profile.outer[:, k] for k in range(profile.sectors)] + [widest]
    edges = _bands(profile.y, [], curves, first, last)
    return [np.array(_ring_points(profile, rows)) for rows in
            ([a + r for r in _hull_rows(profile.y[a:b + 1], -widest[a:b + 1])] for a, b in zip(edges[:-1], edges[1:]))]


def _ring_points(profile: Profile, rows: Sequence[int]) -> List[Tuple[float, float, float]]:
    points = []
    for i in rows:
        flat = [(profile.outer[i, k] * math.cos(profile.angle(k)), profile.outer[i, k] * math.sin(profile.angle(k)))
                for k in range(profile.sectors) if not np.isnan(profile.outer[i, k])]
        points += [(x, profile.y[i], z) for x, z in _ring(flat)]
    return points


def _shell_parts(profile: Profile, slices: int, first_deg: float, first: int, last: int) -> List[np.ndarray]:
    """A hollow part (a Cup half, a Goal's taper) as slices, half frame inches."""
    span = 2.0 * math.pi / slices
    columns = [(math.radians(first_deg) + c * span, profile.sectors_from(math.radians(first_deg) + c * span, span))
               for c in range(slices)]
    inner_curves = []
    outer_curves = []
    for _, ks in columns:
        inner = profile.inner[:, ks]
        inner_curves.append(np.where(np.all(np.isnan(inner), axis=1), np.nan,
                                     np.nanmin(np.where(np.isnan(inner), np.inf, inner), axis=1)))
        outer = profile.outer[:, ks]
        outer_curves.append(np.where(np.all(np.isnan(outer), axis=1), np.nan,
                                     np.nanmax(np.where(np.isnan(outer), -np.inf, outer), axis=1)))
    parts = []
    for (start, ks), inner, outer in zip(columns, inner_curves, outer_curves):
        end = start + span
        edges = _bands(profile.y, [inner], [outer], first, last)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            points = []
            for i in _hull_rows(profile.y[e0:e1 + 1], inner[e0:e1 + 1]):
                reach = inner[e0 + i] / math.cos(span / 2.0)
                points += [(reach * math.cos(a), profile.y[e0 + i], reach * math.sin(a)) for a in (start, end)]
            for i in _hull_rows(profile.y[e0:e1 + 1], -outer[e0:e1 + 1]):
                top = outer[e0 + i]
                points += [(top * math.cos(a), profile.y[e0 + i], top * math.sin(a))
                           for a in (start, (start + end) / 2.0, end)]
            if len({round(p[1], 6) for p in points}) >= 2:
                parts.append(np.array(points))
    return parts


def _place(parts: Sequence[np.ndarray], axes: np.ndarray, centre_y: float) -> List[np.ndarray]:
    return [((p - np.array([0.0, centre_y, 0.0])) @ axes.T) * INCH for p in parts]


def _other_half(parts: Sequence[np.ndarray], relation: dict) -> List[np.ndarray]:
    rotation = np.array(relation["rotation"])
    offset = np.array(relation["offset_in"])
    return [p @ rotation.T + offset for p in parts]


@dataclass(frozen=True)
class PieceShape:
    """`hulls`: convex parts, piece frame (metres).  `volume` (m^3) and
    `gyration` (axial, across, m^2: inertia over mass about the middle)
    come from the CAD solid.  `radius_at` is the widest the piece is at a
    point along its axis; `squeezed_width_at` the narrowest it is across
    there, which jaws squeezing it turn it to."""
    kind: str
    hulls: Tuple[np.ndarray, ...]
    half_length: float
    volume: float
    gyration: Tuple[float, float]
    profile_s: np.ndarray
    profile_r: np.ndarray
    profile_squeezed: np.ndarray

    def radius_at(self, s: float) -> float:
        if abs(s) > self.half_length:
            return 0.0
        return float(np.interp(abs(s), self.profile_s, self.profile_r))

    def squeezed_width_at(self, s: float) -> float:
        if abs(s) > self.half_length:
            return 0.0
        return float(np.interp(abs(s), self.profile_s, self.profile_squeezed))


def _gyration(part: dict, centre_y: float) -> Tuple[float, float]:
    """(axial, across) inertia over mass of the whole piece, m^2, from one
    half's: the two halves are the same part turned end for end."""
    inertia = np.array(part["inertia_per_volume_in2"])
    along = part["centroid_in"][1] - centre_y
    return inertia[1][1] * INCH ** 2, (inertia[0][0] + along ** 2) * INCH ** 2


def _outer_radius(profile: Profile, centre_y: float, toward_end: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance from the middle, widest radius, narrowest width across), metres."""
    widest = np.nanmax(np.where(np.isnan(profile.outer), -np.inf, profile.outer), axis=1)
    s = (profile.y - centre_y) * toward_end
    keep = (s >= 0.0) & np.isfinite(widest)
    order = np.argsort(s[keep])
    return s[keep][order] * INCH, widest[keep][order] * INCH, _narrowest_across(profile)[keep][order] * INCH


def _narrowest_across(profile: Profile) -> np.ndarray:
    """Per row, the narrowest width of the cross-section's outline over the
    sector directions: the support distance one way plus the other way."""
    angles = np.array([profile.angle(k) for k in range(profile.sectors)])
    reach = np.where(np.isnan(profile.outer), -np.inf, profile.outer)
    support = np.max(reach[:, None, :] * np.cos(angles[None, :, None] - angles[None, None, :]), axis=2)
    half = profile.sectors // 2
    return np.min(support[:, :half] + support[:, half:], axis=1)


@lru_cache(maxsize=None)
def pin() -> PieceShape:
    """A Pin: the flange where the halves meet is one convex part, and each
    half's taper is split into bands."""
    part = shapes()["parts"]["pin_half"]
    profile = Profile.of(part)
    middle = profile.row(0.0)
    widest = np.nanmax(np.where(np.isnan(profile.outer), -np.inf, profile.outer), axis=1)
    drop = np.diff(widest[middle:])
    flange_end = middle + int(np.argmin(drop)) + 1
    flange_rows = [middle + r for r in _hull_rows(profile.y[middle:flange_end], -widest[middle:flange_end])]
    flange = np.array(_ring_points(profile, flange_rows))
    taper = _solid_parts(profile, flange_end, len(profile.y) - 1)
    relation = part["other_half"]
    own = [np.vstack([flange] + _other_half([flange], relation))] + taper
    hulls = _place(own + _other_half(taper, relation), _PIN_AXES, 0.0)
    s, r, squeezed = _outer_radius(profile, 0.0, 1.0)
    return PieceShape("pin", tuple(hulls), float(profile.y[-1] * INCH), 2.0 * part["volume_in3"] * INCH ** 3,
                      _gyration(part, 0.0), s, r, squeezed)


def cup_middle_y(part: dict) -> float:
    """Where a Cup's middle is along its first half's axis: halfway between
    the two rims."""
    rim = part["low_in"][1]
    relation = part["other_half"]
    other_rim = np.array(relation["rotation"]) @ np.array([0.0, rim, 0.0]) + np.array(relation["offset_in"])
    return float((rim + other_rim[1]) / 2.0)


@lru_cache(maxsize=None)
def cup() -> PieceShape:
    part = shapes()["parts"]["cup_half"]
    profile = Profile.of(part)
    middle = cup_middle_y(part)
    own = _shell_parts(profile, CUP_SLICES, CUP_FIRST_SLICE_DEG, 0, profile.row(middle))
    hulls = _place(own, _CUP_AXES, middle) + _place(_other_half(own, part["other_half"]), _CUP_AXES, middle)
    s, r, squeezed = _outer_radius(profile, middle, -1.0)
    return PieceShape("cup", tuple(hulls), float((middle - profile.y[0]) * INCH), 2.0 * part["volume_in3"] * INCH ** 3,
                      _gyration(part, middle), s, r, squeezed)


def half_frames(kind: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Where a Pin's or a Cup's two halves are: for the half at the low end
    of its axis and then the one at the high end, the rotation and offset
    (inches) that take a point of the half's CAD part into the piece frame."""
    if kind == "pin":
        part, axes, middle = shapes()["parts"]["pin_half"], _PIN_AXES, 0.0
    else:
        part = shapes()["parts"]["cup_half"]
        axes, middle = _CUP_AXES, cup_middle_y(part)
    rotation = np.array(part["other_half"]["rotation"])
    offset = np.array(part["other_half"]["offset_in"])
    to_middle = np.array([0.0, middle, 0.0])
    return (axes @ rotation, axes @ (offset - to_middle)), (axes, -(axes @ to_middle))


#: Goal taper frame (x, y up, z) -> field (x, y, z): y up becomes z up.
_GOAL_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@lru_cache(maxsize=None)
def goal_taper() -> Tuple[np.ndarray, ...]:
    """The Goal's tapered part as convex slices: metres, x and y as on the
    field, z up from the bottom of the part."""
    part = shapes()["parts"]["goal_taper"]
    profile = Profile.of(part)
    own = _shell_parts(profile, GOAL_SLICES, GOAL_FIRST_SLICE_DEG, 0, len(profile.y) - 1)
    return tuple(_place(own, _GOAL_AXES, profile.y[0]))


def riser_size(kind: str) -> Tuple[float, float, float]:
    """(across x, across y, height) of a Goal's riser, metres."""
    part = shapes()["parts"][f"{kind}_riser"]
    low, high = np.array(part["low_in"]), np.array(part["high_in"])
    size = (high - low) * INCH
    return float(size[0]), float(size[2]), float(size[1])


def _widest(profile: Profile) -> np.ndarray:
    return np.nanmax(np.where(np.isnan(profile.outer), -np.inf, profile.outer), axis=1)


def _narrowest(profile: Profile) -> np.ndarray:
    return np.nanmin(np.where(np.isnan(profile.inner), np.inf, profile.inner), axis=1)


@dataclass(frozen=True)
class Column:
    """A piece standing upright, as radii against height (inches) from its
    middle: `outer` the widest it is, `cavity` the narrowest its hollow is
    (0 where it is solid), between `bottom` and `top`."""
    z: np.ndarray
    outer: np.ndarray
    cavity: np.ndarray
    bottom: float
    top: float


def _column(z: np.ndarray, outer: np.ndarray, cavity: np.ndarray) -> Column:
    order = np.argsort(z)
    return Column(z[order], outer[order], cavity[order], float(z.min()), float(z.max()))


@lru_cache(maxsize=None)
def pin_column() -> Column:
    profile = Profile.of(shapes()["parts"]["pin_half"])
    keep = profile.y >= 0.0
    s = profile.y[keep]
    outer = _widest(profile)[keep]
    return _column(np.concatenate([-s[::-1], s]), np.concatenate([outer[::-1], outer]), np.zeros(2 * len(s)))


@lru_cache(maxsize=None)
def cup_column() -> Column:
    part = shapes()["parts"]["cup_half"]
    profile = Profile.of(part)
    middle = cup_middle_y(part)
    keep = profile.y <= middle
    s = middle - profile.y[keep]
    outer, cavity = _widest(profile)[keep], _narrowest(profile)[keep]
    return _column(np.concatenate([-s, s]), np.concatenate([outer, outer]), np.concatenate([cavity, cavity]))


@lru_cache(maxsize=None)
def goal_column() -> Column:
    """The Goal's tapered part from its bottom up; what is under it is solid."""
    profile = Profile.of(shapes()["parts"]["goal_taper"])
    z = profile.y - profile.y[0]
    return _column(z, _widest(profile), np.where(np.isfinite(_narrowest(profile)), _narrowest(profile), 0.0))


def _lower_until_touching(overlap, high: float, low: float) -> float:
    """The highest offset between `high` and `low` at which `overlap(h)` is
    true, found in 0.01 in steps and then in 0.0002 in steps."""
    for step in (0.01, 0.0002):
        h = high
        while h > low and not overlap(h):
            h -= step
        if h <= low:
            raise ValueError("the two pieces pass through each other")
        high, low = h + step, h - step
    return h + 0.0002


def _pin_into(around: Column, solid_below: bool) -> float:
    """Height of a Pin's middle over `around`'s middle once lowered into it
    until they touch, both upright and on one axis, inches."""
    pin = pin_column()
    grid = np.arange(around.bottom - pin.top + pin.bottom, around.top + 0.001, 0.001)
    cavity = np.interp(grid, around.z, around.cavity, left=0.0 if solid_below else np.inf, right=np.inf)
    return _lower_until_touching(
        lambda h: bool(np.any(np.interp(grid - h, pin.z, pin.outer, left=0.0, right=0.0) > cavity + 1e-9)),
        around.top - pin.bottom, around.bottom - pin.top)


def _cup_over_pin() -> float:
    """Height of a Cup's middle over a Pin's middle once lowered over it
    until they touch, inches."""
    pin, cup = pin_column(), cup_column()
    grid = np.arange(pin.bottom, pin.top + 0.001, 0.001)
    reach = np.interp(grid, pin.z, pin.outer)
    return _lower_until_touching(
        lambda h: bool(np.any(reach > np.interp(grid - h, cup.z, cup.cavity, left=np.inf, right=np.inf) + 1e-9)),
        pin.top - cup.bottom, pin.bottom - cup.top)


@dataclass(frozen=True)
class Seats:
    """Where pieces nest, upright and lined up, metres: a Pin's middle over
    the middle of the Cup it stands in, a Cup's middle over the middle of
    the Pin it is set over, and a Pin's middle over the bottom of a Goal's
    tapered part."""
    pin_in_cup: float
    cup_on_pin: float
    pin_in_goal: float


@lru_cache(maxsize=None)
def seats() -> Seats:
    return Seats(_pin_into(cup_column(), solid_below=False) * INCH, _cup_over_pin() * INCH,
                 _pin_into(goal_column(), solid_below=True) * INCH)


# ----------------------------------------------------------------- field elements

#: VEX's part numbers in the field CAD
NEUTRAL_RISER_PART = "276-9250-004"
CENTER_RISER_PART = "276-9250-005"
TOGGLE_BRACKET_PARTS = ("276-9250-021", "276-9250-022")
TOGGLE_BAR_PART = "276-9250-023"
TOGGLE_END_CAP_PART = "276-9250-024"
TOGGLE_WASHER_PART = "276-9250-027"
LOADER_CHUTE_PART = "276-9250-031"
LOADER_BACK_PART = "276-9250-032"
LOADER_HANDLE_PART = "276-9250-034"


@lru_cache(maxsize=None)
def view() -> dict:
    """Every field part's mesh (`view.bin`) and where the CAD places it."""
    layout_path, mesh_path = CAD_DIR / "view.json", CAD_DIR / "view.bin"
    for path in (layout_path, mesh_path):
        if not path.is_file():
            raise MissingCad(f"{path} is missing. {MAKE_IT}")
    layout = json.loads(layout_path.read_text())
    layout["blob"] = mesh_path.read_bytes()
    return layout


def part_mesh(number: str) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices, triangles) of one part, inches, in its own frame turned to the field's axes."""
    layout = view()
    mesh = layout["meshes"][number]
    points = np.frombuffer(layout["blob"], np.float32, mesh["vertices"] * 3, mesh["vertex_offset"])
    triangles = np.frombuffer(layout["blob"], np.uint32, mesh["triangles"] * 3, mesh["index_offset"])
    return points.reshape(-1, 3).astype(float) @ np.array(layout["cad_axes"]).T, triangles.reshape(-1, 3)


def placed_points(number: str, instance: dict) -> np.ndarray:
    """A part's vertices where the CAD places this copy of it, field inches."""
    points, _ = part_mesh(number)
    return points @ np.array(instance["rotation"]).T + np.array(instance["position_in"])


#: Flat sheets have vertices only at their edges' ends; cutting a part every
#: this far along each axis puts points along every edge between them
DENSE_STEP_IN = 0.05


def dense_points(number: str, instance: dict) -> np.ndarray:
    """A placed part's vertices and where planes DENSE_STEP_IN apart along x, y and z cut its triangles' edges."""
    points, triangles = part_mesh(number)
    points = points @ np.array(instance["rotation"]).T + np.array(instance["position_in"])
    out = [points]
    starts, ends = points[triangles], points[np.roll(triangles, -1, axis=1)]
    starts, ends = starts.reshape(-1, 3), ends.reshape(-1, 3)
    for axis in range(3):
        low, high = np.minimum(starts[:, axis], ends[:, axis]), np.maximum(starts[:, axis], ends[:, axis])
        first = np.ceil(low / DENSE_STEP_IN).astype(int)
        last = np.floor(high / DENSE_STEP_IN).astype(int)
        for k in range(1, int((last - first).max(initial=0)) + 2):
            plane = (first + k - 1)
            cut = (plane <= last) & (high > low)
            if not cut.any():
                break
            t = (plane[cut] * DENSE_STEP_IN - starts[cut, axis]) / (ends[cut, axis] - starts[cut, axis])
            out.append(starts[cut] + t[:, None] * (ends[cut] - starts[cut]))
    return np.vstack(out)


def nearest_instance(number: str, x_in: float, y_in: float) -> dict:
    return min(view()["instances"][number],
               key=lambda instance: math.hypot(instance["position_in"][0] - x_in, instance["position_in"][1] - y_in))


def solid_properties(number: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """Volume (in^3), centroid and inertia over volume about the centroid
    (in^2, 3x3) of a closed part, in its frame turned to the field's axes."""
    points, triangles = part_mesh(number)
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = volumes.sum()
    centroid = (volumes[:, None] * (a + b + c)).sum(axis=0) / (4.0 * volume)
    second = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            second[i, j] = (volumes * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j]
                                       + (a[:, i] + b[:, i] + c[:, i]) * (a[:, j] + b[:, j] + c[:, j]))).sum() / 20.0
    covariance = second / volume - np.outer(centroid, centroid)
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return float(volume), centroid, inertia


def riser_hull(kind: str) -> np.ndarray:
    """A Goal's riser, solid (its convex outside), metres, from the Goal's middle on the floor."""
    number = CENTER_RISER_PART if kind == "center" else NEUTRAL_RISER_PART
    instance = view()["instances"][number][0]
    points = placed_points(number, instance) - np.array(instance["position_in"])
    return points * INCH


@dataclass(frozen=True)
class ToggleShape:
    """A Toggle and its two brackets in the mount frame: inches from the
    axle at rest, n into the field, a along the wall, z up.  The bar turns
    about a hub that rides up a slot in each bracket's ear.  Each entry of
    `hubs` and `brackets` is the points of one convex part; `bar` is the
    bar with its end caps, a prism of its outline across."""
    bar: np.ndarray
    hubs: Tuple[np.ndarray, ...]
    brackets: Tuple[np.ndarray, ...]
    slot_direction: Tuple[float, float]
    slot_travel: float
    axle_from_center: float
    axle_height: float
    volume: float
    centroid: np.ndarray
    inertia_per_volume: np.ndarray


def _mount_frame(points: np.ndarray, axle: np.ndarray, inward: Tuple[float, float]) -> np.ndarray:
    along = (-inward[1], inward[0])
    relative = points - axle
    return np.column_stack((relative[:, 0] * inward[0] + relative[:, 1] * inward[1],
                            relative[:, 0] * along[0] + relative[:, 1] * along[1], relative[:, 2]))


#: The ear at a bracket's outer end is this far out along the axle, bent up
#: from the sheet's end; the bar's end caps stop at the bend and only their
#: hubs go on through the ear's slot
TOGGLE_EAR_FROM_MIDDLE_IN = 13.02
TOGGLE_BEND_FROM_MIDDLE_IN = 13.0


@lru_cache(maxsize=None)
def toggle_shape() -> ToggleShape:
    """The red alliance's left Toggle, as the CAD has it; the four are the same assembly."""
    inward = (1.0, 0.0)
    bar_instance = min(view()["instances"][TOGGLE_BAR_PART], key=lambda instance: instance["position_in"][0])
    axle = np.array(bar_instance["position_in"])
    near = lambda number: [instance for instance in view()["instances"][number]
                           if np.linalg.norm(np.array(instance["position_in"][:2]) - axle[:2]) < 20.0]
    frame = lambda number, instance: _mount_frame(dense_points(number, instance), axle, inward)
    bar = [frame(TOGGLE_BAR_PART, bar_instance)]
    hubs = []
    for instance in near(TOGGLE_END_CAP_PART):
        cap = frame(TOGGLE_END_CAP_PART, instance)
        bar.append(cap[np.abs(cap[:, 1]) < TOGGLE_BEND_FROM_MIDDLE_IN])
        hubs.append(cap[np.abs(cap[:, 1]) >= TOGGLE_BEND_FROM_MIDDLE_IN])
    hubs += [frame(TOGGLE_WASHER_PART, instance) for instance in near(TOGGLE_WASHER_PART)]
    slot_top = _slot_top(near(TOGGLE_BRACKET_PARTS[0])[0], axle, inward)
    brackets = []
    for number in TOGGLE_BRACKET_PARTS:
        for instance in near(number):
            brackets += _bracket_parts(frame(number, instance), slot_top)
    volume, centroid, inertia = _toggle_solid()
    travel = float(np.hypot(*slot_top))
    bar = np.vstack(bar)
    outline = _outline_2d(bar[:, [0, 2]])
    ends = (bar[:, 1].min(), bar[:, 1].max())
    prism = np.vstack([np.column_stack((outline[:, 0], np.full(len(outline), end), outline[:, 1])) for end in ends])
    return ToggleShape(prism, tuple(hubs), tuple(brackets),
                       (float(slot_top[0] / travel), float(slot_top[1] / travel)), travel,
                       float(-axle[0]), float(axle[2]), volume, centroid, inertia)


#: The bracket's sheet is 0.1 in thick; its legs are what lies within this of its back and field-side edges
TOGGLE_BRACKET_LEG_IN = 0.12


def _bracket_parts(points: np.ndarray, slot_top: np.ndarray) -> List[np.ndarray]:
    """A bracket bent over the wall's top as its convex parts: the leg down
    the wall's back, the plate sloping up toward the field that the bar
    rests a face on, the leg down the field side, and the ear at its outer
    end on each side of the slot and above it."""
    sheet = points[np.abs(points[:, 1]) < TOGGLE_BEND_FROM_MIDDLE_IN]
    ear = points[np.abs(points[:, 1]) >= TOGGLE_EAR_FROM_MIDDLE_IN]
    back_leg_edge = sheet[:, 0].min() + TOGGLE_BRACKET_LEG_IN
    field_leg_edge = sheet[:, 0].max() - TOGGLE_BRACKET_LEG_IN
    parts = [sheet[sheet[:, 0] <= back_leg_edge], sheet[(sheet[:, 0] > back_leg_edge) & (sheet[:, 0] < field_leg_edge)],
             sheet[sheet[:, 0] >= field_leg_edge]]
    radius = np.hypot(ear[:, 0], ear[:, 2]).min()
    slot_back, slot_front = min(0.0, slot_top[0]) - radius, max(0.0, slot_top[0]) + radius
    beside = (ear[:, 0] > slot_back) & (ear[:, 0] < slot_front)
    parts += [ear[ear[:, 0] <= slot_back], ear[ear[:, 0] >= slot_front], ear[beside & (ear[:, 2] >= slot_top[1] + radius)]]
    return parts


def _outline_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points."""
    order = sorted(set(map(tuple, np.round(points, 6))))

    def half(sequence):
        out: List[tuple] = []
        for point in sequence:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (point[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (point[0] - out[-2][0])) <= 0.0:
                out.pop()
            out.append(point)
        return out

    return np.array(half(order)[:-1] + half(reversed(order))[:-1])


def _inside_outline(points: np.ndarray, outline: np.ndarray, margin: float) -> np.ndarray:
    """Which 2D points lie more than `margin` inside a counter-clockwise convex outline."""
    inside = np.ones(len(points), dtype=bool)
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        inside &= (points - a) @ normal > margin
    return inside


def _slot_top(instance: dict, axle: np.ndarray, inward: Tuple[float, float]) -> np.ndarray:
    """(n, z) of the middle of the slot's top end, from the axle at rest, inches.
    The slot is the ear's edge inside the ear's outline; its bottom end is
    round about the axle and its top end is fitted as a circle."""
    points = _mount_frame(placed_points(TOGGLE_BRACKET_PARTS[0], instance), axle, inward)
    ear = points[np.abs(points[:, 1]) >= TOGGLE_EAR_FROM_MIDDLE_IN][:, [0, 2]]
    slot = ear[_inside_outline(ear, _outline_2d(ear), 0.02)]
    radius = np.hypot(slot[:, 0], slot[:, 1]).min()
    top = slot[slot[:, 1] > radius]
    fit = np.linalg.lstsq(np.column_stack((2.0 * top[:, 0], 2.0 * top[:, 1], np.ones(len(top)))),
                          top[:, 0] ** 2 + top[:, 1] ** 2, rcond=None)[0]
    return np.array(fit[:2])


def _toggle_solid() -> Tuple[float, np.ndarray, np.ndarray]:
    """The turning parts together: the bar, its two end caps and two washers,
    at one density.  (volume in^3, centroid and inertia over volume about it
    in^2) in the mount frame."""
    pieces = []
    bar_instance = min(view()["instances"][TOGGLE_BAR_PART], key=lambda instance: instance["position_in"][0])
    axle = np.array(bar_instance["position_in"])
    for number in (TOGGLE_BAR_PART, TOGGLE_END_CAP_PART, TOGGLE_WASHER_PART):
        volume, centroid, inertia = solid_properties(number)
        for instance in view()["instances"][number]:
            if np.linalg.norm(np.array(instance["position_in"][:2]) - axle[:2]) > 20.0:
                continue
            rotation = np.array(instance["rotation"])
            middle = _mount_frame((rotation @ centroid + np.array(instance["position_in"]))[None, :], axle, (1.0, 0.0))[0]
            turn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ rotation
            pieces.append((volume, middle, turn @ inertia @ turn.T))
    total = sum(volume for volume, _, _ in pieces)
    centroid = sum(volume * middle for volume, middle, _ in pieces) / total
    inertia = np.zeros((3, 3))
    for volume, middle, own in pieces:
        offset = middle - centroid
        inertia += volume * (own + np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return float(total), centroid, inertia / total


@dataclass(frozen=True)
class LoaderShape:
    """A lowered Loader as convex parts, inches from the middle of its foot
    on the wall's inner face: n into the field, a along the wall, z up.
    `cad_width` is how wide the CAD has it along the wall."""
    parts: Tuple[np.ndarray, ...]
    cad_width: float


#: A Loader's chute is open under its front; its sides reach the floor only near the wall
LOADER_SIDE_SPLIT_IN = 2.8
LOADER_FRONT_IN = 3.4
LOADER_BACK_PLATE_IN = 0.15
#: Where the chute's front turns into its sides, out from its middle
LOADER_CORNER_IN = 1.35


@lru_cache(maxsize=None)
def loader_shape() -> LoaderShape:
    """The blue alliance's Loader nearer the audience, as the CAD has it:
    the chute (its front, and each side below and above the opening and
    over the wall), the channel holding it to the wall, and the handle."""
    instance = nearest_instance(LOADER_CHUTE_PART, 70.0, -60.0)
    foot = np.array([instance["position_in"][0], instance["position_in"][1], 0.0])
    inward = (-1.0, 0.0)
    wall_face = 70.205
    origin = np.array([wall_face, foot[1], 0.0])
    frame = lambda number: _mount_frame(dense_points(number, nearest_instance(number, 70.0, -60.0)), origin, inward)
    chute, channel, handle = frame(LOADER_CHUTE_PART), frame(LOADER_BACK_PART), frame(LOADER_HANDLE_PART)
    parts = [chute[chute[:, 0] >= LOADER_FRONT_IN]]
    for side in (1.0, -1.0):
        own = chute[(side * chute[:, 1] > LOADER_CORNER_IN) | ((side * chute[:, 1] > 0.0) & (chute[:, 0] < LOADER_FRONT_IN))]
        in_field = own[own[:, 0] >= 0.0]
        parts += [in_field[in_field[:, 2] < LOADER_SIDE_SPLIT_IN], in_field[in_field[:, 2] >= LOADER_SIDE_SPLIT_IN],
                  own[own[:, 0] < 0.0]]
        flange = channel[(side * channel[:, 1] > 0.0) & (channel[:, 0] >= LOADER_BACK_PLATE_IN)]
        parts.append(flange)
    parts.append(channel[channel[:, 0] < LOADER_BACK_PLATE_IN])
    parts.append(handle)
    return LoaderShape(tuple(parts), float(chute[:, 1].max() - chute[:, 1].min()))

"""Read VEX's Override field CAD and write what vexsim takes from it.

    .venv/bin/python tools/override_cad.py "276-9250-000 (2026-04-26).STEP"

The STEP file is VEX's, from the Override field CAD download.  What this
writes is made from it, so it stays out of git: everything goes in
`cad/override/` at the top of this checkout.

- `shapes.json`: the Pin half, the Cup half and the Goal's tapered part cut
  across their axes every `STEP_IN` inches.  For each cut, in each of
  `SECTORS` slices around the axis, the nearest and farthest material from
  the axis.  `vexsim.override_cad` builds the convex parts the Pins, Cups and
  Goals collide with from these.  Also each part's volume, centroid and
  inertia, how the two halves of a Pin and of a Cup sit against each other,
  and the risers under the taller Goals.
- `field.json`: where the CAD puts the tiles, walls, Goals, Loaders, Toggles,
  Pins and Cups, in vexsim's field frame (inches, x toward the blue Alliance
  Station, y away from the audience, z up from the tiles).
- `view.json` and `view.bin`: triangles of the parts the 3D view draws.
- `profiles-<file id>.json`: the cuts, kept so running this again on the
  same STEP file skips the 20 minutes of cutting.

The CAD is in millimetres with Y up; vexsim x is CAD X, y is -Z, and z is Y
less the height of the tile tops.

Needs `cadquery-ocp` and `numpy`.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GCPnts import GCPnts_UniformAbscissa
from OCP.GProp import GProp_GProps
from OCP.OCP.TopoDS import TopoDS
from OCP.OCP.collections import Sequence_TDF_Label
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

MM = 25.4
OUT = Path(__file__).resolve().parent.parent / "cad" / "override"
STEP_IN = 0.005
SECTORS = 72
SAMPLE_IN = 0.005
VIEW_DEFLECTION_IN = 0.01
#: The two halves of a Cup overlap 0.30 in where they join
JOIN_REACH_IN = 0.5

PIN_HALF = "276-9250-001"
CUP_HALF = "276-9250-002"
GOAL_TAPER = "276-9250-003"
NEUTRAL_RISER = "276-9250-004"
CENTER_RISER = "276-9250-005"
TOGGLE_BAR = "276-9250-023"
LOADER_TUBE = "276-9250-031"
TILE = "276-6904-001"
PERIMETER_RAIL = "276-7596-35"

#: Parts the 3D view draws, by part number.  Several CAD parts share a
#: number and differ only in colour; the view colours them itself.
VIEW_PARTS = (PIN_HALF, CUP_HALF, GOAL_TAPER, NEUTRAL_RISER, CENTER_RISER, "276-9250-021", "276-9250-022",
              TOGGLE_BAR, "276-9250-024", "276-9250-027", LOADER_TUBE, "276-9250-032", "276-9250-034")


def read_step(path: Path):
    doc = TDocStd_Document(TCollection_ExtendedString("field"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != 1:
        raise SystemExit(f"{path} did not read as STEP")
    if not reader.Transfer(doc):
        raise SystemExit(f"{path} read but did not transfer")
    return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()), doc


def label_name(label) -> str:
    attribute = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return attribute.Get().ToExtString()
    return ""


def placement(location: TopLoc_Location) -> np.ndarray:
    t = location.Transformation()
    return np.array([[t.Value(r, c) for c in range(1, 5)] for r in range(1, 4)])


def leaf_instances(tool):
    """[(part name, assembly it is in, prototype id, 3x4 placement in mm)]
    and {prototype id: shape}."""
    shapes, out = {}, []

    def walk(label, location, parent):
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            components = Sequence_TDF_Label()
            XCAFDoc_ShapeTool.GetComponents_s(label, components, False)
            for i in range(1, components.Length() + 1):
                component = components.Value(i)
                referred = TDF_Label()
                XCAFDoc_ShapeTool.GetReferredShape_s(component, referred)
                walk(referred, location.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(component)), label_name(label))
            return
        key = f"{label_name(label)}|{label.Tag()}"
        if key not in shapes:
            shapes[key] = (label_name(label), XCAFDoc_ShapeTool.GetShape_s(label))
        out.append((label_name(label), parent, key, placement(location)))

    free = Sequence_TDF_Label()
    tool.GetFreeShapes(free)
    for i in range(1, free.Length() + 1):
        walk(free.Value(i), TopLoc_Location(), "")
    return out, shapes


def prototype(shapes, number: str):
    """The first CAD part with this part number: (name, shape)."""
    for name, shape in shapes.values():
        if name.startswith(number):
            return name, shape
    raise SystemExit(f"the CAD has no part {number}")


def bounds(shape):
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    low, high = box.CornerMin(), box.CornerMax()
    return np.array([low.X(), low.Y(), low.Z()]) / MM, np.array([high.X(), high.Y(), high.Z()]) / MM


def mass_properties(shape) -> dict:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = props.Mass() / MM ** 3
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()
    inertia = [[m.Value(r, k) / MM ** 5 / volume for k in range(1, 4)] for r in range(1, 4)]
    return dict(volume_in3=volume, centroid_in=[c.X() / MM, c.Y() / MM, c.Z() / MM],
                inertia_per_volume_in2=inertia)


def write_brep(shape, path: Path) -> None:
    BRepTools.Write_s(shape, str(path))


def read_brep(path: str):
    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, path, BRep_Builder())
    return shape


def radial_profile(job):
    """Cut a part whose axis is its local Y every STEP_IN between y_from and
    y_to; in each of SECTORS slices around the axis keep the nearest and
    farthest cut point.  Straight edges are sampled every SAMPLE_IN so a flat
    face's middle is not missed."""
    brep, y_from, y_to = job
    shape = read_brep(brep)
    count = int(round((y_to - y_from) / STEP_IN)) + 1
    ys, inner, outer = [], [], []
    for i in range(count):
        y = y_from + i * STEP_IN
        cut = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(0.0, y * MM, 0.0), gp_Dir(0.0, 1.0, 0.0)))
        cut.Build()
        near = [math.inf] * SECTORS
        far = [0.0] * SECTORS
        explorer = TopExp_Explorer(cut.Shape(), TopAbs_EDGE)
        while explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge(explorer.Current()))
            sampler = GCPnts_UniformAbscissa(curve, SAMPLE_IN * MM)
            for k in range(1, sampler.NbPoints() + 1):
                p = curve.Value(sampler.Parameter(k))
                x, z = p.X() / MM, p.Z() / MM
                r = math.hypot(x, z)
                sector = int((math.atan2(z, x) % (2.0 * math.pi)) / (2.0 * math.pi) * SECTORS) % SECTORS
                near[sector] = min(near[sector], r)
                far[sector] = max(far[sector], r)
            explorer.Next()
        ys.append(round(y, 4))
        inner.append([None if v == math.inf else round(v, 4) for v in near])
        outer.append([None if v == 0.0 else round(v, 4) for v in far])
    return dict(y=ys, inner=inner, outer=outer)


def triangles(shape):
    """Vertices (inches, the part's own frame) and outward triangles."""
    BRepMesh_IncrementalMesh(shape, VIEW_DEFLECTION_IN * MM, False, 0.3, True)
    vertices, faces = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            transform = location.Transformation()
            base = len(vertices)
            for i in range(1, mesh.NbNodes() + 1):
                p = mesh.Node(i).Transformed(transform)
                vertices.append((p.X() / MM, p.Y() / MM, p.Z() / MM))
            flipped = face.Orientation() == TopAbs_REVERSED
            for i in range(1, mesh.NbTriangles() + 1):
                a, b, c = mesh.Triangle(i).Get()
                faces.append((base + a - 1, base + c - 1, base + b - 1) if flipped else (base + a - 1, base + b - 1, base + c - 1))
        explorer.Next()
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


class Frame:
    """CAD millimetres, Y up -> vexsim inches, z up from the tile tops."""

    def __init__(self, tile_top_in: float):
        self.tile_top = tile_top_in
        self.axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    def point(self, cad_mm) -> list:
        p = self.axes @ (np.asarray(cad_mm) / MM)
        return [float(p[0]), float(p[1]), float(p[2] - self.tile_top)]

    def direction(self, cad) -> list:
        return [float(v) for v in self.axes @ np.asarray(cad)]

    def rotation(self, cad_rotation) -> list:
        return (self.axes @ np.asarray(cad_rotation) @ self.axes.T).tolist()


def tile_top(instances, shapes) -> float:
    highest = -math.inf
    for name, _, key, place in instances:
        if name.startswith(TILE):
            low, high = bounds(shapes[key][1])
            corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            world_y = (place[:, :3] @ corners.T)[1] + place[1, 3] / MM
            highest = max(highest, float(world_y.max()))
    return highest


def pose(frame: Frame, place) -> dict:
    return dict(position_in=frame.point(place[:, 3]), rotation=frame.rotation(place[:, :3]))


def paired_halves(instances, number: str, assembly_prefix: str, join_y_in: float):
    """Halves of one part number that belong to the same Pin or Cup: in the
    same assembly, pointing opposite ways, with the ends they join at
    (`join_y_in` along their own y) within JOIN_REACH_IN of each other."""
    halves = [(name, parent, place) for name, parent, _, place in instances
              if name.startswith(number) and parent.startswith(assembly_prefix)]

    def join(place):
        return place[:, 3] / MM + place[:, :3] @ np.array([0.0, join_y_in, 0.0])

    pairs, used = [], set()
    for i, (name, parent, place) in enumerate(halves):
        if i in used:
            continue
        mates = [j for j, (_, other_parent, other) in enumerate(halves)
                 if j != i and j not in used and other_parent == parent
                 and float(place[:, 1] @ other[:, 1]) < -0.9
                 and np.linalg.norm(join(other) - join(place)) < JOIN_REACH_IN]
        if len(mates) != 1:
            raise SystemExit(f"{name} in {parent} has {len(mates)} other halves near it")
        used.update((i, mates[0]))
        pairs.append((halves[i], halves[mates[0]]))
    return pairs


def halves_relation(first, second) -> dict:
    """The second half in the first half's frame: rotation, and where its
    origin is (inches)."""
    rotation = first[:, :3].T @ second[:, :3]
    offset = first[:, :3].T @ ((second[:, 3] - first[:, 3]) / MM)
    return dict(rotation=np.round(rotation, 6).tolist(), offset_in=np.round(offset, 5).tolist())


def piece_poses(frame: Frame, pairs, rim_y_in: float, outward: float) -> list:
    """Middle of each piece and, per half, its colour and the way its own
    end points.  `rim_y_in` is where the half's outer end is along its
    local y, and `outward` the sign of local y toward that end."""
    out = []
    for (name_a, parent, a), (name_b, _, b) in pairs:
        ends = [a[:, 3] / MM + a[:, :3] @ np.array([0.0, rim_y_in, 0.0]),
                b[:, 3] / MM + b[:, :3] @ np.array([0.0, rim_y_in, 0.0])]
        middle = (ends[0] + ends[1]) / 2.0
        halves = [dict(part=name, toward=frame.direction(place[:, :3] @ np.array([0.0, outward, 0.0])))
                  for name, place in ((name_a, a), (name_b, b))]
        out.append(dict(assembly=parent, middle_in=frame.point(middle * MM), halves=halves))
    return out


def field_layout(frame: Frame, instances, cup_rim_y_in: float, cup_waist_y_in: float) -> dict:
    layout = dict(goals=[], risers=[], loaders=[], toggles=[], rails=[])
    kinds = ((GOAL_TAPER, "goals"), (NEUTRAL_RISER, "risers"), (CENTER_RISER, "risers"), (LOADER_TUBE, "loaders"),
             (TOGGLE_BAR, "toggles"), (PERIMETER_RAIL, "rails"))
    for name, parent, _, place in instances:
        for number, kind in kinds:
            if name.startswith(number):
                layout[kind].append(dict(part=name, assembly=parent, **pose(frame, place)))
    layout["pins"] = piece_poses(frame, paired_halves(instances, PIN_HALF, "276-9250-80", 0.0), 0.0, 1.0)
    layout["cups"] = piece_poses(frame, paired_halves(instances, CUP_HALF, "276-9250-81", cup_waist_y_in),
                                 cup_rim_y_in, -1.0)
    return layout


def main(step: Path) -> None:
    started = time.time()
    if not step.is_file():
        raise SystemExit(f"{step} is not a file")
    digest = hashlib.sha256(step.read_bytes()).hexdigest()
    tool, _doc = read_step(step)
    instances, shapes = leaf_instances(tool)
    print(f"read {step.name}: {len(instances)} parts placed, {len(shapes)} different, {time.time() - started:.0f} s",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "brep"
    scratch.mkdir(exist_ok=True)
    frame = Frame(tile_top(instances, shapes))

    jobs = {}
    for key, number in (("pin_half", PIN_HALF), ("cup_half", CUP_HALF), ("goal_taper", GOAL_TAPER)):
        name, shape = prototype(shapes, number)
        brep = scratch / f"{number}.brep"
        write_brep(shape, brep)
        low, high = bounds(shape)
        jobs[key] = (name, shape, (str(brep), round(float(low[1]), 3), round(float(high[1]), 3)))
    cut = OUT / f"profiles-{digest[:16]}.json"
    if cut.is_file():
        profiles = json.loads(cut.read_text())
        print(f"read the profiles cut from this STEP file before, {cut.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            profiles = dict(zip(jobs, pool.map(radial_profile, [job[2] for job in jobs.values()])))
        cut.write_text(json.dumps(profiles))
        print(f"cut the Pin, Cup and Goal profiles, {time.time() - started:.0f} s", flush=True)

    cup_waist_y = jobs["cup_half"][2][2]
    (_, _, pin_a), (_, _, pin_b) = paired_halves(instances, PIN_HALF, "276-9250-80", 0.0)[0]
    (_, _, cup_a), (_, _, cup_b) = paired_halves(instances, CUP_HALF, "276-9250-81", cup_waist_y)[0]
    parts = {}
    for key, (name, shape, _) in jobs.items():
        low, high = bounds(shape)
        parts[key] = dict(part=name, sectors=SECTORS, step_in=STEP_IN, low_in=low.tolist(), high_in=high.tolist(),
                          **mass_properties(shape), **profiles[key])
    parts["pin_half"]["other_half"] = halves_relation(pin_a, pin_b)
    parts["cup_half"]["other_half"] = halves_relation(cup_a, cup_b)
    for key, number in (("neutral_riser", NEUTRAL_RISER), ("center_riser", CENTER_RISER), ("loader_tube", LOADER_TUBE),
                        ("toggle_bar", TOGGLE_BAR)):
        name, shape = prototype(shapes, number)
        low, high = bounds(shape)
        parts[key] = dict(part=name, low_in=low.tolist(), high_in=high.tolist())

    source = dict(file=step.name, sha256=digest, bytes=step.stat().st_size)
    (OUT / "shapes.json").write_text(json.dumps(dict(source=source, parts=parts)))
    (OUT / "field.json").write_text(json.dumps(dict(source=source, tile_top_cad_in=frame.tile_top,
                                                    **field_layout(frame, instances, parts['cup_half']['low_in'][1],
                                                                   cup_waist_y))))

    meshes, blob, offset = {}, bytearray(), 0
    for number in VIEW_PARTS:
        name, shape = prototype(shapes, number)
        vertices, faces = triangles(shape)
        entry = dict(part=name, vertices=len(vertices), triangles=len(faces), vertex_offset=offset)
        blob += vertices.tobytes()
        offset += vertices.nbytes
        entry["index_offset"] = offset
        blob += faces.tobytes()
        offset += faces.nbytes
        meshes[number] = entry
    instances_out = {}
    for name, parent, _, place in instances:
        number = next((n for n in VIEW_PARTS if name.startswith(n)), None)
        if number is not None:
            instances_out.setdefault(number, []).append(dict(part=name, assembly=parent, **pose(frame, place)))
    (OUT / "view.bin").write_bytes(bytes(blob))
    (OUT / "view.json").write_text(json.dumps(dict(source=source, units="inches, each part in its own CAD frame",
                                                   frame="vexsim field inches; rotation maps the part frame",
                                                   cad_axes=frame.axes.tolist(), tile_top_cad_in=frame.tile_top,
                                                   meshes=meshes, instances=instances_out)))
    for brep in scratch.iterdir():
        brep.unlink()
    scratch.rmdir()
    print(f"wrote {OUT} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))

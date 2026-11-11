"""Planar rigid-body dynamics for a wheeled robot.

Three body degrees of freedom (forward, sideways, yaw) plus one spin state and
two lagged slip states per wheel.  Nothing is assumed about the wheels rolling
without slipping -- every force on the robot comes out of a contact patch.

Per step:

  1. Distribute weight over the contact patches, including load transfer from
     the accelerations of the previous step.
  2. Work out each wheel's contact-patch velocity from the body twist, turn
     that into slip, and lag the slip through the relaxation length.
  3. Get a force from each tire, sum forces and moments about the CG.
  4. Integrate the body and each wheel's spin.
  5. Rock the chassis up onto an edge of its wheels if the accelerations
     need more than the weight can balance (`Tilt`).

The weight distribution is the only part that is not a plain equation.  With
more than three contact points the load split is statically indeterminate, so
it is solved as the distribution closest to even that still balances the
vertical force and both tipping moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .chassis import Chassis
from .units import G, clamp
from .wheels import tire_forces

V_FLOOR = 0.05        # m/s, slip denominator floor -- below this a tire creeps
CREEP_SPEED = 0.025   # m/s, patch sliding below which the tire force fades out
RELAX_TAU_MAX = 0.05  # s, slowest the slip state is allowed to lag the motion
OMEGA_EPS = 0.5       # rad/s, smoothing for direction-dependent drag
KAPPA_CLAMP = 20.0
#: m/s^2, a thousand g: no tire or motor gets near it, and a diverging step
#: would otherwise throw the robot over and hide that it diverged
DIVERGED_ACCELERATION = 1.0e4


@dataclass
class BodyState:
    x: float = 0.0        # world position, m
    y: float = 0.0
    theta: float = 0.0    # heading, rad, CCW from +x
    u: float = 0.0        # body-frame forward velocity, m/s
    v: float = 0.0        # body-frame left velocity, m/s
    r: float = 0.0        # yaw rate, rad/s
    ax: float = 0.0       # body-frame accelerations, m/s^2
    ay: float = 0.0
    alpha: float = 0.0    # yaw acceleration, rad/s^2

    @property
    def speed(self) -> float:
        return math.hypot(self.u, self.v)

    def world_velocity(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.u * c - self.v * s, self.u * s + self.v * c)


def _solve3(a, b):
    """Solve a consistent 3x3 system, including redundant constraints.

    A rank-deficient support layout has fewer independent moment equations;
    that does not make its remaining force/moment constraints optional.
    Free dual variables are set to zero. Return None only if inconsistent.
    """
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    tolerance = 1e-12 * max(1.0, max(abs(v) for row in a for v in row))
    rhs_tolerance = 1e-12 * max(1.0, max(abs(v) for v in b))
    pivots = []
    rank = 0
    for col in range(3):
        piv = max(range(rank, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= tolerance:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        pv = m[rank][col]
        for r in range(rank + 1, 3):
            f = m[r][col] / pv
            for c in range(col, 4):
                m[r][c] -= f * m[rank][c]
        pivots.append(col)
        rank += 1
    if any(abs(m[r][3]) > rhs_tolerance for r in range(rank, 3)):
        return None
    x = [0.0, 0.0, 0.0]
    for r in reversed(range(rank)):
        col = pivots[r]
        s = m[r][3] - sum(m[r][c] * x[c] for c in range(col + 1, 3))
        x[col] = s / m[r][col]
    return x


#: Both the support face and its normal matrix depend only on where the wheels
#: are, which does not change while the robot drives.  Only `target` moves, so
#: build them once per geometry instead of 2000 times a second.
_FACE_CACHE: Dict[tuple, tuple] = {}


def _face(positions, free):
    key = (tuple(positions), tuple(free))
    cached = _FACE_CACHE.get(key)
    if cached is None:
        a = [[1.0, positions[i][0], positions[i][1]] for i in free]
        gram = [[sum(v[row] * v[col] for v in a) for col in range(3)]
                for row in range(3)]
        if len(_FACE_CACHE) > 512:
            _FACE_CACHE.clear()
        cached = _FACE_CACHE[key] = (a, gram)
    return cached


def _minimum_loads(positions, free, target):
    """Least-squared loads on one candidate support face, before lift limits."""
    a, gram = _face(positions, free)
    dual = _solve3(gram, target)
    if dual is None:
        return None
    d0, d1, d2 = dual
    return [v[0] * d0 + v[1] * d1 + v[2] * d2 for v in a]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points):
    """Convex hull of two or more distinct points, counter-clockwise."""
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _support_point(positions, target):
    """Closest point to target in the convex hull of the contact positions.

    Outside that hull no nonnegative support loads can balance the requested
    moments. Saturating at its edge preserves vertical force; `Tilt` is what
    rocks the chassis up onto that edge.
    """
    points = sorted(set(positions))
    if len(points) == 1:
        return points[0]
    hull = _hull(points)
    cross = _cross
    edges = list(zip(hull, hull[1:] + hull[:1]))
    if len(hull) >= 3 and all(cross(a, b, target) >= 0 for a, b in edges):
        return target
    candidates = []
    for a, b in edges:
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = clamp(((target[0] - a[0]) * dx + (target[1] - a[1]) * dy)
                  / (dx * dx + dy * dy), 0.0, 1.0)
        candidates.append((a[0] + t * dx, a[1] + t * dy))
    return min(candidates, key=lambda p: (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2)


def _nonnegative_loads(positions, target):
    """Minimum-norm nonnegative loads with all feasible moments preserved.

    Removing every negative unconstrained load at once can discard a support
    needed by the constrained optimum. Instead explore those candidate faces
    separately, memoizing faces and pruning against the best feasible norm.
    At least one currently negative component must be zero at the optimum:
    otherwise a short step toward the unconstrained solution is feasible and
    strictly improves the norm. Normal fully supported layouts need one solve.
    """
    n = len(positions)
    best = None
    best_norm = math.inf
    visited = set()
    tolerance = 1e-12 * max(1.0, abs(target[0]))

    def visit(free):
        nonlocal best, best_norm
        if not free or free in visited:
            return
        visited.add(free)
        solution = _minimum_loads(positions, free, target)
        if solution is None:
            return
        norm = sum(v * v for v in solution)
        if norm >= best_norm:
            return
        negative = sorted((v, i) for i, v in zip(free, solution) if v < -tolerance)
        if not negative:
            best = [0.0] * n
            for i, v in zip(free, solution):
                best[i] = max(0.0, v)
            best_norm = norm
            return
        for _, lifted in negative:
            visit(tuple(i for i in free if i != lifted))

    visit(tuple(range(n)))
    return best


def normal_loads(chassis: Chassis, ax: float, ay: float,
                 positions: List[tuple], gravity: Optional[float] = None) -> List[float]:
    """Vertical load on each contact patch, N.

    Solves for the loads closest to an even split that still satisfy

        sum(Fz)          = m*g - preloads
        sum(x_i * Fz_i)  = -h*m*ax     (pitch)
        sum(y_i * Fz_i)  = -h*m*ay     (roll)

    Additional loads are nonnegative; configured preloads remain fixed lower
    bounds. For impossible tipping moments the center of pressure saturates at
    the support hull; `Tilt` is what then rocks the chassis up onto that edge.
    """
    units = chassis.wheel_units
    n = len(units)
    if n == 0:
        return []

    preload = [u.preload for u in units]
    g = chassis.gravity if gravity is None else gravity
    total = chassis.mass * g - sum(preload)
    if total <= 0.0:
        # Fixed preloads alone already consume the available weight. If they
        # exceed it there is no solution under this lower-bound model.
        return preload
    h = chassis.cg_height

    cx = -h * chassis.mass * ax
    cy = -h * chassis.mass * ay
    # preloaded wheels contribute their own moments
    cx -= sum(preload[i] * positions[i][0] for i in range(n))
    cy -= sum(preload[i] * positions[i][1] for i in range(n))

    # A spring-loaded wheel carries its spring force and nothing more: the
    # chassis rests on the others. Sharing the weight over every contact put
    # 30-38% of it on two 4 N odometry wheels and cost the drive that much grip.
    bearing = [i for i in range(n) if preload[i] <= 0.0]
    if not bearing:
        bearing = list(range(n))
    bearing_positions = [positions[i] for i in bearing]
    target = [total, cx, cy]
    extra = None
    direct = _minimum_loads(bearing_positions, range(len(bearing)), target)
    if direct is not None and min(direct) >= 0.0:
        extra = direct
    else:
        px, py = _support_point(bearing_positions, (cx / total, cy / total))
        extra = _nonnegative_loads(bearing_positions, [total, total * px, total * py])
        if extra is None:
            raise RuntimeError("normal-load constraints could not be resolved on the support hull")
    loads = list(preload)
    for k, i in enumerate(bearing):
        loads[i] += extra[k]
    return loads


GUESSES: Dict[str, str] = {
    "Pitch and roll inertia": "Unless a world sets them, a solid box the frame's length and width and twice the "
                              "CG height, about the CG: m (L^2 + (2h)^2) / 12. No VEX robot's has been "
                              "published.",
    "Tipping": "The robot rocks as one rigid body about one edge of its wheels; the wheels off the floor carry "
               "nothing until it lands, and it lands without bouncing. Not modelled: rocking about a corner, the "
               "weight on the floor changing as the CG rises and falls, and anything after it falls past the "
               "edge (it stops where it is).",
}


@dataclass(frozen=True)
class FootprintEdge:
    """One side of the wheels' footprint: its outward normal in the body
    frame, how far it is from the CG, and the wheels on it."""
    nx: float
    ny: float
    reach: float
    wheels: Tuple[int, ...]


def footprint_edges(positions, bearing) -> List[FootprintEdge]:
    """The sides of the convex hull of the bearing wheels' contacts (relative
    to the CG) that the robot can rock up onto.  Fewer than three contacts
    out of line make no footprint, so nothing to rock on."""
    points = sorted({positions[i] for i in bearing})
    hull = _hull(points) if len(points) >= 2 else points
    if len(hull) < 3:
        return []
    edges = []
    for a, b in zip(hull, hull[1:] + hull[:1]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        nx, ny = (b[1] - a[1]) / length, (a[0] - b[0]) / length
        reach = nx * a[0] + ny * a[1]
        on_edge = tuple(i for i in bearing if abs(nx * positions[i][0] + ny * positions[i][1] - reach) < 1e-9)
        edges.append(FootprintEdge(nx, ny, reach, on_edge))
    return edges


@dataclass
class Tilt:
    """The chassis rocked up about one edge of its wheel footprint.

    `angle` is 0 with every wheel down.  Past `balance` the CG is over the
    edge and the robot falls over: `fallen` is then set and stays set.
    `most` holds the largest angle since it was last cleared, and the edge
    and balance angle it had then."""
    edge: Optional[FootprintEdge] = None
    angle: float = 0.0          # rad
    rate: float = 0.0           # rad/s
    balance: float = math.pi / 2.0
    most: float = 0.0           # rad
    most_edge: Optional[FootprintEdge] = None
    most_balance: float = math.pi / 2.0
    fallen: bool = False

    def clear_most(self) -> None:
        self.most, self.most_edge, self.most_balance = self.angle, self.edge, self.balance


def _edge_loads(positions, edge: FootprintEdge, along: float, total: float) -> Dict[int, float]:
    """Split `total` over the wheels on `edge` so that it acts `along` the
    edge (measured on the edge's tangent), clamped to the end wheels."""
    tx, ty = -edge.ny, edge.nx
    spots = {i: tx * positions[i][0] + ty * positions[i][1] for i in edge.wheels}
    low, high = min(spots.values()), max(spots.values())
    if high - low < 1e-12:
        return {i: total / len(spots) for i in spots}
    along = clamp(along, low, high)
    count = len(spots)
    sum_s = sum(spots.values())
    sum_ss = sum(v * v for v in spots.values())
    det = count * sum_ss - sum_s * sum_s
    alpha = (sum_ss - sum_s * along) / det
    beta = (count * along - sum_s) / det
    loads = {i: total * (alpha + beta * v) for i, v in spots.items()}
    if min(loads.values()) >= 0.0:
        return loads
    first = min(spots, key=spots.get)
    last = max(spots, key=spots.get)
    share = (along - low) / (high - low)
    loads = {i: 0.0 for i in spots}
    loads[first], loads[last] = total * (1.0 - share), total * share
    return loads


class Dynamics:
    """Steps a chassis forward in time."""

    def __init__(self, chassis: Chassis, gravity: Optional[float] = None):
        self.chassis = chassis
        self.gravity = chassis.gravity if gravity is None else gravity
        self.chassis.gravity = self.gravity
        self.state = BodyState()
        # locked = the robot is pinned (shoved against a wall, wheels held).
        # Forces and currents are still computed; nothing moves.
        self.locked = False
        # constrained = the chassis is against an obstacle but its wheels remain
        # free to turn.  The obstacle reaction pins the body while tire forces
        # still reveal how hard the robot can push.
        self.constrained = False
        # wheel positions relative to the CG, cached
        self.positions = [(u.x - chassis.cg_x, u.y - chassis.cg_y)
                          for u in chassis.wheel_units]
        self.bearing = [i for i, u in enumerate(chassis.wheel_units) if u.preload <= 0.0] \
            or list(range(len(chassis.wheel_units)))
        self.edges = footprint_edges(self.positions, self.bearing)
        self.tilt = Tilt()
        # cos/sin of each wheel's contact direction, rebuilt only when a wheel
        # actually steers -- on a differential drive, never
        self._geometry = [None] * len(chassis.wheel_units)
        # aggregate outputs, refreshed each step
        self.total_fx = 0.0
        self.total_fy = 0.0
        self.total_mz = 0.0
        self.total_normal = 0.0
        self.wheel_slip_power = 0.0     # W lost to tires scrubbing
        self.rolling_loss_power = 0.0
        self.drag_power = 0.0
        self.scrub_loss_power = 0.0    # already included in drag_power
        self.normal_moment_residual = (0.0, 0.0)
        self.normal_balance_feasible = True
        self.kinetic_energy = 0.0
        self.any_slipping = False
        self.tipping_margin = 1.0
        self.lateral_transfer_ratio = 0.0
        self.pitch_transfer_ratio = 0.0
        # Anything pushing on the robot that is not a tire: game pieces, another
        # robot, a hand.  Body frame, N and N*m, cleared by whoever sets them.
        self.ext_fx = 0.0
        self.ext_fy = 0.0
        self.ext_mz = 0.0

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> None:
        ch = self.chassis
        st = self.state
        units = ch.wheel_units
        pos = self.positions

        if self.tilt.fallen:
            loads = [0.0] * len(units)
        elif self.tilt.edge is not None:
            loads = self._tilted_loads()
        else:
            loads = normal_loads(ch, st.ax, st.ay, pos, self.gravity)
        self.normal_moment_residual = (
            sum(x * fz for (x, y), fz in zip(pos, loads)) + ch.cg_height * ch.mass * st.ax,
            sum(y * fz for (x, y), fz in zip(pos, loads)) + ch.cg_height * ch.mass * st.ay,
        )
        self.normal_balance_feasible = (
            abs(sum(loads) - ch.mass * self.gravity) < 1e-8
            and max(map(abs, self.normal_moment_residual)) < 1e-8)

        fx_sum = fy_sum = mz_sum = 0.0
        slip_power = 0.0
        roll_power = 0.0
        scrub_power = 0.0
        self.any_slipping = False

        wheel_accels = []
        # one wheel's fair share of the robot's weight
        share = ch.mass * self.gravity / max(len(units), 1)
        mu = ch.surface_mu
        geometry = self._geometry
        # yaw scrub is the same for every wheel, and the math lookups in here
        # run six times a wheel at 2 kHz
        yaw_scrub = math.tanh(st.r / 0.2)
        tanh = math.tanh
        exp = math.exp
        hypot = math.hypot
        for i, u in enumerate(units):
            x, y = pos[i]
            fz = loads[i]
            u.fz = fz

            # contact patch velocity in the body frame
            vpx = st.u - st.r * y
            vpy = st.v + st.r * x
            roller = u.wheel.roller_angle
            cached = geometry[i]
            if cached is None or cached[0] != (u.steer_angle, roller):
                if 1e-9 < abs(roller) < math.pi / 2.0 - 1e-9:
                    # roller_angle is measured from the wheel axle.  The
                    # direction that cannot roll freely is therefore diagonal on
                    # a mecanum; its sign selects the roller mounting hand.
                    roller_offset = math.copysign(math.pi / 2.0 - abs(roller), roller)
                else:
                    roller_offset = 0.0
                d = u.steer_angle + roller_offset
                cached = geometry[i] = ((u.steer_angle, roller), roller_offset,
                                        math.cos(d), math.sin(d),
                                        math.cos(roller_offset))
            _, roller_offset, cd, sd, cos_roller = cached
            vx = cd * vpx + sd * vpy
            vy = -sd * vpx + cd * vpy
            u.vx, u.vy = vx, vy

            # Normalise by the faster of the two surfaces, not just the body.
            # Using |vx| alone makes kappa explode off the line -- the wheel is
            # turning, the robot is not yet, and the floor divides a real 1 m/s
            # of wheel speed by 0.05 -- which reads as total wheelspin when the
            # tire is only creeping a few percent.
            surface_speed = u.omega * u.radius * cos_roller
            denom = max(abs(vx), abs(surface_speed), V_FLOOR)
            kappa_ss = clamp((surface_speed - vx) / denom, -KAPPA_CLAMP, KAPPA_CLAMP)
            tan_a_ss = clamp(vy / denom, -KAPPA_CLAMP, KAPPA_CLAMP)
            u.kappa_ss, u.tan_alpha_ss = kappa_ss, tan_a_ss

            # Relaxation: slip builds up over a rolling distance, not instantly.
            # The rolling-distance form degenerates as the robot slows down --
            # at the speed floor it puts the tire force 0.6 s behind the motion,
            # and a lagged lateral force is what makes a chassis fishtail on the
            # way out of a turn.  Cap the lag at RELAX_TAU_MAX, which is what
            # every Pacejka implementation does for the same reason.
            tire = u.wheel.tire
            rate = max(denom / max(tire.relaxation_length, 1e-4),
                       1.0 / RELAX_TAU_MAX)
            decay = 1.0 - exp(-rate * dt)
            u.kappa += (kappa_ss - u.kappa) * decay
            u.tan_alpha += (tan_a_ss - u.tan_alpha) * decay

            fx, fy = tire_forces(tire, fz, u.kappa, u.tan_alpha, mu)

            # Fade the force out as the contact patch stops sliding.  A slip
            # model normalises by speed, so at a standstill a hundredth of an
            # inch per second of creep reads as full slip and the tire answers
            # with full force -- which flips sign every few steps and rings the
            # whole robot.  Friction opposes sliding, so with no sliding there
            # is no force: below CREEP_SPEED of actual patch motion this eases
            # it off.  Above about an inch a second nothing changes.
            slide = hypot(surface_speed - vx, vy)
            fade = tanh(slide / CREEP_SPEED)
            fx *= fade
            fy *= fade
            u.fx, u.fy = fx, fy
            # How far past the grip peak this tire is, as a multiple of it.
            # Sideways motion only counts against a wheel that has to scrub
            # sideways to do it: an omni just rolls on its rollers, so a big
            # slip angle there is the wheel working as designed, not a slide.
            measure = abs(u.kappa) / tire.slip_ratio_peak
            if abs(u.wheel.roller_angle) < 0.1:
                measure = max(measure, abs(u.tan_alpha) / tire.slip_angle_peak)
            u.slip_measure = measure
            # A wheel carrying almost no load spins up freely; that is lift-off,
            # not a tire losing grip, and it should not read as slip.
            u.slipping = measure > 1.5 and fz > 0.2 * share
            self.any_slipping = self.any_slipping or u.slipping

            # back to the body frame
            fbx = cd * fx - sd * fy
            fby = sd * fx + cd * fy
            fx_sum += fbx
            fy_sum += fby
            mz_sum += x * fby - y * fbx

            # contact patch twisting under the wheel resists yaw
            scrub_torque = tire.scrub_arm(fz, u.radius) * fz * yaw_scrub
            mz_sum -= scrub_torque
            scrub_power += scrub_torque * st.r

            # power actually burned in the contact patch
            slip_power += abs(fx * (surface_speed - vx)) + abs(fy * vy)

            # rolling resistance acts on the wheel, not the body
            rr = (tire.rolling_resistance(fz, u.radius) * fz * u.radius
                  * tanh(u.omega / OMEGA_EPS))
            u.roll_resist_torque = rr
            roll_power += abs(rr * u.omega)

            tau = u.motor_torque() if u.driven else 0.0
            inertia = max(u.effective_inertia, 1e-9)
            contact_torque = fx * u.radius * cos_roller
            wheel_accels.append((tau - contact_torque - rr) / inertia)

        # body-level losses
        drag_x = ch.bearing_drag * st.u + ch.aero_drag * st.u * abs(st.u)
        drag_y = ch.bearing_drag * st.v
        yaw_drag = ch.yaw_drag * st.r
        self.scrub_loss_power = scrub_power
        self.drag_power = (abs(drag_x * st.u) + abs(drag_y * st.v)
                           + abs(yaw_drag * st.r) + scrub_power)

        m = ch.mass
        # These are the CG's inertial acceleration components expressed in the
        # rotating body frame.  du/dt and dv/dt additionally contain the frame
        # terms below; load transfer and the IMU need the inertial quantities.
        st.ax = (fx_sum - drag_x + self.ext_fx) / m
        st.ay = (fy_sum - drag_y + self.ext_fy) / m
        st.alpha = (mz_sum - yaw_drag + self.ext_mz) / ch.inertia_z

        if self.locked:
            st.u = st.v = st.r = 0.0
            st.ax = st.ay = st.alpha = 0.0
            for u in units:
                u.omega = 0.0
                u.sync_motors()
        else:
            if self.constrained or self.tilt.fallen:
                st.u = st.v = st.r = 0.0
                st.ax = st.ay = st.alpha = 0.0
            else:
                # Apply the force impulse at the middle orientation, advance
                # position with mean velocity, then rotate velocity into the
                # new body frame exactly. Explicit Euler on the rotating-frame
                # terms multiplies free-flight energy by 1 + (r*dt)^2 per step.
                turn = (st.r + 0.5 * st.alpha * dt) * dt
                half_c, half_s = math.cos(0.5 * turn), math.sin(0.5 * turn)
                ax = half_c * st.ax - half_s * st.ay
                ay = half_s * st.ax + half_c * st.ay
                u_inertial = st.u + ax * dt
                v_inertial = st.v + ay * dt
                st.r += st.alpha * dt

                c, s = math.cos(st.theta), math.sin(st.theta)
                mean_u = 0.5 * (st.u + u_inertial)
                mean_v = 0.5 * (st.v + v_inertial)
                st.x += (mean_u * c - mean_v * s) * dt
                st.y += (mean_u * s + mean_v * c) * dt
                st.theta += turn
                c, s = math.cos(turn), math.sin(turn)
                st.u = c * u_inertial + s * v_inertial
                st.v = -s * u_inertial + c * v_inertial
                st.ax, st.ay = half_c * st.ax + half_s * st.ay, -half_s * st.ax + half_c * st.ay

            for i, u in enumerate(units):
                u.omega += wheel_accels[i] * dt
                u.angle += u.omega * dt
                u.sync_motors()

        if not (abs(st.u) < 1e6 and abs(st.r) < 1e6 and abs(st.x) < 1e6
                and abs(st.ax) < DIVERGED_ACCELERATION and abs(st.ay) < DIVERGED_ACCELERATION):
            raise RuntimeError(
                "dynamics diverged -- check the drag coefficients, mass and "
                "inertia, or lower SimConfig.dt")

        self.total_fx, self.total_fy, self.total_mz = fx_sum, fy_sum, mz_sum
        self.total_normal = sum(loads)
        self.wheel_slip_power = slip_power
        self.rolling_loss_power = roll_power
        self.kinetic_energy = 0.5 * m * (st.u ** 2 + st.v ** 2) \
            + 0.5 * ch.inertia_z * st.r ** 2 \
            + sum(0.5 * u.effective_inertia * u.omega ** 2 for u in units)
        self._load_transfer(loads)
        self._rock(dt)

    def _pressure_center(self):
        """Where on the floor, relative to the CG, the weight has to act to
        balance the accelerations, and the weight the bearing wheels share."""
        ch, st = self.chassis, self.state
        preload = [u.preload for u in ch.wheel_units]
        total = ch.mass * self.gravity - sum(preload)
        x = -ch.cg_height * ch.mass * st.ax - sum(f * p[0] for f, p in zip(preload, self.positions))
        y = -ch.cg_height * ch.mass * st.ay - sum(f * p[1] for f, p in zip(preload, self.positions))
        return x / total, y / total, total

    def _tilted_loads(self) -> List[float]:
        """The weight on the wheels along the edge the chassis is rocked up on;
        the wheels off the floor carry nothing."""
        x, y, total = self._pressure_center()
        edge = self.tilt.edge
        loads = [u.preload for u in self.chassis.wheel_units]
        for i, load in _edge_loads(self.positions, edge, -edge.ny * x + edge.nx * y, total).items():
            loads[i] += load
        return loads

    def _rock(self, dt: float) -> None:
        """Rock the chassis about the footprint edge its accelerations push the
        centre of pressure past, until it lands again or falls over."""
        tilt, ch, st = self.tilt, self.chassis, self.state
        if tilt.fallen or not self.edges:
            return
        if tilt.edge is None:
            x, y, _ = self._pressure_center()
            edge = max(self.edges, key=lambda e: e.nx * x + e.ny * y - e.reach)
            if edge.nx * x + edge.ny * y <= edge.reach:
                return
            tilt.edge = edge
        edge = tilt.edge
        h, d, g = ch.cg_height, edge.reach, self.gravity
        push = -(st.ax * edge.nx + st.ay * edge.ny)
        c, s = math.cos(tilt.angle), math.sin(tilt.angle)
        inertia = ch.level_inertia(edge.nx, edge.ny) + ch.mass * (d * d + h * h)
        tilt.rate += ch.mass * (push * (h * c + d * s) - g * (d * c - h * s)) / inertia * dt
        tilt.angle += tilt.rate * dt
        if tilt.angle <= 0.0:
            tilt.edge, tilt.angle, tilt.rate = None, 0.0, 0.0
            return
        tilt.balance = math.atan2(d, h)
        if tilt.angle > tilt.most:
            tilt.most, tilt.most_edge, tilt.most_balance = tilt.angle, edge, tilt.balance
        tilt.fallen = tilt.angle >= tilt.balance

    def _load_transfer(self, loads: List[float]) -> None:
        """Load transfer ratios, the standard rollover measure.

        LTR is the difference between the two sides' loads over the total.  At
        1.0 one side carries everything and the robot is on two wheels; the
        pitch version is the same thing front to back.  `tipping_margin` is
        how much is left before that happens.
        """
        total = sum(loads) or 1.0
        left = right = front = rear = 0.0
        for i, fz in enumerate(loads):
            x, y = self.positions[i]
            if y > 0:
                left += fz
            elif y < 0:
                right += fz
            if x > 0:
                front += fz
            elif x < 0:
                rear += fz
        self.lateral_transfer_ratio = abs(left - right) / total
        self.pitch_transfer_ratio = abs(front - rear) / total
        self.tipping_margin = clamp(
            1.0 - max(self.lateral_transfer_ratio, self.pitch_transfer_ratio), 0.0, 1.0)

    # ----------------------------------------------------------------- reset
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        """Reset motion and observations, retaining caller-imposed constraints."""
        self.state = BodyState(x=x, y=y, theta=theta)
        self.total_fx = self.total_fy = self.total_mz = self.total_normal = 0.0
        self.wheel_slip_power = self.rolling_loss_power = self.drag_power = 0.0
        self.scrub_loss_power = self.kinetic_energy = 0.0
        self.normal_moment_residual = (0.0, 0.0)
        self.normal_balance_feasible = True
        self.any_slipping = False
        self.tipping_margin = 1.0
        self.lateral_transfer_ratio = self.pitch_transfer_ratio = 0.0
        self.tilt = Tilt()
        for u in self.chassis.wheel_units:
            u.omega = 0.0
            u.angle = 0.0
            u.kappa = 0.0
            u.tan_alpha = 0.0
            u.kappa_ss = u.tan_alpha_ss = 0.0
            u.vx = u.vy = u.fz = 0.0
            u.fx = u.fy = 0.0
            u.slipping = False
            u.slip_measure = 0.0
            u.drive_torque = u.roll_resist_torque = u.power_gear_loss = 0.0
            u.sync_motors()

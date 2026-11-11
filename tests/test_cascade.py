"""The cascade-lift robot as MuJoCo bodies: how fast the lift rises and what
stops it, the pneumatic claw that grips by its jaws' contacts, the bite that
seats a Goal under the claw, Pins in Goals, the Toggle roller, Match Loads, the robot's size
and motors against the Override rules, and how its pieces fit together."""

import dataclasses
import math
import os
import sys
import unittest
import unittest.mock

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Simulator, cascade, presets
from vexsim import override_cad as cad
from vexsim import override_field as of
from vexsim.motor import BrakeMode
from vexsim.override import OverrideWorld
from vexsim.pieces import turned_by
from vexsim.sim import SimConfig
from vexsim.units import INCH

GEOMETRY = cascade.GEOMETRY
CHASSIS = presets.override_cascade()
HALF_LENGTH = CHASSIS.length / 2.0
CLAW_AHEAD = cascade.CascadeRobot().claw_forward(CHASSIS)
CLEAR_OF_EVERYTHING = (-40.0 * INCH, -12.0 * INCH, 0.0)
CUP_HALF_IN = cad.cup().half_length / INCH
#: How far below a Cup's top its rim starts to flare, from the CAD's profile
CUP_FLARE_IN = 0.373


class LiftHold:
    """Holds the claw at a height: a spring on the lift's motors toward the
    angle that winds in string for it, moved on slowly by how far the
    stretched strings leave the claw short of it, plus, off the bottom, the
    voltage that balances what the first stage carries against the bands."""

    def __init__(self, world):
        self.world = world
        self.target = 0.0
        self.stretch = 0.0

    def height(self, grip_height_in):
        self.target = (grip_height_in - GEOMETRY.grip_lowest) * INCH / GEOMETRY.stages

    def step(self):
        g, lift = GEOMETRY, self.world.mechanism.lift
        held = self.world.grip.mass if self.world.grip is not None else 0.0
        stage_shares = sum(k * g.stage_mass_kg for k in range(1, g.stages))
        carried = (stage_shares + g.stages * (g.carriage_mass_kg + held)) * self.world.sim.config.gravity
        stall_force = sum(m.stall_torque for m in lift.motors) * g.lift_efficiency / g.string_per_motor_rad
        balance = (carried - lift.bands()) / stall_force * 12.0 if self.target > 0.0 else 0.0
        short = self.target - lift.rise / g.stages
        if self.target > 0.0 and abs(short) < 0.1 * INCH:
            self.stretch += 0.005 * short
        target_angle = (self.target + self.stretch) / g.string_per_motor_rad
        volts = 40.0 * (target_angle - lift.angle) - 0.8 * lift.motors[0].velocity + balance
        for motor in lift.motors:
            motor.set_voltage(max(-12.0, min(12.0, volts)))


def world_with(pose, preload=None, objects=False):
    sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
    sim.set_pose(*pose)
    world = OverrideWorld(sim, objects=objects, preload=preload, robot=cascade.CascadeRobot())
    hold = LiftHold(world)
    hold.height(GEOMETRY.grip_lowest)
    return sim, world, hold


def run(world, hold, seconds):
    for _ in range(round(seconds * 1000)):
        hold.step()
        world.step(0.001)


def raise_to(world, hold, grip_height_in, seconds=1.0):
    hold.height(grip_height_in)
    run(world, hold, seconds)


def drive(sim, volts):
    for motor in sim.motors:
        motor.set_voltage(volts)


def claw_at(x, y, heading_deg, back_in=0.0, side_in=0.0):
    """The pose that puts the claw's middle `back_in` short of (x, y), `side_in` to the left of it."""
    heading = math.radians(heading_deg)
    reach = CLAW_AHEAD + back_in * INCH
    side = side_in * INCH
    return (x - reach * math.cos(heading) - side * math.sin(heading),
            y - reach * math.sin(heading) + side * math.cos(heading), heading)


def claw_point(world):
    st = world.sim.state
    return (st.x + CLAW_AHEAD * math.cos(st.theta), st.y + CLAW_AHEAD * math.sin(st.theta),
            world.robot.grip_height())


def claw_off(world, x, y):
    point = claw_point(world)
    return math.hypot(point[0] - x, point[1] - y)


def stack_in_the_claw(world):
    """A gray-up Cup with a red/yellow Pin in it, standing under the open claw."""
    x, y, _ = claw_point(world)
    cup = world.pieces.add_cup((x, y, world.pieces.standing_height("cup")), of.DOWN)
    pin = world.pieces.add_pin(("red", "yellow"), (x, y, cup.center()[2] + cad.seats().pin_in_cup), of.UP)
    return cup, pin


def turn_about_its_axis(world, piece, angle):
    q = world.pieces.qpos_at[piece.body]
    turned = np.empty(4)
    mujoco.mju_mulQuat(turned, np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]),
                       world.data.qpos[q + 3:q + 7].copy())
    world.pieces.place_turned(piece, piece.center(), turned)
    mujoco.mj_forward(world.model, world.data)


def claw_middle(world):
    """How high the claw's middle is over the floor, with the robot standing level where it is."""
    jaw = world.mechanism.jaw_bodies[0]
    return float(world.data.xpos[jaw][2]) + GEOMETRY.grip_lowest * INCH


def jaw_contacts(world):
    """Each contact a jaw presses on something with: (which jaw, the body it presses, height, newtons)."""
    m, d = world.model, world.data
    left, _ = world.mechanism.jaw_bodies
    force = np.zeros(6)
    out = []
    for i in range(d.ncon):
        g1, g2 = d.contact.geom[i].tolist()
        if d.contact.efc_address[i] < 0 or not (g1 in world._jaw_geoms or g2 in world._jaw_geoms):
            continue
        jaw, other = (g1, g2) if g1 in world._jaw_geoms else (g2, g1)
        mujoco.mj_contactForce(m, d, i, force)
        if force[0] > 0.0:
            side = "left" if m.geom_bodyid[jaw] == left else "right"
            out.append((side, int(m.geom_bodyid[other]), float(d.contact.pos[i][2]), float(force[0])))
    return out


def carry_to(world, pose):
    """Set the robot down at `pose`, standing still, with what its claw holds
    moved along with it, as if it had driven there.  It must not land in anything."""
    sim, data = world.sim, world.data
    x0, y0, theta0 = sim.state.x, sim.state.y, sim.state.theta
    x1, y1, theta1 = pose
    turn = theta1 - theta0
    c, s = math.cos(turn), math.sin(turn)
    held = [world.grip.piece, *world.grip.carried] if world.grip is not None else []
    moved = []
    for piece in held:
        q = world.pieces.qpos_at[piece.body]
        x, y, z = data.qpos[q:q + 3]
        dx, dy = x - x0, y - y0
        moved.append((piece, (x1 + c * dx - s * dy, y1 + s * dx + c * dy, z), turned_by(turn, data.qpos[q + 3:q + 7])))
    rest = world.drive.robot_qpos()
    dx, dy = rest[0] - x0, rest[1] - y0
    rest[0:2] = (math.cos(theta0) * dx + math.sin(theta0) * dy, -math.sin(theta0) * dx + math.cos(theta0) * dy)
    unturned = np.empty(4)
    mujoco.mju_mulQuat(unturned, np.array([math.cos(theta0 / 2.0), 0.0, 0.0, -math.sin(theta0 / 2.0)]),
                       rest[3:7].copy())
    rest[3:7] = unturned
    world.drive.place(rest, x1, y1, theta1)
    for piece, position, quat in moved:
        world.pieces.place_turned(piece, position, quat)
    mujoco.mj_forward(world.model, data)
    sim.imu.reset(theta1)
    sim.odometry.reset(x1, y1, theta1)
    landed_in = [what for what, _, _ in world.pieces.robot_overlaps() if what not in held]
    if landed_in:
        raise ValueError(f"the robot would land in {landed_in}")


def toggle_at(world, quadrant):
    return next(bar for bar in world.toggles.bars if bar.mount.quadrant == quadrant)


def turn_toggle(world, quadrant, angle):
    world.data.qpos[toggle_at(world, quadrant)._hinge] = angle
    mujoco.mj_forward(world.model, world.data)


class TestLift(unittest.TestCase):
    def test_full_height_takes_about_1_2_seconds(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        for motor in world.mechanism.lift.motors:
            motor.set_voltage(12.0)
        steps = 0
        while world.robot.grip_height() < (GEOMETRY.grip_highest - 0.1) * INCH:
            world.step(0.001)
            steps += 1
            self.assertLess(steps, 1250)
        self.assertGreater(steps, 1120)

    def test_raising_it_draws_its_current_from_the_robots_battery(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        for motor in world.mechanism.lift.motors:
            motor.set_voltage(12.0)
        peak = 0.0
        for _ in range(300):
            world.step(0.001)
            peak = max(peak, sim.battery.current)
        self.assertGreater(peak, 3.0)
        self.assertGreater(sim.battery.charge_used_ah, 0.0)

    def test_each_stage_rises_as_far_on_the_one_below_as_the_first_on_the_frame_less_the_strings_stretch(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        raise_to(world, hold, 20.0)
        rises = world.mechanism.lift.stage_rises()
        self.assertEqual(len(rises), 5)
        masses = cascade.stage_masses(GEOMETRY)
        gravity = world.sim.config.gravity
        stretch = [gravity * sum((i - j) * masses[i - 1] for i in range(j + 1, 6)) / GEOMETRY.string_stiffness
                   for j in range(1, 5)]
        for n, rise in enumerate(rises[1:], start=2):
            with self.subTest(stage=n):
                short = sum((n - j) * stretch[j - 1] for j in range(1, n))
                self.assertAlmostEqual(rise, n * rises[0] - short, delta=0.02 * INCH)
        self.assertAlmostEqual(world.robot.grip_height() / INCH, 20.0, delta=0.1)

    def test_with_the_motors_coasting_it_creeps_down_at_about_0_4_in_a_second(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        raise_to(world, hold, 30.0, seconds=1.5)
        for motor in world.mechanism.lift.motors:
            motor.stop(BrakeMode.COAST)
        for _ in range(500):
            world.step(0.001)
        start = world.robot.grip_height()
        for _ in range(1000):
            world.step(0.001)
        fell = (start - world.robot.grip_height()) / INCH
        self.assertGreater(fell, 0.3)
        self.assertLess(fell, 0.6)

    def test_at_0_v_the_shorted_motors_let_it_down_at_about_0_13_in_a_second(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        raise_to(world, hold, 30.0, seconds=1.5)
        for motor in world.mechanism.lift.motors:
            motor.set_voltage(0.0)
        for _ in range(500):
            world.step(0.001)
        start = world.robot.grip_height()
        for _ in range(1000):
            world.step(0.001)
        fell = (start - world.robot.grip_height()) / INCH
        self.assertGreater(fell, 0.08)
        self.assertLess(fell, 0.2)

    def test_the_slides_rub_harder_the_less_they_overlap(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        lift = world.mechanism.lift
        run(world, hold, 0.2)
        down = float(world.model.dof_frictionloss[lift.dofs[0]])
        raise_to(world, hold, GEOMETRY.grip_highest - 0.5)
        up = float(world.model.dof_frictionloss[lift.dofs[0]])
        self.assertGreater(up, down)

    def test_into_the_top_stop_at_full_speed_the_top_bands_keep_the_pin_in_a_held_cup(self):
        fastest, slowing, thrown = into_the_top_stop(12.0)
        self.assertGreater(fastest, 50.0 * INCH)
        self.assertGreater(slowing, 30.0 * 9.81)
        self.assertLess(thrown, 0.01 * INCH)

    def test_into_the_top_stop_at_a_third_of_full_speed_the_pin_stays_in_its_cup(self):
        fastest, _, thrown = into_the_top_stop(5.0)
        self.assertLess(fastest, 25.0 * INCH)
        self.assertLess(thrown, 1.0 * INCH)


def into_the_top_stop(volts):
    """Raise a held stack into the top stop at `volts`: (the lift's top
    speed, its fastest slowing over 1 ms, how far the Pin rose off its seat)."""
    sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
    cup, pin = stack_in_the_claw(world)
    run(world, hold, 0.3)
    world.robot.claw.command(True)
    run(world, hold, 0.25)
    lift = world.mechanism.lift
    seat = pin.center()[2] - cup.center()[2]
    for motor in lift.motors:
        motor.set_voltage(volts)
    highest, fastest, slowing = seat, 0.0, 0.0
    speed = lift.rise_speed
    for _ in range(2000):
        world.step(0.001)
        fastest = max(fastest, lift.rise_speed)
        slowing = max(slowing, (speed - lift.rise_speed) / 0.001)
        speed = lift.rise_speed
        highest = max(highest, pin.center()[2] - cup.center()[2])
    if world.grip is None or world.grip.piece is not cup:
        raise AssertionError("the claw let go of the Cup")
    return fastest, slowing, highest - seat


class TestRules(unittest.TestCase):
    def test_it_starts_inside_18_inches_with_the_preload_in_its_claw(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING, preload="yellow")
        for size in world.robot.extent(sim.chassis):
            self.assertLessEqual(size, 18.0 * INCH)
        top = max(center[2] + radius for center, radius in world.preload.spheres())
        self.assertLessEqual(top, 18.0 * INCH)

    def test_fully_up_it_stays_inside_24_by_50(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        for motor in world.mechanism.lift.motors:
            motor.set_voltage(12.0)
        for _ in range(1500):
            world.step(0.001)
        length, width, height = world.robot.extent(sim.chassis)
        self.assertLessEqual(max(length, width), 24.0 * INCH)
        self.assertLessEqual(height, 50.0 * INCH)
        self.assertAlmostEqual(height / INCH, 49.97, delta=0.02)

    def test_the_motors_use_all_88_w_and_the_claw_uses_none(self):
        drive_watts = sum(m.spec.peak_power_rating for m in presets.override_cascade().motors)
        mechanism_watts = cascade.CascadeRobot().motor_watts()
        self.assertEqual(mechanism_watts, {"lift": 33.0, "roller": 11.0})
        self.assertEqual(drive_watts, 44.0)
        self.assertEqual(drive_watts + sum(mechanism_watts.values()), 88.0)

    def test_the_drive_is_four_450_rpm_motors_on_2_75_in_omnis_and_a_sprung_sideways_tracking_wheel_in_the_middle(self):
        driven = [u for u in CHASSIS.wheel_units if u.motors]
        self.assertEqual(len(driven), 4)
        for unit in driven:
            self.assertAlmostEqual(unit.radius / INCH, 1.375)
            self.assertAlmostEqual(unit.motors[0].free_speed / unit.gear_ratio * 60.0 / (2.0 * math.pi), 450.0)
        tracking = [u for u in CHASSIS.wheel_units if not u.motors]
        self.assertEqual(len(tracking), 1)
        (wheel,) = tracking
        self.assertEqual((wheel.x, wheel.y), (0.0, 0.0))
        self.assertAlmostEqual(wheel.radius / INCH, 1.0)
        self.assertAlmostEqual(wheel.steer_angle, math.pi / 2.0)
        self.assertGreater(wheel.preload, 0.0)
        self.assertAlmostEqual(CHASSIS.mass, 6.35, delta=0.001)

    def test_the_parts_vex_weighs_and_the_lift_leave_3_50_kg_for_the_rest(self):
        eleven_watt_motor_kg, battery_kg, brain_kg = 0.155, 0.350, 0.285
        robot = cascade.CascadeRobot()
        motors = len(CHASSIS.motors) + len(robot.lift_motors) + len([robot.roller_motor])
        self.assertEqual(motors, 8)
        weighed = motors * eleven_watt_motor_kg + battery_kg + brain_kg
        lifted = cascade.moving_mass(GEOMETRY)
        self.assertAlmostEqual(lifted, 0.98)
        self.assertAlmostEqual(CHASSIS.mass - weighed - lifted, 3.50, delta=0.005)

    def test_the_preload_has_one_of_its_own_colours_up(self):
        with self.assertRaisesRegex(ValueError, "has no 'blue' half"):
            world_with(CLEAR_OF_EVERYTHING, preload="blue")


class TestTrackingWheel(unittest.TestCase):
    def test_the_sideways_tracking_wheel_reads_how_far_the_robot_slides_and_odometry_follows_it(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        body = world.drive.body
        push = [20.0]

        def shove(dt):
            world.data.xfrc_applied[body][1] += push[0]

        world.drive.forces.append(shove)
        start = sim.state.y
        run(world, hold, 0.5)
        push[0] = 0.0
        run(world, hold, 1.0)
        wheel = next(unit for unit in sim.chassis.wheel_units if unit.name == "track_perp")
        slid = sim.state.y - start
        self.assertGreater(slid, 5.0 * INCH)
        self.assertAlmostEqual(sim.tracking["track_perp"].position * wheel.radius, slid, delta=0.01 * slid)
        self.assertAlmostEqual(sim.odometry.y - start, slid, delta=0.01 * slid)


class TestStalledOnAWall(unittest.TestCase):
    def test_the_wheels_hold_at_2_and_4_v_and_spin_at_6_v_where_the_motors_out_pull_their_grip(self):
        """Stalled, 4 V pushes each wheel's tread 97% of the way to its grip
        and bends it about 0.08 in; a wheel that holds turns no further than
        that bend changes, one that slides turns inches.  At 6 V the motors
        reach their 2.5 A limit: 13.4 N a wheel against 11.3 N of grip."""
        for volts, most, least in ((2.0, 0.01, 0.0), (4.0, 0.1, 0.0), (6.0, math.inf, 10.0)):
            with self.subTest(volts=volts):
                sim, world, hold = world_with((-45.0 * INCH, -of.HALF + HALF_LENGTH + 1.0 * INCH, math.pi / 2.0))
                drive(sim, -2.0)
                for _ in range(2000):
                    run(world, hold, 0.001)
                    if world.touching_perimeter:
                        break
                self.assertTrue(world.touching_perimeter)
                drive(sim, -volts)
                run(world, hold, 0.3)
                wheels = [unit for unit in sim.chassis.wheel_units if unit.driven]
                start, y = [unit.angle for unit in wheels], sim.state.y
                run(world, hold, 2.0)
                self.assertAlmostEqual(sim.state.y, y, delta=0.001 * INCH)
                for unit, angle in zip(wheels, start):
                    turned = abs(unit.angle - angle) * unit.radius
                    self.assertLess(turned, most * INCH)
                    self.assertGreaterEqual(turned, least * INCH)


class TestClaw(unittest.TestCase):
    def test_the_piston_closes_in_130_ms_and_opens_in_160_ms(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        mechanism = world.mechanism
        for closed, gap, stroke_ms in ((True, GEOMETRY.jaw_closed_gap, 130), (False, GEOMETRY.jaw_open_gap, 160)):
            with self.subTest(closed=closed):
                world.robot.claw.command(closed)
                steps = 0
                while (mechanism.gap - gap * INCH) * (-1.0 if closed else 1.0) < -0.001 * INCH:
                    world.step(0.001)
                    steps += 1
                self.assertEqual(steps, stroke_ms)
                run(world, hold, 0.2)
                self.assertAlmostEqual(mechanism.gap / INCH, gap, delta=0.015)

    def test_it_takes_a_stack_by_its_cup_and_lifts_the_pin_in_it_too(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        cup, pin = stack_in_the_claw(world)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIs(world.grip.piece, cup)
        self.assertEqual(world.grip.carried, (pin,))
        self.assertGreater(world.mechanism.gap, (GEOMETRY.jaw_closed_gap + 0.2) * INCH)
        self.assertGreater(world.mechanism.squeeze, 40.0)
        raise_to(world, hold, GEOMETRY.grip_highest - 1.0, seconds=1.5)
        self.assertGreater(pin.center()[2], (GEOMETRY.grip_highest - 2.0) * INCH)
        self.assertEqual(world.drops, [])
        self.assertEqual(world.grips[0]["stacked"], [pin.id])

    def test_on_a_stack_each_jaw_presses_the_cup_below_its_rim_and_the_pin_above_its_flange(self):
        for pin_turn in (0.0, 30.0, 60.0):
            with self.subTest(pin_turn=pin_turn):
                sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
                cup, pin = stack_in_the_claw(world)
                turn_about_its_axis(world, pin, math.radians(pin_turn))
                run(world, hold, 0.3)
                world.robot.claw.command(True)
                run(world, hold, 0.25)
                flare_starts = cup.center()[2] + (CUP_HALF_IN - CUP_FLARE_IN) * INCH
                flange_top = pin.center()[2] + cascade.PIN_FLANGE_HALF_IN * INCH
                pressed = {"left": {cup.body: 0.0, pin.body: 0.0}, "right": {cup.body: 0.0, pin.body: 0.0}}
                for side, body, height, force in jaw_contacts(world):
                    self.assertIn(body, (cup.body, pin.body))
                    if body == cup.body:
                        self.assertLess(height, flare_starts)
                    else:
                        self.assertGreater(height, flange_top + 0.05 * INCH)
                    pressed[side][body] += force
                for side, on in pressed.items():
                    self.assertGreater(on[cup.body], 10.0, side)
                    self.assertGreater(on[pin.body], 10.0, side)

    def test_the_recess_clears_the_ridge_where_a_stacks_pin_sits_on_its_cup(self):
        cup, pin = cad.cup(), cad.pin()
        cup_middle = -GEOMETRY.cup_middle_under_claw * INCH
        heights = np.linspace(-GEOMETRY.cup_band_top * INCH, GEOMETRY.pin_band_bottom * INCH, 400)
        widest = max(max(cup.radius_at(z - cup_middle), pin.radius_at(z)) for z in heights)
        recess_face = (GEOMETRY.cup_band_radius + GEOMETRY.recess_depth) * INCH
        self.assertGreater(recess_face - widest, 0.05 * INCH)

    def test_the_bands_press_only_where_the_pin_and_cup_taper_straight(self):
        cup, pin = cad.cup(), cad.pin()
        cup_band = np.linspace(GEOMETRY.cup_middle_under_claw - GEOMETRY.jaw_bottom,
                               GEOMETRY.cup_middle_under_claw - GEOMETRY.cup_band_top, 50) * INCH
        pin_band = np.linspace(GEOMETRY.pin_band_bottom, GEOMETRY.jaw_top, 50) * INCH
        for shape, band in ((cup, cup_band), (pin, pin_band)):
            radii = np.array([shape.radius_at(s) for s in band])
            slopes = np.diff(radii) / np.diff(band)
            self.assertLess(np.ptp(slopes), 0.02)

    def test_a_held_stack_tilts_under_2_5_degrees_in_a_full_voltage_spin(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        cup, pin = stack_in_the_claw(world)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.3)
        raise_to(world, hold, GEOMETRY.grip_lowest + 9.35)
        for unit in sim.chassis.left_units():
            for motor in unit.motors:
                motor.set_voltage(-12.0)
        for unit in sim.chassis.right_units():
            for motor in unit.motors:
                motor.set_voltage(12.0)
        worst = 0.0
        for seconds, volts in ((0.35, None), (0.6, 0.0)):
            if volts is not None:
                drive(sim, volts)
            for _ in range(round(seconds * 1000)):
                run(world, hold, 0.001)
                worst = max(worst, math.degrees(math.acos(min(1.0, abs(cup.axis()[2])))))
        self.assertLess(worst, 2.5)
        self.assertIs(world.grip.piece, cup)
        self.assertEqual(world.grip.carried, (pin,))

    def test_a_lone_cup_is_held_by_the_bottom_bands_alone(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        x, y, _ = claw_point(world)
        cup = world.pieces.add_cup((x, y, world.pieces.standing_height("cup")), of.DOWN)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIs(world.grip.piece, cup)
        cup_band_top = world.robot.grip_height() - GEOMETRY.cup_band_top * INCH
        for _, body, height, _ in jaw_contacts(world):
            self.assertLess(height, cup_band_top + 0.05 * INCH)
        raise_to(world, hold, 20.0)
        self.assertIs(world.grip.piece, cup)

    def test_the_jaws_push_a_stack_onto_their_middle_with_its_pin_still_standing_in_the_cup(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        cup, pin = stack_in_the_claw(world)
        for piece in (cup, pin):
            world.pieces.shift(piece, (0.0, 0.15 * INCH, 0.0))
        run(world, hold, 0.3)
        seat = pin.center()[2] - cup.center()[2]
        world.robot.claw.command(True)
        highest = seat
        for _ in range(400):
            run(world, hold, 0.001)
            highest = max(highest, pin.center()[2] - cup.center()[2])
        self.assertLess(claw_off(world, *cup.center()[:2]), 0.1 * INCH)
        self.assertEqual(world.grip.carried, (pin,))
        self.assertLess(highest - seat, 0.5 * INCH)
        self.assertAlmostEqual(pin.center()[2] - cup.center()[2], seat, delta=0.05 * INCH)

    def test_a_lone_pin_standing_on_the_floor_is_not_held(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        x, y, _ = claw_point(world)
        world.pieces.add_pin(("red", "yellow"), (x, y, world.pieces.standing_height("pin")), of.UP)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIsNone(world.grip)

    def test_a_lone_pin_hangs_by_its_flange_on_the_bottom_bands_squeezed_by_the_top_ones(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING, preload="yellow")
        run(world, hold, 0.2)
        pin = world.preload
        self.assertIs(world.grip.piece, pin)
        flange_top = pin.center()[2] + cascade.PIN_FLANGE_HALF_IN * INCH
        squeezed = sum(force for _, _, height, force in jaw_contacts(world) if height > flange_top)
        self.assertGreater(squeezed, 80.0)
        for lift_to in (GEOMETRY.grip_lowest, 20.0):
            raise_to(world, hold, lift_to)
            with self.subTest(lift_to=lift_to):
                self.assertIs(world.grip.piece, pin)
                flange_bottom = pin.center()[2] - cascade.PIN_FLANGE_HALF_IN * INCH
                cup_band_top = claw_middle(world) - GEOMETRY.cup_band_top * INCH
                self.assertGreater(flange_bottom - cup_band_top, -0.001 * INCH)
                self.assertLess(flange_bottom - cup_band_top, 0.03 * INCH)

    def test_a_stack_driven_onto_at_speed_stays_standing_under_the_claw(self):
        stack = (-23.55 * INCH, -23.55 * INCH)
        for volts in (1.5, 3.0, 6.0):
            with self.subTest(volts=volts):
                sim, world, hold = world_with(claw_at(*stack, 90.0, back_in=6.0), objects=True)
                cup = min(world.pieces.cups, key=lambda p: math.dist(p.center()[:2], stack))
                drive(sim, volts)
                run(world, hold, 0.4 + 2.0 / volts)
                drive(sim, 0.0)
                run(world, hold, 0.8)
                self.assertGreater(abs(cup.axis()[2]), 0.99)
                world.robot.claw.command(True)
                run(world, hold, 0.25)
                self.assertIs(world.grip.piece, cup)

    def test_a_stack_is_carried_onto_a_neutral_goal_and_scores(self):
        stack = (-23.55 * INCH, -23.55 * INCH)
        sim, world, hold = world_with(claw_at(*stack, 90.0, back_in=6.0), objects=True)
        turn_toggle(world, "left", 2.0 * math.pi * 2.0 / 3.0)
        before = world.score()["red"].total
        cup = min(world.pieces.cups, key=lambda p: math.dist(p.center()[:2], stack))
        drive(sim, 1.5)
        run(world, hold, 1.4)
        drive(sim, 0.0)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIs(world.grip.piece, cup)
        raise_to(world, hold, cascade.LIFT_HEIGHTS["over a neutral goal"])
        goal = of.goal("neutral left")
        carry_to(world, claw_at(goal.x, goal.y, 0.0, back_in=6.0))
        drive(sim, 4.0)
        run(world, hold, 1.4)
        drive(sim, 0.0)
        run(world, hold, 0.3)
        self.assertLess(claw_off(world, goal.x, goal.y), 0.3 * INCH)
        raise_to(world, hold, cascade.LIFT_HEIGHTS["on a neutral goal"], 0.5)
        world.robot.claw.command(False)
        run(world, hold, 0.3)
        raise_to(world, hold, cascade.LIFT_HEIGHTS["over a stack on a neutral goal"], 0.5)
        drive(sim, -3.0)
        run(world, hold, 0.6)
        drive(sim, 0.0)
        run(world, hold, 1.0)
        layers = next(s.layers for s in world.stacks() if s.goal == "neutral left")
        self.assertEqual(layers, [of.StackPin("yellow", "yellow"), of.StackCup(lower_opaque=True),
                                  of.StackPin("yellow", "yellow")])
        self.assertEqual(world.toggle_colors["left"], "red")
        self.assertEqual(world.score()["red"].total, before + of.YELLOW_PIN_POINTS)
        self.assertEqual(len(world.grips), 1)
        self.assertEqual([d["reason"] for d in world.drops], ["the claw opened"])

    def test_a_held_stack_rammed_into_a_goal_at_8_v_is_taken_once_and_let_go_once(self):
        goal = of.goal("red left")
        sim, world, hold = world_with(claw_at(goal.x, goal.y, 0.0, back_in=9.0))
        cup, pin = stack_in_the_claw(world)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        drive(sim, 8.0)
        run(world, hold, 1.5)
        self.assertLess(abs(sim.state.speed), 1.0 * INCH)
        self.assertIs(world.grip.piece, cup)
        self.assertEqual(world.grip.carried, (pin,))
        drive(sim, 0.0)
        world.robot.claw.command(False)
        run(world, hold, 0.3)
        self.assertEqual(len(world.grips), 1)
        self.assertEqual([d["reason"] for d in world.drops], ["the claw opened"])

    def test_an_open_claw_takes_nothing_its_jaws_press(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        cup, _ = stack_in_the_claw(world)
        with unittest.mock.patch.object(world, "_gripped", return_value=cup):
            run(world, hold, 0.1)
        self.assertIsNone(world.grip)
        self.assertEqual(world.grips, [])

    def test_jaws_opening_off_the_cup_before_the_pin_let_go_of_the_whole_stack(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        cup, pin = stack_in_the_claw(world)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIs(world.grip.piece, cup)
        world.robot.claw.command(False)
        with unittest.mock.patch.object(world, "_gripped", return_value=pin):
            run(world, hold, 0.1)
        self.assertIsNone(world.grip)
        self.assertEqual(len(world.grips), 1)
        self.assertEqual([(d["piece"], d["stacked"], d["reason"]) for d in world.drops],
                         [(cup.id, [pin.id], "the claw opened")])


class TestClawAir(unittest.TestCase):
    def test_each_change_fills_one_side_of_the_piston_from_the_tank(self):
        claw = cascade.PneumaticClaw(GEOMETRY)
        claw.command(True)
        closed = 100.0 * 200.0 / (200.0 + claw.closing_ml)
        self.assertAlmostEqual(claw.air_psi, closed)
        claw.command(True)
        self.assertAlmostEqual(claw.air_psi, closed)
        claw.command(False)
        self.assertAlmostEqual(claw.air_psi, closed * 200.0 / (200.0 + claw.opening_ml))
        self.assertAlmostEqual(claw.closing_ml, 2.83 + 1.0, delta=0.01)
        self.assertAlmostEqual(claw.opening_ml, 2.51 + 1.0, delta=0.01)

    def test_the_grip_weakens_as_the_air_runs_down(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        claw = world.robot.claw
        squeezes = []
        for _ in range(2):
            cup, pin = stack_in_the_claw(world)
            run(world, hold, 0.3)
            claw.command(True)
            run(world, hold, 0.25)
            squeezes.append((world.mechanism.squeeze, claw.air_psi))
            claw.command(False)
            run(world, hold, 0.3)
            for piece in (cup, pin):
                world.pieces.shift(piece, (0.0, 30.0 * INCH, 0.0))
            for _ in range(20):
                claw.command(not claw.closed)
        (first, first_psi), (later, later_psi) = squeezes
        self.assertLess(later, 0.85 * first)
        self.assertAlmostEqual(later / first, later_psi / first_psi, delta=0.08)

    def test_the_older_kits_numbers_give_the_strokes_vex_quotes_for_them(self):
        older_kit = dataclasses.replace(GEOMETRY, air_tank_ml=150.0, piston_bore_mm=10.0, piston_travel_mm=50.8)
        claw = cascade.PneumaticClaw(older_kit)
        strokes = 0
        while claw.air_psi >= 25.0:
            claw.command(not claw.closed)
            strokes += 1
        self.assertAlmostEqual(strokes, 45, delta=2)

    def test_the_cascade_claw_strokes_77_times_before_its_tank_is_down_to_25_psi(self):
        claw = cascade.PneumaticClaw(GEOMETRY)
        strokes = 0
        while claw.air_psi >= 25.0:
            claw.command(not claw.closed)
            strokes += 1
        self.assertEqual(strokes, 77)


def seat_on(goal_name, height, side_in):
    """Drive the bite onto a Goal from `side_in` to its left: (how far the
    Goal's middle ends ahead of the claw's, and to its left, inches)."""
    goal = of.goal(goal_name)
    sim, world, hold = world_with(claw_at(goal.x, goal.y, 0.0, back_in=8.0, side_in=side_in))
    raise_to(world, hold, cascade.LIFT_HEIGHTS[height])
    drive(sim, 4.0)
    run(world, hold, 1.4)
    drive(sim, 0.0)
    run(world, hold, 0.3)
    if (f"the {goal_name} Goal", "drivetrain") not in {(c.element, c.part) for c in world.reported_contacts()}:
        raise AssertionError(f"the bite never pressed on the {goal_name} Goal")
    x, y, _ = claw_point(world)
    c, s = math.cos(sim.state.theta), math.sin(sim.state.theta)
    dx, dy = goal.x - x, goal.y - y
    return (dx * c + dy * s) / INCH, (-dx * s + dy * c) / INCH


class TestBite(unittest.TestCase):
    def test_it_seats_a_neutral_goal_or_the_center_goal_under_the_claw_from_off_to_the_side(self):
        for goal_name, height in (("neutral bottom", "over a neutral goal"), ("center", "over the center goal")):
            for offset in (-1.5, 0.75):
                with self.subTest(goal=goal_name, offset=offset):
                    ahead, left = seat_on(goal_name, height, offset)
                    self.assertLess(math.hypot(ahead, left), 0.1)

    def test_on_an_alliance_goal_the_frame_meets_the_taper_0_1_to_0_4_inches_short(self):
        for offset in (0.0, -1.5, 0.75):
            with self.subTest(offset=offset):
                ahead, left = seat_on("red left", "over a neutral goal", offset)
                self.assertGreater(ahead, -0.4)
                self.assertLess(ahead, -0.1)
                self.assertLess(abs(left), 0.25)


def pin_seat(goal):
    """How high a Pin standing in `goal` has its middle."""
    return goal.riser + cad.seats().pin_in_goal


def lowest_point(piece):
    return min(center[2] - radius for center, radius in piece.spheres())


def tilt_deg(piece):
    return math.degrees(math.acos(min(1.0, abs(piece.axis()[2]))))


def seated_with_the_claw_over(goal_name, preload=None, pin_in_goal=False):
    """Drive the bite onto a Goal with the claw raised over a neutral Goal
    (over the centre Goal for that one), a Pin standing in the Goal if
    `pin_in_goal`: (sim, world, hold, the Goal, the lowest the preload came on
    the way)."""
    goal = of.goal(goal_name)
    sim, world, hold = world_with(claw_at(goal.x, goal.y, 0.0, back_in=8.0), preload=preload)
    if pin_in_goal:
        world.pieces.add_pin(("red", "yellow"), (goal.x, goal.y, pin_seat(goal)), of.UP)
    raise_to(world, hold, cascade.LIFT_HEIGHTS["over the center goal" if goal.kind == "center" else "over a neutral goal"])
    lowest = math.inf
    drive(sim, 4.0)
    for _ in range(1400):
        run(world, hold, 0.001)
        if preload is not None:
            lowest = min(lowest, lowest_point(world.preload))
    drive(sim, 0.0)
    run(world, hold, 0.3)
    return sim, world, hold, goal, lowest


def elements_touching(world, piece):
    """The field elements `piece` touches now."""
    m, d = world.model, world.data
    touching = set()
    for first, second in d.contact.geom[:d.ncon]:
        if m.geom_bodyid[first] == piece.body and second in world.pieces.elements:
            touching.add(world.pieces.elements[second])
        if m.geom_bodyid[second] == piece.body and first in world.pieces.elements:
            touching.add(world.pieces.elements[first])
    return touching


class TestPreloadIntoGoalsKnownLimitation(unittest.TestCase):
    """Known limitation, open with the robot's owner: the preload, hung by
    its flange on the bottom bands, cannot be lowered onto a Goal's seat.
    With the claw at its lowest it hangs 0.53 in over where a Pin sits in a
    neutral Goal and 3.06 in over an alliance Goal's, so it has to be let go.
    Its 2.35 in cone has 0.01 in a side in the Goal's 2.37 in hole: over the
    centre Goal its tip lands on the rim unless the claw is within 0.01 in of
    the middle.  Let go from the lowest over an alliance Goal, it seated 1
    time in 4 on 2026-11-05 (the claw put 0, 0.1, 0.3 and 0.5 in off the
    middle each way; only 0.3 seated).  These tests hold what happens now, so
    a change to it shows."""

    def test_at_the_claws_lowest_the_preload_hangs_0_53_in_over_a_neutral_seat_and_3_06_in_over_an_alliance_seat(self):
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING, preload="yellow")
        run(world, hold, 0.5)
        middle = world.preload.center()[2]
        self.assertAlmostEqual((middle - pin_seat(of.goal("neutral left"))) / INCH, 0.53, delta=0.03)
        self.assertAlmostEqual((middle - pin_seat(of.goal("red left"))) / INCH, 3.06, delta=0.03)
        self.assertAlmostEqual((of.GOAL_HOLE_DIAMETER - of.PIN_CONE_DIAMETER) / INCH, 0.02, places=6)

    def test_lowered_over_a_neutral_goal_and_let_go_the_preload_does_not_stay_in_it(self):
        sim, world, hold, goal, _ = seated_with_the_claw_over("neutral left", preload="yellow")
        raise_to(world, hold, GEOMETRY.grip_lowest, 0.8)
        self.assertGreater(world.preload.center()[2] - pin_seat(goal), 0.5 * INCH)
        world.robot.claw.command(False)
        run(world, hold, 1.0)
        raise_to(world, hold, cascade.LIFT_HEIGHTS["over a neutral goal"], 0.8)
        drive(sim, -3.0)
        run(world, hold, 0.6)
        drive(sim, 0.0)
        run(world, hold, 1.0)
        self.assertIsNone(world.preload.below)
        self.assertEqual([s for s in world.stacks() if s.goal == goal.name], [])

    def test_lowered_over_the_centre_goal_the_preloads_tip_lands_on_the_rim_and_it_leans_off_the_hole(self):
        sim, world, hold, goal, _ = seated_with_the_claw_over("center", preload="yellow")
        self.assertGreater(claw_off(world, goal.x, goal.y), 0.01 * INCH)
        raise_to(world, hold, GEOMETRY.grip_lowest, 0.8)
        self.assertIn("the center Goal", elements_touching(world, world.preload))
        self.assertGreater(math.dist(world.preload.center()[:2], (goal.x, goal.y)), 3.0 * INCH)
        self.assertGreater(tilt_deg(world.preload), 10.0)
        self.assertIs(world.grip.piece, world.preload)


class TestGoalsAndPins(unittest.TestCase):
    def test_a_raised_claw_carries_the_preload_over_an_alliance_goal_as_the_bite_takes_it(self):
        sim, world, hold, goal, lowest = seated_with_the_claw_over("red left", preload="yellow")
        self.assertEqual({c.part for c in world.reported_contacts() if c.element == "the red left Goal"},
                         {"drivetrain"})
        self.assertGreater(lowest - goal.height, 8.0 * INCH)
        self.assertLess(claw_off(world, goal.x, goal.y), 0.4 * INCH)
        self.assertIs(world.grip.piece, world.preload)

    def test_closing_on_a_pin_standing_in_a_neutral_goal_the_bottom_bands_grip_its_flange_and_lift_it_out(self):
        sim, world, hold, goal, _ = seated_with_the_claw_over("neutral left", pin_in_goal=True)
        pin = next(p for p in world.pieces.pins)
        raise_to(world, hold, GEOMETRY.grip_lowest, 0.8)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.4)
        on_the_pin = [(side, height, force) for side, body, height, force in jaw_contacts(world) if body == pin.body]
        self.assertEqual({side for side, _, _ in on_the_pin}, {"left", "right"})
        for side, height, force in on_the_pin:
            self.assertAlmostEqual(height, pin.center()[2], delta=0.05 * INCH)
            self.assertGreater(force, 40.0)
        self.assertIs(world.grip.piece, pin)
        raise_to(world, hold, world.robot.grip_height() / INCH + 3.0, 0.8)
        self.assertGreater(pin.center()[2] - pin_seat(goal), 2.5 * INCH)
        self.assertIsNone(world.pieces.below(pin))

    def test_closing_on_a_pin_standing_in_an_alliance_goal_the_jaws_pass_over_its_tip_and_touch_nothing(self):
        sim, world, hold, goal, _ = seated_with_the_claw_over("red left", pin_in_goal=True)
        pin = next(p for p in world.pieces.pins)
        raise_to(world, hold, GEOMETRY.grip_lowest, 0.8)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.4)
        self.assertEqual(jaw_contacts(world), [])
        self.assertAlmostEqual(world.mechanism.gap / INCH, GEOMETRY.jaw_closed_gap, delta=0.02)
        self.assertIsNone(world.grip)
        self.assertAlmostEqual(pin.center()[2], pin_seat(goal), delta=0.01 * INCH)


def backed_to_toggle(quadrant, gap_in):
    """A robot with its back `gap_in` from the wall in the middle of the `quadrant` Toggle."""
    mount = next(m for m in of.TOGGLES if m.quadrant == quadrant)
    out = of.HALF - HALF_LENGTH - gap_in * INCH
    sim, world, hold = world_with((-mount.inward_x * out, -mount.inward_y * out,
                                   math.atan2(mount.inward_y, mount.inward_x)))
    return sim, world, hold, toggle_at(world, quadrant)


def facing_toggle(quadrant, gap_in):
    """A robot with its front `gap_in` from the wall in the middle of the `quadrant` Toggle."""
    mount = next(m for m in of.TOGGLES if m.quadrant == quadrant)
    out = of.HALF - HALF_LENGTH - gap_in * INCH
    sim, world, hold = world_with((-mount.inward_x * out, -mount.inward_y * out,
                                   math.atan2(-mount.inward_y, -mount.inward_x)))
    return sim, world, hold, toggle_at(world, quadrant)


def onto_the_wall_at(quadrant):
    """A robot backed onto the wall at the middle of the `quadrant` Toggle with
    its flaps upright, then held there with its drive at 2 V."""
    sim, world, hold, toggle = backed_to_toggle(quadrant, 1.0)
    world.data.qpos[world.mechanism.roller.at] = math.pi / 2.0
    drive(sim, -3.0)
    for _ in range(1500):
        run(world, hold, 0.001)
        if world.touching_perimeter:
            break
    if not world.touching_perimeter:
        raise AssertionError("the robot did not reach the wall")
    drive(sim, -2.0)
    run(world, hold, 0.2)
    return sim, world, hold, toggle


def roll(world, hold, toggle, volts, seconds):
    """Run the roller at `volts`: each new face the Toggle settles on, as
    (seconds from the start, faces turned toward the wall, the colour then)."""
    face_turn = 2.0 * math.pi / 3.0
    world.mechanism.roller.motor.set_voltage(volts)
    settled, last = [], 0
    for ms in range(1, round(seconds * 1000) + 1):
        run(world, hold, 0.001)
        face = round(toggle.angle / face_turn)
        if face != last and abs(toggle.angle - face * face_turn) < of.TOGGLE_SEATED_TOLERANCE:
            settled.append((ms / 1000.0, face, world.toggle_colors[toggle.mount.quadrant]))
            last = face
    return settled


def gap_to_wall(sim):
    return of.HALF - abs(sim.state.x) - HALF_LENGTH


class TestRoller(unittest.TestCase):
    def test_the_flap_tips_reach_0_75_in_behind_the_frame_and_pass_over_the_wall(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        flaps = [part for part in world.robot.parts(sim.chassis) if part.name == "roller flap"]
        self.assertEqual(len(flaps), 4)
        self.assertAlmostEqual(-HALF_LENGTH - min(flap.back for flap in flaps), 0.75 * INCH, delta=0.001 * INCH)
        g = GEOMETRY
        lowest_over_the_wall = g.roller_axle_height - math.sqrt(g.flap_tip_reach ** 2 - g.roller_axle_in_from_back ** 2)
        self.assertAlmostEqual(lowest_over_the_wall - of.WALL_HEIGHT / INCH, 0.17, delta=0.01)

    def test_at_12_v_each_flap_turns_the_toggle_one_face_top_toward_the_wall(self):
        for quadrant, first, second in (("left", "blue", "red"), ("right", "red", "blue")):
            with self.subTest(quadrant=quadrant):
                sim, world, hold, toggle = onto_the_wall_at(quadrant)
                settled = roll(world, hold, toggle, 12.0, 1.6)
                self.assertEqual([(face, color) for _, face, color in settled],
                                 [(1, first), (2, second), (3, "yellow")])
                self.assertLess(settled[0][0], 0.65)
                self.assertLess(settled[2][0], 1.6)

    def test_run_backward_the_flaps_press_the_toggle_onto_its_plates_and_it_stays(self):
        sim, world, hold, toggle = onto_the_wall_at("left")
        most = 0.0
        world.mechanism.roller.motor.set_voltage(-12.0)
        for _ in range(1500):
            run(world, hold, 0.001)
            most = max(most, abs(toggle.angle))
        self.assertLess(most, math.radians(3.0))
        self.assertLess(world.mechanism.roller.angle, -math.radians(360.0))

    def test_with_the_drive_off_the_first_flap_pushes_the_robot_off_the_wall_out_of_reach(self):
        sim, world, hold, toggle = onto_the_wall_at("left")
        drive(sim, 0.0)
        settled = roll(world, hold, toggle, 12.0, 2.0)
        self.assertEqual([(face, color) for _, face, color in settled], [(1, "blue")])
        self.assertGreater(gap_to_wall(sim), 0.5 * INCH)

    def test_backing_in_at_full_speed_with_the_roller_held_level_turns_the_bar_under_20_degrees_and_it_falls_back(self):
        sim, world, hold, toggle = backed_to_toggle("left", 24.0)
        world.mechanism.roller.motor.stop(BrakeMode.HOLD)
        drive(sim, -12.0)
        most, fastest = 0.0, 0.0
        for _ in range(1100):
            run(world, hold, 0.001)
            most = max(most, abs(toggle.angle))
            fastest = max(fastest, abs(sim.state.u))
        drive(sim, 0.0)
        run(world, hold, 1.0)
        self.assertGreater(fastest, 60.0 * INCH)
        self.assertGreater(most, math.radians(5.0))
        self.assertLess(most, math.radians(20.0))
        self.assertLess(abs(toggle.angle), math.radians(3.0))
        self.assertLess(abs(world.mechanism.roller.angle), math.radians(5.0))

    def test_a_held_stack_driven_at_a_toggle_below_level_with_or_above_its_axle_never_touches_it(self):
        for above_axle in (-1.0, 0.0, 1.0):
            with self.subTest(above_axle=above_axle):
                sim, world, hold, toggle = facing_toggle("left", 3.0)
                cup, pin = stack_in_the_claw(world)
                run(world, hold, 0.3)
                world.robot.claw.command(True)
                run(world, hold, 0.25)
                raise_to(world, hold, of.TOGGLE_AXLE_HEIGHT / INCH + above_axle)
                drive(sim, 6.0)
                most = 0.0
                touched = set()
                for _ in range(1200):
                    run(world, hold, 0.001)
                    most = max(most, abs(toggle.angle))
                    touched |= {c.element for c in world.contacts}
                self.assertTrue(world.touching_perimeter)
                self.assertNotIn("the left Toggle", touched)
                self.assertLess(most, math.radians(0.5))
                self.assertIs(world.grip.piece, cup)

    def test_with_its_front_on_the_wall_at_the_toggle_nothing_on_the_robot_touches_it(self):
        sim, world, hold, toggle = facing_toggle("left", 1.0)
        drive(sim, 4.0)
        run(world, hold, 1.0)
        drive(sim, 0.0)
        run(world, hold, 0.5)
        self.assertLess(abs(toggle.angle), math.radians(0.5))
        self.assertNotIn("the left Toggle", {c.element for c in world.contacts})


FIT_TOLERANCE = 0.001 * INCH


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return None if length < 1e-12 else tuple(c / length for c in v)


def _faces_and_edges(part):
    """A part's face normals and edge directions, robot frame."""
    if part.profile:
        sides = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(part.profile, part.profile[1:] + part.profile[:1])]
        return [(dz, 0.0, -dx) for dx, dz in sides] + [(0.0, 1.0, 0.0)], [(dx, 0.0, dz) for dx, dz in sides] + [
            (0.0, 1.0, 0.0)]
    corners = part.corners()
    sides = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(corners, corners[1:] + corners[:1])]
    return [(dy, -dx, 0.0) for dx, dy in sides] + [(0.0, 0.0, 1.0)], [(dx, dy, 0.0) for dx, dy in sides] + [
        (0.0, 0.0, 1.0)]


def _span(points, axis):
    values = [sum(p[k] * axis[k] for k in range(3)) for p in points]
    return min(values), max(values)


def least_overlap(a, b):
    """(how far two parts' solids overlap along the axis they overlap least on, that axis): negative is a gap."""
    faces_a, edges_a = _faces_and_edges(a)
    faces_b, edges_b = _faces_and_edges(b)
    axes = [_unit(v) for v in faces_a + faces_b + [_cross(u, w) for u in edges_a for w in edges_b]]
    points_a, points_b = a.vertices(), b.vertices()
    best = None
    for axis in axes:
        if axis is None:
            continue
        low_a, high_a = _span(points_a, axis)
        low_b, high_b = _span(points_b, axis)
        overlap = min(high_a, high_b) - max(low_a, low_b)
        if best is None or overlap < best[0]:
            best = (overlap, axis)
    return best


def touching_width(a, b, axis):
    """How wide the faces of two touching parts that meet across `axis` overlap: 0 for an edge or a corner."""
    plane = max(_span(a.vertices(), axis)[0], _span(b.vertices(), axis)[0])
    side = 1.0 if abs(_span(a.vertices(), axis)[1] - plane) < abs(_span(a.vertices(), axis)[0] - plane) else -1.0
    face_a = [p for p in a.vertices() if abs(sum(p[k] * axis[k] for k in range(3)) - _span(a.vertices(), axis)[
        1 if side > 0 else 0]) < FIT_TOLERANCE]
    face_b = [p for p in b.vertices() if abs(sum(p[k] * axis[k] for k in range(3)) - _span(b.vertices(), axis)[
        0 if side > 0 else 1]) < FIT_TOLERANCE]
    across = _unit(_cross(axis, (1.0, 0.0, 0.0))) or _unit(_cross(axis, (0.0, 1.0, 0.0)))
    along = _cross(axis, across)
    widths = []
    for direction in (across, along, _unit(tuple(x + y for x, y in zip(across, along))),
                      _unit(tuple(x - y for x, y in zip(across, along)))):
        low_a, high_a = _span(face_a, direction)
        low_b, high_b = _span(face_b, direction)
        widths.append(min(high_a, high_b) - max(low_a, low_b))
    return min(widths)


def posed_pieces(lift_up, claw_closed, roller_turn):
    robot = cascade.CascadeRobot()
    rise = GEOMETRY.lift_travel * INCH if lift_up else 0.0
    closing = GEOMETRY.jaw_travel * INCH if claw_closed else 0.0
    components = robot.components(CHASSIS)
    turns = {hinge.name: 0.0 for c in components for hinge in c.hinges}
    turns[cascade.ROLLER] = roller_turn
    return {c.key: (c, c.posed(c.rise_share * rise, closing, 0.0, turns)) for c in components}


POSES = {"lift down, claw open": (False, False, 0.0), "lift up, claw open": (True, False, 0.0),
         "lift up, claw closed": (True, True, 0.0), "lift down, claw closed": (False, True, 0.0),
         "lift down, roller turned a quarter": (False, False, math.pi / 2.0)}


class TestPiecesFitTogether(unittest.TestCase):
    def test_every_piece_but_the_frame_hangs_from_the_frame_through_the_pieces_it_is_on(self):
        pieces = posed_pieces(False, False, 0.0)
        for key, (component, _) in pieces.items():
            with self.subTest(piece=key):
                seen, frontier = {key}, list(component.parents)
                while frontier:
                    parent = frontier.pop()
                    self.assertIn(parent, pieces)
                    if parent not in seen:
                        seen.add(parent)
                        frontier += pieces[parent][0].parents
                self.assertIn(cascade.FRAME_BLOCK, seen | {key})

    def test_each_piece_touches_the_pieces_it_is_on_face_to_face(self):
        for pose, (lift_up, claw_closed, roller_turn) in POSES.items():
            pieces = posed_pieces(lift_up, claw_closed, roller_turn)
            for key, (component, part) in pieces.items():
                for parent in component.parents:
                    with self.subTest(pose=pose, piece=key, on=parent):
                        depth, axis = least_overlap(part, pieces[parent][1])
                        self.assertLess(abs(depth), FIT_TOLERANCE)
                        self.assertGreater(touching_width(part, pieces[parent][1], axis), 0.1 * INCH)

    def test_no_two_pieces_overlap(self):
        for pose, (lift_up, claw_closed, roller_turn) in POSES.items():
            parts = [(key, part) for key, (_, part) in posed_pieces(lift_up, claw_closed, roller_turn).items()]
            for i, (key_a, a) in enumerate(parts):
                for key_b, b in parts[i + 1:]:
                    with self.subTest(pose=pose, pieces=(key_a, key_b)):
                        self.assertLess(least_overlap(a, b)[0], FIT_TOLERANCE)

    def test_the_bumpers_follow_the_frame_outline_into_the_bite(self):
        robot = cascade.CascadeRobot()
        outline = robot.frame_outline(CHASSIS)
        frame_pieces = [c.part for c in robot.components(CHASSIS) if c.part.name == "drivetrain"]
        self.assertAlmostEqual(polygon_area(outline), sum(polygon_area(part.corners()) for part in frame_pieces),
                               places=12)
        inside = robot.bumper_inside(CHASSIS)
        self.assertEqual(len(inside), len(outline))
        bumper = GEOMETRY.bumper_thickness * INCH
        for (xa, ya), (xb, yb), (xc, yc), (xd, yd) in zip(outline, outline[1:] + outline[:1], inside,
                                                            inside[1:] + inside[:1]):
            length = math.hypot(xb - xa, yb - ya)
            inward = (-(yb - ya) / length, (xb - xa) / length)
            for x, y in ((xc, yc), (xd, yd)):
                self.assertAlmostEqual((x - xa) * inward[0] + (y - ya) * inward[1], bumper, places=12)


def polygon_area(points):
    return 0.5 * sum(xa * yb - xb * ya for (xa, ya), (xb, yb) in zip(points, points[1:] + points[:1]))


class TestMatchLoads(unittest.TestCase):
    def test_the_claw_takes_a_match_load_from_under_the_raised_loader(self):
        loader = of.loader("red right")
        sim, world, hold = world_with(claw_at(loader.x, loader.y, 180.0, back_in=8.0), objects=True)
        cup = world.match_load("red right")
        self.assertEqual(world.raised_loaders, ["red right"])
        self.assertEqual(world.match_loads_left, {"cup": 9, ("red", "yellow"): 9, ("yellow", "yellow"): 1})
        drive(sim, 3.0)
        run(world, hold, 1.2)
        self.assertTrue(world.touching_perimeter)
        drive(sim, 0.0)
        run(world, hold, 0.3)
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        self.assertIs(world.grip.piece, cup)
        drive(sim, -4.0)
        run(world, hold, 0.6)
        drive(sim, 0.0)
        raise_to(world, hold, cascade.LIFT_HEIGHTS["over a neutral goal"])
        self.assertIs(world.grip.piece, cup)
        self.assertEqual([p.colors for p in world.grip.carried], [("red", "yellow")])
        world.lower_loader("red right")
        self.assertEqual(world.raised_loaders, [])

    def test_each_alliance_has_one_yellow_pin_to_load(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        world.match_load("red right", ("yellow", "yellow"))
        with self.assertRaisesRegex(ValueError, "no yellow/yellow Pins left"):
            world.match_load("red left", ("yellow", "yellow"))

    def test_the_open_claw_can_wait_with_its_middle_2_8_inches_from_where_a_match_load_will_stand(self):
        loader = of.loader("red left")
        sim, world, _ = world_with(claw_at(loader.x, loader.y, 180.0, back_in=2.8))
        world.match_load("red left")
        self.assertEqual(world.match_loads_left["cup"], 9)

    def test_nothing_is_loaded_where_the_closed_claw_stands(self):
        loader = of.loader("red left")
        sim, world, hold = world_with(CLEAR_OF_EVERYTHING)
        world.match_load("red left")
        for piece in world.pieces.pieces():
            world.pieces.shift(piece, (30.0 * INCH, 0.0, 0.0))
        world.robot.claw.command(True)
        run(world, hold, 0.25)
        carry_to(world, claw_at(loader.x, loader.y, 180.0))
        with self.assertRaisesRegex(ValueError, "the robot is where the Cup would stand under the red left Loader"):
            world.match_load("red left")
        self.assertEqual(world.match_loads_left["cup"], 9)

    def test_a_loader_does_not_come_down_on_a_stack(self):
        sim, world, _ = world_with(CLEAR_OF_EVERYTHING)
        world.match_load("red left")
        with self.assertRaisesRegex(ValueError, "under the red left Loader"):
            world.lower_loader("red left")
        with self.assertRaisesRegex(ValueError, "already stands under the red left Loader"):
            world.match_load("red left")


if __name__ == "__main__":
    unittest.main()

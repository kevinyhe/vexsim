"""A robot rocks up onto one edge of its wheels when it brakes, speeds up or
turns harder than the height of its CG allows, and falls over once it rocks
past the point where its CG is over that edge."""

import math
import os
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vexsim import Simulator, cascade, presets
from vexsim.dynamics import Tilt
from vexsim.mujoco_drive import wheel_body_name
from vexsim.override import OverrideWorld
from vexsim.sim import SimConfig
from vexsim.units import INCH


def full_speed_then_reverse(chassis, seconds, watch=lambda sim: None):
    sim = Simulator(chassis, config=SimConfig(log_every=0))
    for motor in sim.motors:
        motor.set_voltage(12.0)
    for _ in range(1500):
        sim.step(0.001)
    for motor in sim.motors:
        motor.set_voltage(-12.0)
    for _ in range(round(seconds * 1000)):
        sim.step(0.001)
        watch(sim)
    return sim


def with_cg_at(height_in, preset="four_motor_450"):
    chassis = presets.get(preset)
    chassis.cg_height = height_in * INCH
    return chassis


def front_edge(sim):
    return max(sim.dynamics.edges, key=lambda edge: edge.nx)


class TestTipping(unittest.TestCase):
    def test_a_robot_with_its_cg_5_in_up_keeps_every_wheel_down_braking_from_full_speed(self):
        sim = full_speed_then_reverse(with_cg_at(5.0), 0.6)
        self.assertEqual(sim.dynamics.tilt.most, 0.0)

    def test_a_tall_robot_braking_hard_falls_forward_and_then_stays_where_it_fell(self):
        sim = full_speed_then_reverse(presets.tall_bot(), 0.6)
        tilt = sim.dynamics.tilt
        self.assertTrue(tilt.fallen)
        self.assertEqual(tilt.edge, front_edge(sim))
        self.assertAlmostEqual(tilt.balance, math.atan2(tilt.edge.reach, sim.chassis.cg_height))
        self.assertGreaterEqual(tilt.angle, tilt.balance)
        resting = (sim.state.x, sim.state.y)
        for _ in range(200):
            sim.step(0.001)
        self.assertEqual((sim.state.x, sim.state.y), resting)

    def test_a_robot_just_past_its_limit_rocks_onto_its_front_wheels_and_lands_again(self):
        rear_loads, rocked = [], []

        def watch(sim):
            rocked.append(sim.dynamics.tilt.edge is not None)
            rocked_through_the_whole_step = rocked[-2:] == [True, True]
            if rocked_through_the_whole_step:
                rear_loads.extend(unit.fz for unit in sim.chassis.wheel_units if unit.x < 0.0)

        sim = full_speed_then_reverse(with_cg_at(12.0), 0.3, watch)
        self.assertGreater(math.degrees(sim.dynamics.tilt.most), 0.05)
        self.assertFalse(sim.dynamics.tilt.fallen)
        self.assertEqual(rocked[-1], False)
        self.assertIn(True, rocked)
        self.assertEqual(max(rear_loads), 0.0)

    def test_rocked_up_the_front_wheels_carry_all_the_weight(self):
        loads = []

        def watch(sim):
            if sim.dynamics.tilt.edge is not None:
                loads.append(sum(unit.fz for unit in sim.chassis.wheel_units))

        sim = full_speed_then_reverse(with_cg_at(12.0), 0.3, watch)
        self.assertTrue(loads)
        for total in loads:
            self.assertAlmostEqual(total, sim.chassis.weight, places=9)

    def test_a_rocking_robot_falls_over_only_with_the_energy_to_lift_its_cg_over_the_edge(self):
        chassis = with_cg_at(9.0)
        sim = Simulator(chassis, config=SimConfig(log_every=0))
        for edge in (front_edge(sim), max(sim.dynamics.edges, key=lambda edge: edge.ny)):
            h, d = chassis.cg_height, edge.reach
            inertia = chassis.level_inertia(edge.nx, edge.ny) + chassis.mass * (d * d + h * h)
            climb = chassis.mass * chassis.gravity * (math.hypot(d, h) - h)
            just_enough = math.sqrt(2.0 * climb / inertia)
            for rate, falls in ((1.02 * just_enough, True), (0.98 * just_enough, False)):
                with self.subTest(edge=(edge.nx, edge.ny), rate=rate):
                    sim.dynamics.reset()
                    sim.dynamics.tilt = Tilt(edge=edge, angle=1e-9, rate=rate)
                    for _ in range(3000):
                        sim.dynamics.step(0.0005)
                    self.assertEqual(sim.dynamics.tilt.fallen, falls)


def tip_toward(world, toward, degrees):
    """Stand the robot still, leaning `degrees` toward `toward` (a level
    direction in the chassis' frame) about its outermost axles that way,
    and then lowered onto whatever of it is lowest."""
    drive, m, d = world.drive, world.model, world.data
    q = drive.free_q
    ahead, beside = toward
    reach = max(unit.x * ahead + unit.y * beside for unit in world.sim.chassis.wheel_units)
    radius = world.sim.chassis.wheel_units[0].radius
    turn = d.xmat[drive.body].reshape(3, 3)
    pivot = d.xpos[drive.body] + turn @ np.array([reach * ahead, reach * beside, 0.0]) + np.array([0.0, 0.0, radius])
    tilt = np.empty(4)
    mujoco.mju_axisAngle2Quat(tilt, turn @ np.array([-beside, ahead, 0.0]), math.radians(degrees))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, tilt)
    d.qpos[q:q + 3] = pivot + rotation.reshape(3, 3) @ (d.qpos[q:q + 3] - pivot)
    turned = np.empty(4)
    mujoco.mju_mulQuat(turned, tilt, d.qpos[q + 3:q + 7].copy())
    d.qpos[q + 3:q + 7] = turned
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    floor = next(g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    robot = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == drive.body and m.geom_contype[g]]
    fromto = np.zeros(6)
    d.qpos[q + 2] -= min(0.0, min(mujoco.mj_geomDistance(m, d, g, floor, 0.05, fromto) for g in robot))
    mujoco.mj_forward(m, d)


def lean_deg(world):
    return math.degrees(math.acos(min(1.0, world.data.xmat[world.drive.body][8])))


class TestCascadeOnMujoco(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(presets.override_cascade(), config=SimConfig(log_every=0))
        self.sim.set_pose(-55.0 * INCH, -12.0 * INCH, 0.0)
        self.world = OverrideWorld(self.sim, robot=cascade.CascadeRobot(), objects=False)
        self.lift = self.world.mechanism.lift

    def run_for(self, seconds):
        for _ in range(round(seconds * 1000)):
            self.world.step(0.001)

    def raise_lift_fully(self):
        for motor in self.lift.motors:
            motor.set_voltage(12.0)
        self.run_for(1.5)

    def test_it_weighs_the_owners_6_35_kg_with_its_cg_3_5_in_up(self):
        self.assertAlmostEqual(self.world.drive.robot_mass, 6.35, delta=0.001)
        self.assertAlmostEqual(self.world.center_of_mass()[2] / INCH, 3.5, delta=0.03)

    def test_the_lift_fully_up_raises_the_cg_by_each_stage_mass_times_its_rise(self):
        down = self.world.center_of_mass()[2]
        self.raise_lift_fully()
        g = cascade.GEOMETRY
        risen = sum(kg * rise for kg, rise in zip(cascade.stage_masses(g), self.lift.stage_rises()))
        self.assertAlmostEqual(self.lift.extension / INCH, g.stage_travel, delta=0.02)
        self.assertAlmostEqual((self.world.center_of_mass()[2] - down) / INCH,
                               risen / self.world.drive.robot_mass / INCH, delta=0.02)

    def test_it_falls_over_its_nose_where_its_cg_passes_the_front_of_the_frame(self):
        for lift_up in (False, True):
            with self.subTest(lift_up=lift_up):
                self.setUp()
                if lift_up:
                    self.raise_lift_fully()
                    for motor in self.lift.motors:
                        motor.set_voltage(1.5)
                    self.run_for(0.3)
                at_rest = self.world.data.qpos.copy()
                limit = math.degrees(self.world.drive.fall_angle((1.0, 0.0)))
                self.assertAlmostEqual(limit, 46.9 if lift_up else 69.4, delta=0.3)
                for past, falls in ((-1.0, False), (1.0, True)):
                    self.world.data.qpos[:] = at_rest
                    tip_toward(self.world, (1.0, 0.0), limit + past)
                    self.run_for(1.5)
                    self.assertEqual(lean_deg(self.world) > limit, falls)
                    self.assertEqual(self.world.drive.tilt.fallen, falls)

    def test_with_the_lift_up_it_falls_sideways_over_the_frame_once_its_wheels_sink_into_the_foam(self):
        for side in ((0.0, 1.0), (0.0, -1.0)):
            with self.subTest(side=side):
                self.setUp()
                self.raise_lift_fully()
                for motor in self.lift.motors:
                    motor.set_voltage(1.5)
                self.run_for(0.3)
                at_rest = self.world.data.qpos.copy()
                limit = math.degrees(self.world.drive.fall_angle(side))
                self.assertAlmostEqual(limit, 38.9, delta=0.2)
                for past, falls in ((-1.0, False), (1.0, True)):
                    self.world.data.qpos[:] = at_rest
                    tip_toward(self.world, side, limit + past)
                    self.run_for(1.5)
                    self.assertEqual(lean_deg(self.world) > limit, falls)
                    self.assertEqual(self.world.drive.tilt.fallen, falls)

    def test_tracking_wheels_hanging_on_their_springs_do_not_change_where_it_tips(self):
        sides = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        chassis = presets.override_cascade()
        sim = Simulator(chassis, config=SimConfig(log_every=0))
        sim.set_pose(-55.0 * INCH, -12.0 * INCH, 0.0)
        world = OverrideWorld(sim, robot=cascade.CascadeRobot(), objects=False)
        on_the_floor = [world.drive.fall_angle(side) for side in sides]
        for unit in chassis.wheel_units:
            if unit.preload > 0.0:
                spring = world.model.joint(f"{wheel_body_name(cascade.ROBOT, unit)} spring")
                world.data.qpos[spring.qposadr[0]] = spring.range[0]
        mujoco.mj_forward(world.model, world.data)
        for side, angle in zip(sides, on_the_floor):
            with self.subTest(side=side):
                self.assertGreater(math.degrees(angle), 50.0)
                self.assertAlmostEqual(math.degrees(world.drive.fall_angle(side)), math.degrees(angle), delta=0.05)

    def test_with_the_lift_fully_up_braking_from_full_speed_lifts_the_back_wheels_under_two_degrees(self):
        self.raise_lift_fully()
        for motor in self.sim.motors:
            motor.set_voltage(12.0)
        self.run_for(0.9)
        self.assertGreater(self.sim.state.u / INCH, 60.0)
        for motor in self.sim.motors:
            motor.set_voltage(-12.0)
        self.run_for(0.6)
        self.assertEqual(self.world.contacts, [])
        self.assertFalse(self.world.drive.tilt.fallen)
        self.assertLess(math.degrees(self.world.drive.tilt.most), 2.0)


if __name__ == "__main__":
    unittest.main()

"""Mechanical balance and power regressions independent of motor calibration."""

import math
import random
import unittest

from vexsim import wheels
from vexsim.chassis import Chassis, WheelUnit
from vexsim.dynamics import Dynamics, normal_loads
from vexsim.motor import Motor


class Shaft:
    def __init__(self, torque):
        self.torque = torque

    def output_torque(self):
        return self.torque


class TestMechanicalRegressions(unittest.TestCase):
    def test_feasible_supports_preserve_force_and_both_moments(self):
        rng = random.Random(197205)
        for count in (1, 2, 3, 4, 6, 8, 10):
            for case in range(120):
                positions = [(rng.uniform(-.2, .2), rng.uniform(-.2, .2))
                             for _ in range(count)]
                if case % 3 == 0:
                    positions = [(x, .5 * x + .015) for x, y in positions]
                witness = [rng.expovariate(1) ** 4 for _ in positions]
                weight = 98.0665
                witness = [v / sum(witness) * weight for v in witness]
                mx = sum(x * f for (x, y), f in zip(positions, witness))
                my = sum(y * f for (x, y), f in zip(positions, witness))
                chassis = Chassis(mass=10, cg_height=.1, wheel_units=[
                    WheelUnit(wheels.omni(), x, y) for x, y in positions])
                loads = normal_loads(chassis, -mx, -my, positions)
                with self.subTest(count=count, case=case):
                    self.assertGreaterEqual(min(loads), 0)
                    self.assertAlmostEqual(sum(loads), weight, places=8)
                    self.assertAlmostEqual(sum(x * f for (x, y), f in zip(positions, loads)),
                                           mx, places=8)
                    self.assertAlmostEqual(sum(y * f for (x, y), f in zip(positions, loads)),
                                           my, places=8)

    def test_redundant_contacts_receive_even_additional_loads(self):
        chassis = Chassis(mass=10, cg_height=.1, wheel_units=[
            WheelUnit(wheels.omni(), 0, 0) for _ in range(5)])
        loads = normal_loads(chassis, 0, 0, [(0, 0)] * 5)
        for load in loads:
            self.assertAlmostEqual(load, chassis.weight / 5)

    def test_spring_loaded_contacts_carry_only_their_preload(self):
        # An odometry wheel on a spring presses with the spring force; the
        # chassis rests on the rest. Sharing the weight put 30-38% of it on
        # two 4 N tracking wheels and took that grip from the drive.
        chassis = Chassis(mass=10, cg_height=.1, wheel_units=[
            WheelUnit(wheels.omni(), 0, 0) for _ in range(5)])
        chassis.wheel_units[0].preload = 4
        loads = normal_loads(chassis, 0, 0, [(0, 0)] * 5)
        share = (chassis.weight - 4) / 4
        for i, load in enumerate(loads):
            self.assertAlmostEqual(load, 4 if i == 0 else share)

    def test_support_permutation_and_rotation_do_not_change_loads(self):
        positions = [(-.2, -.17), (.13, -.19), (.22, .14), (-.16, .21)]
        chassis = Chassis(mass=10, cg_height=.1, wheel_units=[
            WheelUnit(wheels.omni(), x, y) for x, y in positions])
        ax, ay = 10, -11
        expected = normal_loads(chassis, ax, ay, positions)
        phi = 1.3
        c, s = math.cos(phi), math.sin(phi)
        rotated = [(c * x - s * y, s * x + c * y) for x, y in positions]
        actual = normal_loads(chassis, c * ax - s * ay, s * ax + c * ay, rotated)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)
        order = [3, 1, 0, 2]
        actual = normal_loads(chassis, ax, ay, [positions[i] for i in order])
        for index, load in zip(order, actual):
            self.assertAlmostEqual(load, expected[index], places=9)

    def test_unreachable_tipping_is_reported_and_preserves_weight(self):
        chassis = Chassis(mass=10, cg_height=.1, wheel_units=[
            WheelUnit(wheels.omni(), x, y)
            for x in (-.2, .2) for y in (-.2, .2)])
        dynamics = Dynamics(chassis)
        dynamics.state.ax, dynamics.state.ay = 30, 3
        dynamics.step(1e-6)
        self.assertFalse(dynamics.normal_balance_feasible)
        self.assertAlmostEqual(dynamics.total_normal, chassis.weight)
        self.assertGreater(dynamics.normal_moment_residual[0], 10)
        self.assertAlmostEqual(dynamics.normal_moment_residual[1], 0)
        for unit in chassis.wheel_units:
            if unit.x > 0:
                self.assertAlmostEqual(unit.fz, 0, places=10)

    def test_external_gearing_is_passive_in_both_rotation_directions(self):
        for speed in (-10, 10):
            for torque in (-2, 2):
                for efficiency in (.5, .95, 1):
                    unit = WheelUnit(wheels.omni(), 0, 0, motors=[Shaft(torque)],
                                     gear_ratio=3, efficiency=efficiency)
                    unit.omega = speed
                    output_power = unit.motor_torque() * speed
                    input_power = torque * 3 * speed
                    self.assertLessEqual(output_power, input_power + 1e-12)
                    self.assertGreaterEqual(unit.power_gear_loss, 0)
                    self.assertAlmostEqual(input_power - output_power, unit.power_gear_loss)

    def test_opposing_motors_do_not_cancel_transmission_losses(self):
        unit = WheelUnit(wheels.omni(), 0, 0, motors=[Shaft(2), Shaft(-2)],
                         gear_ratio=3, efficiency=.8)
        unit.omega = 10
        self.assertLess(unit.motor_torque(), 0)
        self.assertGreater(unit.power_gear_loss, 0)
        self.assertAlmostEqual(unit.power_gear_loss, -unit.drive_torque * unit.omega)

    def test_free_rotation_preserves_world_motion_and_energy(self):
        chassis = Chassis(mass=10, inertia_z=1, wheel_units=[],
                          bearing_drag=0, yaw_drag=0, aero_drag=0)
        dynamics = Dynamics(chassis)
        dynamics.state.u, dynamics.state.v, dynamics.state.r = 1, -.3, 20
        for _ in range(10000):
            dynamics.step(.001)
        self.assertAlmostEqual(dynamics.state.x, 10, places=8)
        self.assertAlmostEqual(dynamics.state.y, -3, places=8)
        vx, vy = dynamics.state.world_velocity()
        self.assertAlmostEqual(vx, 1, places=8)
        self.assertAlmostEqual(vy, -.3, places=8)
        self.assertAlmostEqual(dynamics.state.speed ** 2, 1.09, places=9)

    def test_reset_clears_motion_outputs_but_preserves_constraints(self):
        unit = WheelUnit(wheels.omni(), 0, 0, motors=[Motor()])
        chassis = Chassis(wheel_units=[unit])
        dynamics = Dynamics(chassis)
        dynamics.state.u, dynamics.state.v, dynamics.state.r = 1, .3, .5
        unit.omega = 8
        unit.sync_motors()
        dynamics.step(.001)
        self.assertGreater(dynamics.kinetic_energy, 0)
        self.assertGreater(unit.fz, 0)
        dynamics.locked = dynamics.constrained = True
        dynamics.ext_fx, dynamics.ext_fy, dynamics.ext_mz = 5, 2, -1
        dynamics.reset(3, 4, .2)
        self.assertEqual((dynamics.state.x, dynamics.state.y, dynamics.state.theta), (3, 4, .2))
        self.assertEqual((dynamics.state.u, dynamics.state.v, dynamics.state.r), (0, 0, 0))
        self.assertEqual((dynamics.state.ax, dynamics.state.ay, dynamics.state.alpha), (0, 0, 0))
        self.assertTrue(dynamics.locked and dynamics.constrained)
        self.assertEqual((dynamics.ext_fx, dynamics.ext_fy, dynamics.ext_mz), (5, 2, -1))
        for name in ('total_fx', 'total_fy', 'total_mz', 'total_normal',
                     'wheel_slip_power', 'rolling_loss_power', 'drag_power',
                     'scrub_loss_power', 'kinetic_energy',
                     'lateral_transfer_ratio', 'pitch_transfer_ratio'):
            self.assertEqual(getattr(dynamics, name), 0, name)
        self.assertTrue(dynamics.normal_balance_feasible)
        self.assertEqual(dynamics.normal_moment_residual, (0, 0))
        self.assertFalse(dynamics.any_slipping)
        self.assertEqual(dynamics.tipping_margin, 1)
        for name in ('omega', 'angle', 'fz', 'fx', 'fy', 'kappa', 'tan_alpha',
                     'kappa_ss', 'tan_alpha_ss', 'vx', 'vy', 'slip_measure',
                     'drive_torque', 'roll_resist_torque', 'power_gear_loss'):
            self.assertEqual(getattr(unit, name), 0, name)
        self.assertFalse(unit.slipping)
        self.assertEqual(unit.motors[0].velocity, 0)
        self.assertEqual(unit.motors[0].position, 0)


if __name__ == '__main__':
    unittest.main()

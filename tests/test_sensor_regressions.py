"""Sensor geometry, timebase and published-range regressions."""
import math
import unittest

from vexsim import presets
from vexsim.sensors import DistanceSensor, IMU
from vexsim.sim import SimConfig, Simulator
from vexsim.units import DEG, G


class SensorRegressions(unittest.TestCase):
    def test_fixed_gear_motor_encoder_ignores_cartridge_label(self):
        chassis = presets.five_five_watt()
        for motor in chassis.motors:
            motor.cartridge = "blue"
        sim = Simulator(chassis, config=SimConfig(log_every=0))
        for encoder in sim.encoders.values():
            self.assertEqual(encoder.ticks_per_rev, 900)

    def test_gyro_integrates_only_elapsed_time(self):
        imu = IMU(bias=0, bias_walk=0, noise=0, scale_error=0)
        imu.update(1, 0, 0, 0)
        self.assertEqual(imu.heading, 0)
        for i in range(1, 21):
            imu.update(1, 0, 0, i * .011)
        self.assertAlmostEqual(imu.heading, .22)

    def test_overload_policy_is_saturation_and_invalid(self):
        imu = IMU(bias=0, bias_walk=0, noise=0, scale_error=0, accel_noise=0)
        imu.update(2000 * DEG, 8 * G, -8 * G, 0)
        self.assertFalse(imu.valid)
        self.assertEqual(imu.rate, 1000 * DEG)
        self.assertEqual(imu.accel_x, 4 * G)
        self.assertEqual(imu.accel_y, -4 * G)
        imu.update(0, 0, 0, .02)
        self.assertTrue(imu.valid)

    def test_distance_uses_finite_segments_and_limits(self):
        sensor = DistanceSensor(field_half=1, noise=0)
        for i, (x, y, angle, valid) in enumerate((
                (0, 0, 0, True), (2, 2, math.pi, False),
                (.999, 0, 0, False), (1, 0, 0, False),
                (.98, 0, 0, True), (2, 0, math.pi, True))):
            sensor.update(x, y, angle, i * .03)
            self.assertEqual(sensor.valid, valid)
            if valid:
                self.assertGreaterEqual(sensor.distance, sensor.min_range)
                self.assertLessEqual(sensor.distance, sensor.max_range)

    def test_reset_clears_all_clocks_values_and_random_sequences(self):
        sim = Simulator(presets.odom_bot(), config=SimConfig(log_every=0))
        sim.run(.1, control=lambda s: None)
        sim.reset()
        sensors = [sim.imu, sim.gps, sim.distance,
                   *sim.encoders.values(), *sim.tracking.values()]
        for sensor in sensors:
            self.assertEqual(sensor._last_update, -1)
            self.assertEqual(sensor._rng.random(), type(sensor)(seed=sensor.seed)._rng.random())
        self.assertEqual(sim._control_accum, 0)
        sim.gps.position_noise = sim.gps.dropout_rate = 0
        sim.step()
        self.assertEqual(sim.gps.x, sim.state.x)


if __name__ == "__main__":
    unittest.main()

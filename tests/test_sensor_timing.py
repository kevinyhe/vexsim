"""Gyro integration and smart-port publication are independent clocks."""

import math
import unittest

from vexsim.sensors import DistanceSensor, Encoder, GPSSensor, IMU, RotationSensor
from vexsim.units import DEG, G, wrap_pi


def exact_imu(**options):
    return IMU(**dict(dict(bias=0, bias_walk=0, noise=0, scale_error=0,
                          accel_noise=0), **options))


class SensorTimingTests(unittest.TestCase):
    def test_variable_rate_triangle_matches_every_published_heading(self):
        for period in (.001, .010, .025):
            with self.subTest(publication_period=period):
                imu = exact_imu(update_period=period)
                previous_publication = -1
                for tick in range(2001):
                    t = tick * .001
                    rate = 2 * math.pi * (t if t <= 1 else 2 - t)
                    imu.update(rate, 0, 0, t)
                    if imu._last_update != previous_publication:
                        expected = (math.pi * t * t if t <= 1 else
                                    math.pi * (1 + 2 * (t - 1) - (t - 1) ** 2))
                        self.assertAlmostEqual(wrap_pi(imu.heading - expected), 0, places=11)
                        previous_publication = imu._last_update

    def test_pulse_between_reports_is_integrated_while_output_is_held(self):
        imu = exact_imu()
        imu.update(0, 0, 0, 0)
        for tick in range(1, 11):
            t = tick * .001
            # Triangle from 2 to 8 ms, height 6 rad/s: exact area .018 rad.
            rate = max(0, 6 - abs(t - .005) * 2000)
            imu.update(rate, tick, -tick, t)
            if tick < 10:
                self.assertEqual((imu.heading, imu.rate, imu.accel_x, imu.accel_y), (0, 0, 0, 0))
        self.assertAlmostEqual(imu.heading, .018, places=12)
        self.assertEqual(imu.rate, 0)
        self.assertEqual((imu.accel_x, imu.accel_y), (10, -10))

    def test_float_roundoff_does_not_add_a_millisecond_to_publication(self):
        sensors = [Encoder(), RotationSensor(), exact_imu(),
                   GPSSensor(dropout_rate=0), DistanceSensor()]
        publications = [[] for _ in sensors]
        t = 0.0
        for tick in range(1, 2001):
            t += .001
            for sensor, stamps in zip(sensors, publications):
                previous = sensor._last_update
                if isinstance(sensor, Encoder):
                    sensor.update(t, t)
                else:
                    sensor.update(0, 0, 0, t)
                if sensor._last_update != previous:
                    stamps.append(tick)
        for sensor, stamps in zip(sensors, publications):
            with self.subTest(sensor=type(sensor).__name__):
                self.assertEqual(stamps, list(range(1, 2001, round(sensor.update_period * 1000))))
        self.assertEqual(publications[0], publications[1])
        self.assertEqual(publications[1], publications[2])

    def test_reset_discards_pending_integral_and_replays_seeded_reports(self):
        imu = IMU(seed=47)

        def reports():
            result = []
            for tick in range(41):
                imu.update(tick * .01, .1, -.2, tick * .001)
                if tick % 10 == 0:
                    result.append((imu.heading, imu.rate, imu.accel_x, imu.accel_y))
            return result

        first = reports()
        imu.update(5, 0, 0, .045)  # Pending unpublished motion must be discarded.
        imu.reset()
        self.assertEqual((imu.heading, imu.rate, imu.accel_x, imu.accel_y), (0, 0, 0, 0))
        self.assertEqual(reports(), first)
        imu.reset(.7)
        imu.update(0, 0, 0, 8)
        self.assertEqual(imu.heading, .7)

    def test_noise_draws_stay_on_publication_clock(self):
        fine, coarse = IMU(seed=29), IMU(seed=29)
        fine_reports, coarse_reports = [], []
        for sensor, stride, reports in ((fine, 1, fine_reports), (coarse, 2, coarse_reports)):
            for tick in range(0, 101, stride):
                sensor.update(1, .2, -.1, tick * .001)
                if tick % 10 == 0:
                    reports.append((sensor.heading, sensor.rate, sensor.accel_x, sensor.accel_y))
        for fine_report, coarse_report in zip(fine_reports, coarse_reports):
            for actual, expected in zip(fine_report, coarse_report):
                self.assertAlmostEqual(actual, expected, places=12)
        other_seed = IMU(seed=30)
        other_seed.update(1, .2, -.1, 0)
        self.assertNotEqual(other_seed.rate, fine_reports[0][1])

    def test_bias_and_scale_still_affect_integrated_heading(self):
        imu = exact_imu(bias=.3, scale_error=.2)
        for tick in range(101):
            imu.update(1.5, 0, 0, tick * .001)
        self.assertAlmostEqual(imu.heading, (1.5 * 1.2 + .3) * .1, places=12)

    def test_overload_clamps_internal_rates_and_published_values(self):
        imu = exact_imu()
        for tick in range(11):
            imu.update(2000 * DEG, 8 * G, -8 * G, tick * .001)
        self.assertAlmostEqual(imu.heading, 1000 * DEG * .010, places=12)
        self.assertFalse(imu.valid)
        self.assertEqual((imu.rate, imu.accel_x, imu.accel_y), (1000 * DEG, 4 * G, -4 * G))
        imu.update(0, 0, 0, .020)
        self.assertTrue(imu.valid)
        self.assertEqual((imu.rate, imu.accel_x, imu.accel_y), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()

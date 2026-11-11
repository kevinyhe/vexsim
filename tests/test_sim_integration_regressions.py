"""Public clock and energy ledger regressions for bounded physics substeps."""
import unittest
from vexsim import presets
from vexsim.sim import SimConfig, Simulator


class IntegrationRegressions(unittest.TestCase):
    def test_public_tick_matches_two_half_steps_physically(self):
        coarse = Simulator(presets.speed_base(), config=SimConfig(log_every=0))
        fine = Simulator(presets.speed_base(), config=SimConfig(dt=.0005, log_every=0))
        for sim in (coarse, fine):
            for motor in sim.motors:
                motor.set_voltage(8)
        for _ in range(100):
            coarse.step()
            fine.step()
            fine.step()
        for name in ("x", "y", "theta", "u", "v", "r"):
            self.assertAlmostEqual(getattr(coarse.state, name), getattr(fine.state, name), places=12)
        self.assertAlmostEqual(coarse.energy_battery, fine.energy_battery, places=12)
        self.assertEqual(coarse.steps, 100)
        self.assertEqual(fine.steps, 200)
        self.assertAlmostEqual(coarse.t, fine.t, places=12)

    def test_algebraic_electrical_to_shaft_ledger_accounts_for_losses(self):
        sim = Simulator(presets.six_motor_450(), config=SimConfig(log_every=0))
        for motor in sim.motors:
            motor.set_voltage(8)
        sim.run(.2)
        ledger = sim.energy_report()
        self.assertGreater(ledger["external_gear_J"], 0)
        self.assertGreater(ledger["motor_friction_J"], 0)
        self.assertAlmostEqual(ledger["unaccounted_J"], 0, places=8)
        sim.reset()
        self.assertEqual(sim.energy_report()["external_gear_J"], 0)
        self.assertEqual(sim.sample()["gear_loss_W"], 0)

    def test_invalid_timestep_is_rejected_without_advancing(self):
        sim = Simulator(presets.four_motor_200())
        for dt in (0, -.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sim.step(dt)
        self.assertEqual(sim.t, 0)


if __name__ == "__main__":
    unittest.main()

"""Where does the grip run out?

Sweeps robot weight, wheel compound and field condition, and reports what
actually limits acceleration in each case: the motors or the tires.

Run:  python3 examples/03_traction_study.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Robot, differential_drive
from vexsim.units import to_inches

print("full-throttle launch, 6 motors on 600 rpm cartridges geared 36:60\n")
print(f"{'wheel':14s} {'lb':>5s} {'mu':>5s} {'motor N':>8s} {'tire N':>8s} "
      f"{'peak g':>7s} {'peak slip':>10s} {'0-40in s':>9s} {'slip J':>7s}")
print("-" * 82)

for wheel in ("omni_3.25", "traction_3.25", "flex_3.25"):
    for mass_lb in (12.0, 18.0, 26.0):
        for mu in (1.0, 0.7):
            c = differential_drive(
                mass_lb=mass_lb, wheel=wheel, cartridge="blue",
                motors_per_side=3, wheels_per_side=3, external_ratio=60.0 / 36.0)
            c.surface_mu = mu
            robot = Robot(c)
            motor_force = sum(u.max_torque / u.radius for u in c.driven_units())
            tire_force = c.max_tractive_force()

            robot.tank(12.0, 12.0)
            robot.sim.run(3.0, until=lambda s: to_inches(s.state.x) >= 40.0)
            L = robot.log
            print(f"{wheel:14s} {mass_lb:5.0f} {mu:5.2f} {motor_force:8.1f} "
                  f"{tire_force:8.1f} {L.max('accel_fwd_g'):7.3f} "
                  f"{max(abs(v) for v in L['w_L1_slip_ratio']):10.3f} "
                  f"{robot.t:9.2f} {robot.sim.energy_slip:7.2f}")
    print()

print("motor N is what the gearing can push; tire N is what the floor will hold.")
print("Whichever is smaller is what you actually get.")

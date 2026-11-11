"""Which gearing actually wins over 6 feet?

A flat-out drag race: 12 V on both sides until the robot has covered 72 inches,
then coast.  Reports the elapsed time, top speed, peak current, how much energy
went into the tires as heat, and how far the encoders lied about the distance.

Run:  python3 examples/02_compare_drivetrains.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Robot, presets
from vexsim.units import to_inches

DISTANCE_IN = 72.0

print(f"{'preset':16s} {'top in/s':>9s} {'72in in s':>10s} {'peak A':>7s} "
      f"{'min V':>6s} {'mAh':>6s} {'slip J':>7s} {'enc err in':>11s}")
print("-" * 84)

for name in presets.ALL:
    robot = Robot(presets.get(name))
    robot.tank(12.0, 12.0)
    robot.sim.run(8.0, until=lambda s: to_inches(s.state.x) >= DISTANCE_IN)
    elapsed = robot.t
    enc_in = to_inches(robot.travelled())
    true_in = to_inches(robot.true_position[0])
    robot.stop()
    r = robot.report()
    print(f"{name:16s} {r['peak_speed_inps']:9.1f} {elapsed:10.2f} "
          f"{r['peak_current_A']:7.2f} {r['min_battery_V']:6.2f} "
          f"{r['battery_mAh_used']:6.1f} {r['slip_energy_J']:7.2f} "
          f"{enc_in - true_in:11.2f}")

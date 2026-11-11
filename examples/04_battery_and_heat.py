"""What a whole match does to the battery and the motors.

Runs the same aggressive routine over and over, on a full pack and on flat
ones.  A V5 robot barely notices a half-empty battery, because the brain only
ever asks for 12 V and the pack sits above that most of its life.  Down near
the bottom of the curve the sag finally pulls the bus under 12 V and the robot
gets slower every cycle.

Run:  python3 examples/04_battery_and_heat.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Battery, Robot, presets
from vexsim.units import to_inches


def hard_cycle(robot: Robot) -> None:
    """Sprint, spin, sprint back.  Roughly what a driver does to a robot."""
    robot.drive_for(48, timeout=4.0)
    robot.turn_for(180, timeout=3.0)
    robot.drive_for(48, timeout=4.0)
    robot.turn_for(180, timeout=3.0)


for soc in (1.00, 0.30, 0.08):
    robot = Robot(presets.six_motor_600(), Battery(soc=soc))
    print(f"\nstarting state of charge {soc:.0%}  "
          f"(open circuit {robot.sim.battery.open_circuit_voltage():.2f} V)")
    print(f"{'cycle':>5s} {'t s':>7s} {'top in/s':>9s} {'min V':>6s} {'peak A':>7s} "
          f"{'soc':>6s} {'hot C':>6s} {'derate':>7s}")
    for cycle in range(1, 9):
        mark = robot.log.n
        hard_cycle(robot)
        window = {k: v[mark:] for k, v in robot.log.columns.items()}
        print(f"{cycle:5d} {robot.t:7.2f} "
              f"{to_inches(max(window['speed_mps'])):9.1f} "
              f"{min(window['battery_V']):6.2f} {max(window['battery_A']):7.2f} "
              f"{robot.sim.battery.soc:6.3f} "
              f"{max(m.temperature for m in robot.sim.motors):6.1f} "
              f"{min(m.derate for m in robot.sim.motors):7.2f}")
    e = robot.sim.energy_report()
    print("  energy out of the pack: "
          f"{e['battery_J']:.0f} J ({e['battery_mAh']:.0f} mAh); "
          f"copper {e['copper_J']:.0f} J, tire slip {e['tire_slip_J']:.1f} J, "
          f"rolling {e['rolling_J']:.0f} J")

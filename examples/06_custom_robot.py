"""Build a robot from parts and interrogate it.

Shows how to describe a chassis that isn't one of the presets, pull the
derived feedforward constants out of the physics instead of measuring them, and
print the motor's own torque/speed/current/power curve.

Run:  python3 examples/06_custom_robot.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Chassis, Feedforward, Motor, Robot, WheelUnit, wheels
from vexsim.units import INCH, LB, to_in_lb, to_rpm

# --- a 5-wheel drive: four omnis on the corners, one traction in the middle ---
motors = lambda n, side, i: [Motor(cartridge="blue", name=f"{side}{i}-{j}")
                             for j in range(n)]

units = []
for side, sy in (("L", +1), ("R", -1)):
    for i, x in enumerate((-4.5, 4.5)):
        units.append(WheelUnit(
            wheel=wheels.omni(3.25), x=x * INCH, y=sy * 5.75 * INCH,
            motors=motors(1, side, i), gear_ratio=72.0 / 36.0,   # 2:1 reduction
            efficiency=0.93, name=f"{side}{i+1}"))
    units.append(WheelUnit(
        wheel=wheels.traction(3.25), x=0.0, y=sy * 5.75 * INCH,
        motors=motors(1, side, 3), gear_ratio=72.0 / 36.0,
        efficiency=0.93, name=f"{side}mid"))

robot_chassis = Chassis(
    mass=17.5 * LB,
    wheel_units=units,
    cg_height=4.5 * INCH,
    cg_x=-1.0 * INCH,          # battery and motors sit behind centre
    length=17.5 * INCH,
    width=15.0 * INCH,
    bearing_drag=1.1,          # a slightly draggy chassis
    surface_mu=0.95,           # a field that has seen some matches
    name="custom 6-wheel, 6 motors, 600 rpm geared 2:1",
)

for k, v in robot_chassis.summary().items():
    print(f"  {k:24s} {v}")

ff = Feedforward.from_chassis(robot_chassis)
print(f"\nfeedforward derived from the physics:")
print(f"  kS {ff.ks:.3f} V   kV {ff.kv:.3f} V/(m/s)   kA {ff.ka:.3f} V/(m/s^2)")
print(f"  predicted free speed {(12.0 - ff.ks) / ff.kv:.2f} m/s")

m = units[0].motors[0]
print(f"\n{m.spec.name} on the {m.cartridge} cartridge, output shaft:")
print(f"{'rpm':>7s} {'A':>7s} {'N*m':>8s} {'in-lb':>8s} {'W in':>8s} "
      f"{'W out':>8s} {'eff %':>7s}")
for i in range(11):
    w = m.free_speed * i / 10.0
    m.velocity = w
    m.set_voltage(12.0)
    m.electrical(12.0, 0.01)
    torque = m.output_torque()
    print(f"{to_rpm(w):7.0f} {m.current:7.2f} {torque:8.3f} {to_in_lb(torque):8.2f} "
          f"{m.power_electrical:8.2f} {m.power_mechanical:8.2f} "
          f"{100 * m.efficiency:7.1f}")

print("\nnow drive it:")
r = Robot(robot_chassis)
r.drive_for(48)
r.turn_for(-135)
r.drive_for(30)
r.print_report()

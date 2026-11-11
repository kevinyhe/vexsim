"""Write an auton the way you would on the robot, then look at what it cost.

Run:  python3 examples/01_basic_auton.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Robot, presets
from vexsim.motor import BrakeMode


def auton(robot: Robot) -> None:
    robot.set_pose(0, 0, 0)
    robot.drive_for(36)             # inches
    robot.turn_for(-90)             # degrees, negative is clockwise
    robot.drive_for(24)
    robot.turn_to(180)              # absolute heading
    robot.drive_profiled(48, max_accel=3.0)
    robot.stop(BrakeMode.HOLD)
    robot.wait(0.3)


robot = Robot(presets.six_motor_600())
auton(robot)

print(f"auton finished in {robot.t:.2f} s")
robot.print_report()

robot.log.to_csv("auton.csv")
print(f"\nlogged {robot.log.n} rows x {len(robot.log.keys())} columns to auton.csv")
print("columns:", ", ".join(sorted(robot.log.keys())[:12]), "...")

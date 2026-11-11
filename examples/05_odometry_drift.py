"""How far off is the robot by the end of the auton?

Runs the same routine 40 times with different sensor error draws -- gyro bias,
bias walk, encoder noise -- and reports the spread of where the robot thinks it
is versus where it actually is.  This is the number that decides whether an
auton needs a reset against a wall halfway through.

Run:  python3 examples/05_odometry_drift.py
"""

import sys, os, math, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexsim import Robot, presets
from vexsim.sim import SimConfig
from vexsim.units import DEG, to_deg, to_inches, wrap_pi

TRIALS = 12


def routine(robot: Robot) -> None:
    for _ in range(4):
        robot.drive_for(24, timeout=3.0)
        robot.turn_for(90, timeout=3.0)


def trial(preset_name: str, seed: int):
    robot = Robot(presets.get(preset_name), config=SimConfig(log_every=0))
    robot.sim.imu.seed = seed
    robot.sim.imu.bias = (seed % 7 - 3) * 0.05 * DEG
    robot.sim.imu.scale_error = ((seed * 37) % 11 - 5) * 0.0015
    robot.sim.imu.__post_init__()
    for i, e in enumerate(robot.sim.encoders.values()):
        e.seed = seed * 100 + i
        e.noise_deg = 0.15
        e.__post_init__()
    routine(robot)
    st = robot.sim.state
    believed = (robot.sim.odometry.x, robot.sim.odometry.y)
    return (
        to_inches(math.hypot(st.x, st.y)),                       # true drift from start
        to_inches(math.hypot(believed[0] - st.x, believed[1] - st.y)),  # odom error
        to_deg(wrap_pi(robot.sim.imu.heading - st.theta)),       # heading error
    )


for preset_name in ("six_motor_200", "odom_bot"):
    closes, errors, headings = [], [], []
    for seed in range(1, TRIALS + 1):
        c, e, h = trial(preset_name, seed)
        closes.append(c)
        errors.append(e)
        headings.append(h)
    print(f"{preset_name}  ({TRIALS} runs of a 24 in square)")
    print(f"  true miss of the start point : mean {statistics.mean(closes):5.2f} in, "
          f"worst {max(closes):5.2f} in")
    print(f"  odometry position error      : mean {statistics.mean(errors):5.2f} in, "
          f"worst {max(errors):5.2f} in")
    print(f"  IMU heading error            : mean {statistics.mean(headings):+5.2f} deg, "
          f"sd {statistics.pstdev(headings):4.2f}, "
          f"worst {max(headings, key=abs):+5.2f} deg")
    print()

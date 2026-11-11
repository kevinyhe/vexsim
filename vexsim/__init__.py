"""vexsim: physics simulator for VEX V5 autonomous routines.

    from vexsim import Robot, presets
    robot = Robot(presets.six_motor_200())
    robot.drive_for(24)      # inches
    robot.turn_for(90)       # degrees
    robot.print_report()
    robot.log.to_csv("run.csv")
"""

from .api import DriveGains, Robot
from .battery import Battery, InfiniteBattery
from .chassis import Chassis, WheelUnit, add_tracking_wheels, differential_drive
from .control import (PID, Feedforward, PurePursuit, SlewLimiter,
                      TrapezoidProfile, arcade)
from .dynamics import BodyState, Dynamics, normal_loads
from .motor import (CARTRIDGES, BrakeMode, Motor, MotorMode, MotorSpec,
                    cortex_393, v5_5_5w, v5_11w)
from .sensors import IMU, DistanceSensor, Encoder, GPSSensor, Odometry, RotationSensor
from .sim import DataLog, SimConfig, Simulator
from .wheels import TireModel, WheelSpec, flex, mecanum, omni, traction
from . import presets, units, wheels

__version__ = "2.0.0"

__all__ = [
    "Robot", "DriveGains", "Simulator", "SimConfig", "DataLog",
    "Chassis", "WheelUnit", "differential_drive", "add_tracking_wheels",
    "Motor", "MotorSpec", "MotorMode", "BrakeMode", "CARTRIDGES",
    "v5_11w", "v5_5_5w", "cortex_393",
    "Battery", "InfiniteBattery",
    "Dynamics", "BodyState", "normal_loads",
    "TireModel", "WheelSpec", "omni", "traction", "flex", "mecanum",
    "PID", "Feedforward", "TrapezoidProfile", "PurePursuit", "SlewLimiter", "arcade",
    "IMU", "Encoder", "RotationSensor", "GPSSensor", "Odometry", "DistanceSensor",
    "presets", "units", "wheels",
]

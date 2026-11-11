"""Ready-made robots that look like things people actually build."""

from __future__ import annotations

from .chassis import Chassis, add_sideways_tracking_wheel, add_tracking_wheels, differential_drive
from .motor import v5_5_5w, cortex_393


def four_motor_200(mass_lb: float = 14.0) -> Chassis:
    """The default competition base: 4 motors, 200 rpm, 3.25 in omnis."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_3.25", cartridge="green",
        motors_per_side=2, wheels_per_side=2,
        track_width_in=11.5, wheel_base_in=10.5,
        name="4m 200rpm 3.25 omni",
    )


def six_motor_200(mass_lb: float = 16.0) -> Chassis:
    """6 motors, 200 rpm, 3.25 in.  Torquey, hard to push around."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_3.25", cartridge="green",
        motors_per_side=3, wheels_per_side=3, center_traction=True,
        track_width_in=11.5, wheel_base_in=11.0,
        name="6m 200rpm 3.25 omni + center traction",
    )


def six_motor_600(mass_lb: float = 15.0) -> Chassis:
    """6 motors, 600 rpm cartridges geared down 36:60 -- the fast-base recipe."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_3.25", cartridge="blue",
        motors_per_side=3, wheels_per_side=3,
        external_ratio=60.0 / 36.0,
        track_width_in=11.5, wheel_base_in=11.0,
        name="6m 600rpm 36:60 3.25 omni",
    )


def six_motor_450(mass_lb: float = 16.5) -> Chassis:
    """6 motors on blue cartridges geared 36:48 -- 450 rpm at a 3.25 in omni.

    600 rpm through a 36-tooth driving a 48-tooth is 450 rpm at the wheel, which
    is 76.6 in/s on a 3.25 in omni: the fast end of what a competition base is
    geared to, and well past what the tires can put down off the line.
    """
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_3.25", cartridge="blue",
        motors_per_side=3, wheels_per_side=3,
        external_ratio=48.0 / 36.0,
        track_width_in=11.5, wheel_base_in=11.0,
        name="6m 450rpm 36:48 3.25 omni",
    )


def four_motor_450(mass_lb: float = 16.5) -> Chassis:
    """4 blue-cartridge motors geared 36:48 on 3.25 in omnis -- 450 rpm, 44 W.

    Override caps drive motors at 55 W (R11); four 11 W motors leave room for
    a two-motor lift inside the 88 W total (R10).  16.5 lb is the drive plus a
    cascade lift.
    """
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_3.25", cartridge="blue",
        motors_per_side=2, wheels_per_side=2,
        external_ratio=48.0 / 36.0,
        track_width_in=11.5, wheel_base_in=10.5,
        cg_height_in=5.0,
        name="4m 450rpm 36:48 3.25 omni",
    )


def override_cascade() -> Chassis:
    """The drive of the robot's owner's Override cascade robot (2026-11-04/05):
    4 blue-cartridge motors geared 36:48 on 2.75 in omnis -- 450 rpm, 44 W.
    About 14 lb with the battery, its CG about 3.5 in up with the lift down.
    One sprung, unpowered tracking wheel at the centre of rotation measures
    how far it slides sideways; its 2 in size is a guess.
    """
    chassis = differential_drive(
        mass_lb=14.0, wheel="omni_2.75", cartridge="blue",
        motors_per_side=2, wheels_per_side=2,
        external_ratio=48.0 / 36.0,
        track_width_in=11.5, wheel_base_in=10.5,
        cg_height_in=3.5,
        name="Override cascade: 4m 450rpm 36:48 2.75 omni + sideways tracking",
    )
    return add_sideways_tracking_wheel(chassis, diameter_in=2.0)


def speed_base(mass_lb: float = 13.0) -> Chassis:
    """4 motors, 600 rpm, 4 in wheels, geared 36:48.  Fast and slippery."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_4", cartridge="blue",
        motors_per_side=2, wheels_per_side=2,
        external_ratio=48.0 / 36.0,
        track_width_in=11.0, wheel_base_in=10.5,
        name="4m 600rpm 36:48 4in omni",
    )


def push_bot(mass_lb: float = 22.0) -> Chassis:
    """Heavy, slow, all traction wheels.  Built to shove."""
    return differential_drive(
        mass_lb=mass_lb, wheel="traction_4", cartridge="red",
        motors_per_side=3, wheels_per_side=3,
        track_width_in=11.5, wheel_base_in=11.0,
        cg_height_in=4.0,
        name="6m 100rpm 4in traction",
    )


def tall_bot(mass_lb: float = 20.0) -> Chassis:
    """Same drive as the 6-motor base but with the CG up high, so it tips."""
    c = six_motor_200(mass_lb)
    c.cg_height = 0.36
    c.name = "6m 200rpm, high CG"
    return c


def odom_bot(mass_lb: float = 16.0) -> Chassis:
    """6-motor base with two spring-loaded tracking wheels."""
    c = six_motor_200(mass_lb)
    c.name = "6m 200rpm + tracking wheels"
    return add_tracking_wheels(c, forward_offset_in=-3.0)


def five_five_watt(mass_lb: float = 9.0) -> Chassis:
    """A light base on 5.5 W motors, the way many V5RC support bots are built."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_2.75", cartridge="green",
        motors_per_side=2, wheels_per_side=2,
        track_width_in=9.0, wheel_base_in=8.0,
        motor_spec=v5_5_5w,
        name="4x 5.5W 200rpm 2.75 omni",
    )


def legacy_393(mass_lb: float = 16.0) -> Chassis:
    """Cortex-era 393s, for seeing how much V5 actually changed."""
    return differential_drive(
        mass_lb=mass_lb, wheel="omni_4", cartridge="393_torque",
        motors_per_side=3, wheels_per_side=3,
        external_ratio=1.0,
        track_width_in=11.5, wheel_base_in=11.0,
        motor_spec=cortex_393,
        name="6x 393 standard 4in omni",
    )


ALL = {
    "four_motor_200": four_motor_200,
    "six_motor_200": six_motor_200,
    "six_motor_450": six_motor_450,
    "six_motor_600": six_motor_600,
    "four_motor_450": four_motor_450,
    "override_cascade": override_cascade,
    "speed_base": speed_base,
    "push_bot": push_bot,
    "tall_bot": tall_bot,
    "odom_bot": odom_bot,
    "five_five_watt": five_five_watt,
    "legacy_393": legacy_393,
}


def get(name: str) -> Chassis:
    if name not in ALL:
        raise KeyError(f"unknown preset '{name}'; have {sorted(ALL)}")
    return ALL[name]()

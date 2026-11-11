"""Command line front end:  python -m vexsim <command>

    list                       show the built-in robots, wheels and cartridges
    spec   <preset>            mass, gearing, top speed, traction limits
    curve  <preset>            the drivetrain's torque / speed / power curve
    run    <preset> [--auton]  run a routine and print the report
    sweep  [--auton]           run every preset through the same routine
    stall  <preset>            hold full power against a wall until it derates
"""

from __future__ import annotations

import argparse
import sys

from . import presets, wheels
from .api import Robot
from .battery import Battery
from .motor import CARTRIDGES, Motor

#: one wheel's share of a 15 lb robot on six wheels, in newtons
NOMINAL_WHEEL_LOAD = 15.0 * 4.448 / 6.0
from .sim import SimConfig
from .units import to_inches, to_lbs, to_rpm


def cmd_list(args) -> None:
    print("robots:")
    for name, fn in presets.ALL.items():
        c = fn()
        print(f"  {name:16s} {c.name:42s} {to_lbs(c.mass):5.1f} lb  "
              f"{c.motor_count} motors  {to_inches(c.theoretical_free_speed()):5.1f} in/s")
    print("\nwheels:")
    for name in sorted(wheels.PRESETS):
        w = wheels.get(name)
        t = w.tire
        print(f"  {name:16s} {to_inches(w.diameter):5.2f} in  "
              f"mu_long {t.mu_long_static:4.2f}  mu_lat {t.mu_lat_static:4.2f}  "
              f"Crr {t.rolling_resistance(NOMINAL_WHEEL_LOAD, w.radius):.3f} at {NOMINAL_WHEEL_LOAD:.0f} N")
    print("\ncartridges:")
    for name, (ratio, out_rpm) in CARTRIDGES.items():
        m = Motor(cartridge=name)
        print(f"  {name:8s} {ratio:5.1f}:1  {out_rpm:6.0f} rpm free  "
              f"{m.stall_torque:5.2f} N*m stall")


def cmd_spec(args) -> None:
    c = presets.get(args.preset)
    s = c.summary()
    print(f"{s['name']}")
    print(f"  mass                {to_lbs(s['mass_kg']):8.2f} lb  ({s['mass_kg']:.2f} kg)")
    print(f"  motors              {s['motors']:8d}")
    print(f"  wheels              {s['wheels']:8d}")
    print(f"  track / wheelbase   {to_inches(s['track_width_m']):8.2f} / "
          f"{to_inches(s['wheel_base_m']):.2f} in")
    print(f"  yaw inertia         {s['inertia_z_kgm2']:8.4f} kg m^2")
    print(f"  free speed          {to_inches(s['free_speed_mps']):8.2f} in/s "
          f"({s['free_speed_mps']:.2f} m/s)")
    print(f"  max tractive force  {s['max_tractive_force_N']:8.2f} N")
    print(f"  max acceleration    {s['max_accel_mps2']:8.2f} m/s^2 "
          f"({s['max_accel_mps2'] / 9.80665:.2f} g)")
    print(f"  limited by          {'traction' if s['traction_limited'] else 'motors':>8s}")
    print(f"  peak draw           {s['peak_electrical_W']:8.0f} W at the ports")
    for u in c.wheel_units:
        print(f"    {u.name:16s} {u.wheel.name:18s} x={to_inches(u.x):6.2f} "
              f"y={to_inches(u.y):6.2f} in  gear {u.gear_ratio:4.2f}  "
              f"{len(u.motors)} motor(s)")


def cmd_curve(args) -> None:
    c = presets.get(args.preset)
    u = c.driven_units()[0]
    m = u.motors[0]
    n_driven = sum(len(uu.motors) for uu in c.driven_units())
    print(f"{c.name}: force / speed curve at 12 V (all {n_driven} motors)")
    print(f"{'speed in/s':>11s} {'wheel rpm':>10s} {'motor A':>8s} "
          f"{'force N':>8s} {'accel g':>8s} {'elec W':>8s} {'mech W':>8s} {'eff %':>7s}")
    v_free = c.theoretical_free_speed()
    for i in range(11):
        v = v_free * i / 10.0
        w_wheel = v / u.radius
        w_motor = w_wheel * u.gear_ratio
        emf = m.spec.ke * w_motor * m.ratio
        cur = min((12.0 - emf) / m.spec.resistance, m.current_limit)
        torque = (m.spec.kt * cur * m.ratio * m.spec.gearbox_efficiency
                  - m.coulomb_torque)
        force = torque * u.gear_ratio * u.efficiency / u.radius * n_driven
        force = min(force, c.max_tractive_force())
        p_elec = 12.0 * cur * n_driven
        p_mech = force * v
        print(f"{to_inches(v):11.1f} {to_rpm(w_wheel):10.1f} {cur:8.2f} "
              f"{force:8.2f} {force / c.mass / 9.80665:8.3f} {p_elec:8.1f} "
              f"{p_mech:8.1f} {100.0 * p_mech / max(p_elec, 1e-9):7.1f}")


AUTONS = {}


def auton(fn):
    AUTONS[fn.__name__] = fn
    return fn


@auton
def square(robot: Robot) -> None:
    """A 24 inch square: the classic drift test."""
    for _ in range(4):
        robot.drive_for(24)
        robot.turn_for(90)


@auton
def sprint(robot: Robot) -> None:
    """Full power down the field and stop."""
    robot.drive_seconds(2.0, 12.0)
    robot.wait(0.5)


@auton
def slalom(robot: Robot) -> None:
    """Pure pursuit through a few waypoints."""
    robot.follow_path([(0, 0), (18, 0), (30, 12), (42, 12), (54, 0), (72, 0)],
                      velocity_mps=0.7)


@auton
def profiled(robot: Robot) -> None:
    """Trapezoid profile out and back."""
    robot.drive_profiled(48, max_accel=3.0)
    robot.turn_for(180)
    robot.drive_profiled(48, max_accel=3.0)


@auton
def match_load(robot: Robot) -> None:
    """Fifteen seconds of hard driving, to watch the battery and heat."""
    for _ in range(6):
        robot.drive_for(30, timeout=3.0)
        robot.turn_for(180, timeout=3.0)


def cmd_run(args) -> None:
    c = presets.get(args.preset)
    battery = Battery(soc=args.soc)
    cfg = SimConfig(dt=args.dt, control_period=args.control_period)
    robot = Robot(c, battery, cfg)
    if args.auton not in AUTONS:
        print(f"unknown auton '{args.auton}'; have {sorted(AUTONS)}")
        sys.exit(1)
    AUTONS[args.auton](robot)
    print(f"{c.name}  /  {args.auton}")
    robot.print_report()
    if args.csv:
        robot.log.to_csv(args.csv)
        print(f"  wrote {robot.log.n} rows to {args.csv}")


def cmd_sweep(args) -> None:
    header = (f"{'preset':16s} {'time s':>7s} {'x in':>7s} {'y in':>7s} "
              f"{'hdg err':>8s} {'odom err':>9s} {'peak A':>7s} {'min V':>6s} "
              f"{'mAh':>6s} {'hot C':>6s} {'slip J':>7s}")
    print(header)
    print("-" * len(header))
    for name in presets.ALL:
        robot = Robot(presets.get(name), Battery(soc=args.soc))
        AUTONS[args.auton](robot)
        r = robot.report()
        print(f"{name:16s} {r['time_s']:7.2f} {r['true_x_in']:7.2f} "
              f"{r['true_y_in']:7.2f} {r['true_heading_deg']:8.2f} "
              f"{r['odom_position_error_in']:9.3f} {r.get('peak_current_A', 0):7.2f} "
              f"{r.get('min_battery_V', 0):6.2f} {r['battery_mAh_used']:6.1f} "
              f"{r['hottest_motor_C']:6.1f} {r.get('slip_energy_J', 0):7.2f}")


def cmd_stall(args) -> None:
    """Drive into a wall: the sim's worst case for current and heat."""
    c = presets.get(args.preset)
    robot = Robot(c, Battery(soc=args.soc))
    robot.sim.dynamics.locked = True     # pinned: wheels cannot turn
    robot.tank(12.0, 12.0)
    print(f"{c.name}: stalled against a wall at 12 V")
    print(f"{'t s':>6s} {'A':>7s} {'V':>6s} {'W':>7s} {'temp C':>7s} "
          f"{'derate':>7s} {'push N':>8s} {'soc':>6s}")
    for i in range(60):
        robot.sim.run(1.0)
        d = robot.sim
        print(f"{d.t:6.1f} {d.battery.current:7.2f} {d.battery.voltage:6.2f} "
              f"{d.battery.power_out:7.1f} "
              f"{max(m.temperature for m in d.motors):7.1f} "
              f"{min(m.derate for m in d.motors):7.2f} "
              f"{sum(u.drive_torque / u.radius for u in c.driven_units()):8.1f} "
              f"{d.battery.soc:6.3f}")
        if min(m.derate for m in d.motors) <= 0.0:
            print("  motors cut out")
            break


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vexsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    sp = sub.add_parser("spec")
    sp.add_argument("preset")
    sp.set_defaults(func=cmd_spec)

    cu = sub.add_parser("curve")
    cu.add_argument("preset")
    cu.set_defaults(func=cmd_curve)

    ru = sub.add_parser("run")
    ru.add_argument("preset")
    ru.add_argument("--auton", default="square", help=f"one of {sorted(AUTONS)}")
    ru.add_argument("--csv", default=None)
    ru.add_argument("--soc", type=float, default=1.0)
    ru.add_argument("--dt", type=float, default=0.001)
    ru.add_argument("--control-period", type=float, default=0.010)
    ru.set_defaults(func=cmd_run)

    sw = sub.add_parser("sweep")
    sw.add_argument("--auton", default="square")
    sw.add_argument("--soc", type=float, default=1.0)
    sw.set_defaults(func=cmd_sweep)

    st = sub.add_parser("stall")
    st.add_argument("preset")
    st.add_argument("--soc", type=float, default=1.0)
    st.set_defaults(func=cmd_stall)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

# vexsim

Physics simulator for VEX V5 autonomous routines. It steps a robot's
drivetrain, sensors and mechanisms at 1 ms, so a routine can be run and
measured before it runs on a robot.

- **Drivetrain.** DC motor model with gearbox losses and a thermal limit, a
  battery that sags, slip-based tires on foam tiles, and load transfer.
- **Sensors.** Encoders, rotation sensor, IMU, GPS and distance sensor, each
  with its refresh period, rounding and seeded noise.
- **Override field (2026-27).** Walls, Goals, Loaders, Toggles, Pins and
  Cups as MuJoCo bodies, a cascade-lift robot, the partner's and opponents'
  robots, and the game's scoring.

[Physics model](docs/model.md) · [Override world](docs/override.md) ·
[Outputs](docs/outputs.md)

[mclib](https://github.com/kevinyhe/mclib)'s `tests/vexsim/` runs mclib's C++
motion code against this simulator, and mclib's path planner runs whole
routines on it.

## Setup

The drivetrain needs Python 3.8 or later and nothing else. The Override
world needs MuJoCo and numpy.

```sh
git clone https://github.com/kevinyhe/vexsim
cd vexsim
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Override field shapes are in `cad/override/`, cut from
[VEX's Override field CAD](https://content.vexrobotics.com/docs/2026-2027/override/files/v5rc-override-fieldcad.zip)
(276-9250-000, 26 April 2026). When VEX publishes new CAD or
`tools/override_cad.py` changes, cut them again from the unzipped STEP file
and commit the four files it writes:

```sh
.venv/bin/pip install cadquery-ocp==8.0.1.1.0
.venv/bin/python tools/override_cad.py "<path>/276-9250-000 (2026-04-26).STEP"
```

The first run takes about 20 minutes; a second run on the same STEP file
skips the cutting.

## Use

```python
from vexsim import Robot, presets

robot = Robot(presets.six_motor_600())
robot.drive_for(36)              # inches
robot.turn_for(-90)              # degrees, counter-clockwise positive
robot.drive_profiled(48, max_accel=3.0)
robot.print_report()
robot.log.to_csv("run.csv")
```

Each `Robot` call blocks: it runs the physics until the move ends or times
out. To run your own controller, set motor voltages and call `sim.step()`.

On the Override field, step the world instead of the simulator:

```python
from vexsim import Simulator, cascade, presets
from vexsim.override import OverrideWorld
from vexsim.units import INCH

sim = Simulator(presets.override_cascade())
sim.set_pose(-40 * INCH, -12 * INCH, 0.0)      # metres from the field centre, radians
world = OverrideWorld(sim, robot=cascade.CascadeRobot(), preload="yellow")
for motor in world.mechanism.lift.motors:
    motor.set_voltage(12.0)
world.step(0.001)
world.score()                                   # points by alliance
world.reported_contacts()                       # each hit: where, peak force, drive current
```

## Command line

```sh
python3 -m vexsim list                               # robots, wheels, cartridges
python3 -m vexsim spec six_motor_450                 # mass, gearing, top speed, traction
python3 -m vexsim curve six_motor_450                # force, current and power against speed at 12 V
python3 -m vexsim run six_motor_200 --auton square --csv run.csv
python3 -m vexsim sweep --auton sprint               # every robot through one routine
python3 -m vexsim stall push_bot                     # full power into a wall until the motors derate
```

Routines for `run` and `sweep`: `square`, `sprint`, `slalom`, `profiled`,
`match_load`.

## Robots

`vexsim/presets.py`. `python3 -m vexsim spec <name>` prints each one's
numbers.

| Preset | Motors | Wheels | Mass |
| --- | --- | --- | --- |
| `four_motor_200` | 4 green, 200 rpm | 3.25 in omni | 14 lb |
| `six_motor_200` | 6 green, 200 rpm | 3.25 in omni, centre traction | 16 lb |
| `six_motor_450` | 6 blue, 36:48, 450 rpm | 3.25 in omni | 16.5 lb |
| `six_motor_600` | 6 blue, 36:60, 360 rpm | 3.25 in omni | 15 lb |
| `four_motor_450` | 4 blue, 36:48, 450 rpm | 3.25 in omni | 16.5 lb |
| `override_cascade` | 4 blue, 36:48, 450 rpm | 2.75 in omni, sprung 2 in sideways tracking wheel | 14 lb, CG 3.5 in up |
| `speed_base` | 4 blue, 36:48, 450 rpm | 4 in omni | 13 lb |
| `push_bot` | 6 red, 100 rpm | 4 in traction | 22 lb |
| `tall_bot` | as `six_motor_200` | | 20 lb, CG 14.2 in up |
| `odom_bot` | as `six_motor_200` | two sprung tracking wheels | 16 lb |
| `five_five_watt` | 4 x 5.5 W, 200 rpm | 2.75 in omni | 9 lb |
| `legacy_393` | 6 x 393 | 4 in omni | 16 lb |

`differential_drive(...)` builds these shapes. For any other, put
`WheelUnit`s in a `Chassis` (`examples/06_custom_robot.py`).

`with_traction_wheels(chassis, "middle")` swaps the middle wheel on each side
of a 6-wheel drive for a traction wheel the same size, and `"all"` swaps every
driven wheel. In a curve at 12 V and 7 V, `six_motor_450` slides sideways at
30 in/s on omnis, 6 in/s with middle traction wheels and 1 in/s on all
traction wheels, and turns at 155, 94 and 38 degrees a second.

## Layout

| Module | Contents |
| --- | --- |
| `sim.py` | `Simulator`: the step, motor writes, sensor clocks, the log |
| `api.py` | `Robot`: blocking moves on a `Simulator` |
| `motor.py`, `battery.py` | motors and cartridges; the battery |
| `wheels.py`, `foam.py` | tire model and wheel presets; grip and rolling resistance on foam |
| `chassis.py`, `dynamics.py` | wheel layout and mass; body dynamics, load transfer, tipping |
| `sensors.py` | encoders, rotation sensor, IMU, GPS, distance sensor, odometry |
| `control.py` | PID, profiles, pure pursuit, feedforward |
| `arena.py` | robot outlines, contact reports, `open_floor()` |
| `override.py`, `override_field.py` | `OverrideWorld`; the field's sizes and rules from the game manual |
| `pieces.py`, `toggles.py`, `override_cad.py` | Pins, Cups, Goals and Toggles in MuJoCo; the CAD shapes |
| `other_robots.py` | the partner's and opponents' robots: hold still or drive a path |
| `cascade.py`, `mujoco_drive.py` | the cascade robot; the drivetrain as MuJoCo bodies |
| `mujoco_checks.py`, `out_of_reach.py`, `setup_cache.py` | MuJoCo error checks, contact culling, a disk cache of world setups |

## Examples

```sh
python3 examples/01_basic_auton.py          # a routine and its report
python3 examples/02_compare_drivetrains.py  # every preset over 6 ft
python3 examples/03_traction_study.py       # weight, wheel compound, field condition
python3 examples/04_battery_and_heat.py     # a match on one charge
python3 examples/05_odometry_drift.py       # Monte Carlo over sensor error
python3 examples/06_custom_robot.py         # a chassis from parts
```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests
```

The Override tests need the field shapes. Passing the suite does not show
the model matches a real robot. Each module lists its guessed numbers in a
`GUESSES` table.

## License

MPL-2.0 ([LICENSE](LICENSE)).

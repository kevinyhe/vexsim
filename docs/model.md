# Physics model

SI units inside; `vexsim.units` converts. +x forward, +y left, angles
counter-clockwise positive. Guessed numbers are listed with their reasons in
the `GUESSES` table of `sim.py`, `dynamics.py`, `foam.py`, `mujoco_drive.py`,
`arena.py`, `override.py`, `override_field.py`, `pieces.py`, `toggles.py`
and `cascade.py`.

## Timing

| | Period |
| --- | --- |
| `Simulator.step()` | 1 ms (`SimConfig.dt`), split into substeps of at most 0.5 ms (`SimConfig.max_physics_dt`) |
| Motor writes from user code | every 5 ms, at 2.5, 7.5, 12.5 ms... Only the last command before a write reaches the motor. |
| Motor velocity, position and hold PID | every 10 ms; output held between updates |
| Smart-port sensors | 10 ms, with a refresh delay |
| `Robot` control loop | 10 ms (`SimConfig.control_period`) |
| Log row | 10 ms (`SimConfig.log_every`) |

A loop that sends on the 10 ms tick waits 2.5 ms for each write. A `Motor`
used without a `Simulator` takes commands at once.

## Motors (`motor.py`)

Passive DC equivalent: `Kt = Ke`, winding resistance, optional inductance
(`electrical_dynamics=True`), direction-aware gearbox efficiency, drag, and a
lumped thermal RC. V5 motors add a speed, current and power governor.

| Model | Fitted to |
| --- | --- |
| 11 W | 100 / 200 / 600 rpm and 2.1 / 1.05 / 0.35 N·m for red / green / blue, with the near-nominal-speed plateau under moderate load |
| 5.5 W | 200 rpm, 0.5 N·m, 5.5 W |
| 393 | 100 rpm, 1.67 N·m, 4.8 A stall, 0.37 A free, at 7.2 V |

These are fitted equivalents, not measured internals. Resistance, gearbox
efficiency, inertia, governor gains and thermal RC are not fixed by the
published endpoints. The 11 W current limit falls to 50% at 55 °C, 25% at
60 °C, 12.5% at 65 °C and 0 at 70 °C.

Coast opens the leads. Brake sets zero winding voltage. A motor driven
faster than the bridge can control reports `regenerative_overcurrent`, and
the bus voltage stays inside the supply rails.

Sources: [11 W](https://kb.vex.com/hc/en-us/articles/360044325872-Understanding-V5-Smart-Motor-11W-Performance),
[5.5 W](https://kb.vex.com/hc/en-us/articles/10002101702932-Understanding-V5-Smart-Motor-5-5W-Performance),
[393](https://projectpartners.pltw.org/Training3/story_content/external_files/VEX_2_Wire_Motor_393.pdf),
[thermal limits](https://www.vexforum.com/t/v5-motor-current-limit-vs-temperature/107172),
[PROS write rate](https://docs.rs/pros-devices/0.2.0/src/pros_devices/smart/motor.rs.html).

## Battery (`battery.py`)

4S LiFePO4, 12.8 V nominal, 1100 mAh
([VEX](https://www.vexrobotics.com/276-4811.html)). Terminal voltage is
`OCV - I*R` with 35 mΩ; the open-circuit curve and resistance are not
measured. Charge is tracked and runs out. Pack current comes from the
bridge's power balance, not the winding current. Regeneration into a full
pack goes to a shunt and is logged as `battery_rejected_W`.

`InfiniteBattery` is an ideal 12 V supply that sinks and sources current.

## Tires (`wheels.py`)

Force comes from slip, not from a grip limit:

```
d      = max(|vx|, |w*r|, 0.05)        normalising speed
kappa  = (w*r - vx) / d                longitudinal slip ratio
tan(a) = vy / d                        lateral slip
```

Each is scaled by the slip at the tire's peak and run through a curve that
rises to the static coefficient and decays to the sliding one. Force fades
below `CREEP_SPEED` of patch sliding. Slip lags through a relaxation length,
capped at `RELAX_TAU_MAX = 0.05 s`.

Forward and sideways have separate coefficients. A 3.25 in omni has
`mu_long = 0.87` and `mu_lat = 0.26`. `axis_coupling` sets how much sideways
slip costs forward grip: 1.0 for a traction wheel (one friction ellipse),
0.12 for an omni, whose sideways motion is on its rollers.

Presets: omni, traction, flex (30A, 35A, 45A, 60A), mecanum and unpowered
tracking wheels in the usual diameters. For an X-pattern mecanum chassis,
pass opposite `roller_sign` values to `mecanum()` on the two diagonals.

In the MuJoCo drive (`mujoco_drive.py`), forward slip is how far the tread
is bent. It bends as fast as the wheel's surface runs ahead of the floor and
straightens as the wheel rolls, with no speed floor, so a wheel stalled on a
wall holds its push without creeping.

## Foam tiles (`foam.py`)

| Wheel | Grip | Source |
| --- | --- | --- |
| 4 in omni | 0.90 | VEX Forum pull test |
| 2.75 in omni | 0.84 | VEX Forum pull test |
| 3.25 in omni | 0.87 | guess between the two |
| traction | 0.81, the same sideways | VEX Forum tilt test; sideways is a guess |
| mecanum | 0.85 | VEX Forum locked-wheel drag |

Rolling resistance and turning scrub come from how far a wheel sinks. Each
patch of tile is a spring of stiffness modulus / thickness (EVA, 2.77 MPa at
80 kg/m³, 0.63 in tile). Rolling resistance is
`3/8 x 0.38 x sqrt(sink / wheel diameter)`: 0.013 for a 15 lb robot on six
3.25 in omnis, which sink 0.6 mm. Turning-in-place torque is 2.4 mm x load
for a 3.25 in omni and 5.5 mm x load for a traction wheel.

## Chassis and dynamics (`chassis.py`, `dynamics.py`)

Three body degrees of freedom (forward, sideways, yaw), plus a spin state
and two lagged slip states per wheel. Every force on the robot comes from a
contact patch.

Wheel loads are the distribution closest to even that satisfies

```
sum(Fz)         = m*g
sum(x_i * Fz_i) = -h*m*ax        pitch
sum(y_i * Fz_i) = -h*m*ay        roll
```

When the centre of pressure falls outside the support polygon, loads
saturate at its edge; `normal_balance_feasible` and
`normal_pitch_residual_Nm` / `normal_roll_residual_Nm` report it.

The chassis rocks onto an edge of its footprint (`dynamics.Tilt`):

```
I_edge * angle'' = m*a*(h cos(angle) + d sin(angle)) - m*g*(d cos(angle) - h sin(angle))
```

`h` is the CG height and `d` the edge's distance from the CG. Wheels off the
floor carry no load. Past `atan(d / h)` the robot falls and `tilt.fallen`
stays set. Braking from full speed at 12 V caps near 0.48 g, so a CG 5 in up
never lifts; 13 in rocks 7 degrees and lands; 15 in falls.

Reflected motor inertia is carried through the gearing. Yaw scrub is in
`drag_power`.

## Sensors (`sensors.py`)

| Sensor | Model |
| --- | --- |
| Motor encoder | 1800 / 900 / 300 ticks per output turn for red / green / blue |
| Rotation sensor | V5 rotation sensor resolution and period |
| IMU | calibration bias, bias random walk, scale error, noise; saturates at ±4 g and ±1000 °/s and marks the reading invalid |
| GPS | position with noise and dropouts |
| Distance | rays against the field walls, 20 to 2000 mm |
| Odometry | arc integration, corrected for where the tracking wheels sit |

Every sensor is seeded, so runs repeat and Monte Carlo over a routine
works. The gyro integrates every rate sample with the trapezoid rule; the
port period only sets when the held value updates.

## Control (`control.py`)

PID with an integral zone, windup clamp, derivative on measurement and
output slew; trapezoid profiles; pure pursuit; and a feedforward whose kS,
kV and kA come from the physical model.

## Not modelled

- Suspension, climbing, and pitch or roll beyond rocking onto one edge.
- Collisions without a world: the bare `Simulator` drives through
  everything. Robots never hit each other.
- `Robot.push()` and `dynamics.locked` hold the chassis or wheels for push
  and stall tests; neither models the obstacle.
- One friction number for the whole floor.
- Mechanisms other than the Override cascade robot's. Their motors draw
  from the same battery through `Simulator.power_mechanisms`; add others the
  same way.

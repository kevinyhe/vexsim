# Override world

`OverrideWorld` (`override.py`) puts a `CascadeRobot` on the 2026-27
Override field as Game Manual 2.0 (3 September 2026) draws it. The drive,
field, pieces and mechanism share one MuJoCo model. Step the world, not the
simulator.

```python
OverrideWorld(sim, robot, objects=True, preload=None, others=())
```

`robot` starts where `sim`'s pose puts it. `preload` is `"red"` or
`"yellow"`: the half of the preload Pin that points up in the closed claw.
`others` are the partner's and opponents' robots (below). A robot placed
inside anything but its preload raises `StartOverlapError`, whose `robot`
says which.

| Call | Returns |
| --- | --- |
| `step(dt)` | advances the world by `dt` seconds |
| `score(autonomous=True)` | points by alliance, `<SC1>` to `<SC8>` |
| `reported_contacts()` | each hit: where, peak force, drive current |
| `telemetry()` | robot state, held piece, Toggles, `raised_loaders`, CG height, tilt, moved pieces, where each other robot is |
| `piece_list()` | id, kind and colours of every Pin and Cup |
| `moved_pieces()` | id -> `[x, y, z, ax, ay, az]` for pieces that moved since the last call |
| `stacks()` | what is nested in what |
| `toggles`, `toggle_colors` | the Toggles; each one's colour |
| `match_load(name)`, `lower_loader(name)` | a Drive Team Match Load (below) |
| `grip`, `grips`, `drops` | the held piece; every take and let-go with time and reason |
| `mechanism.lift.motors`, `robot.claw.command(closed)` | the lift's motors; the claw valve |
| `others.reports()` | each other robot: where it is, how far it moved, how far from its path's end, whether it finished |

## Field

| Part | Model | Module |
| --- | --- | --- |
| Walls, Goals, Loaders | fixed MuJoCo shapes; the robot stops on them and the drive motors feel it | `pieces.py` |
| 37 Pins, 36 Cups | free bodies with the CAD's shapes: fall, slide, tip, roll, bounce, nest | `pieces.py`, `override_cad.py` |
| 4 Toggles | bars on hubs in slotted brackets, turned only by contact | `toggles.py` |
| Other robots | drivetrains with a box frame; hold still braking or drive a path | `other_robots.py` |
| Score | Pin halves, Toggle colours, Autonomous Bonus, Win Point | `override_field.py` |

Masses: Pin 72 g, Cup 77 g, weighed. Inertia is the CAD solid's at even
density. Contact constants at the top of `pieces.py` are guesses: friction
0.5 on the floor, 0.3 between pieces, 0.4 on the robot, restitution 0.15.

A hit on a wall or Goal stops the robot in one step, without a bounce.
Spreading the stop over 10 or 20 ms changes odometry error after the hit
from 2.49 in to 2.30 or 2.26 in at 8 V. The wheels spinning against the
wall cause most of that error.

### Toggles

The hubs slide up the bracket slots as the bar turns. Off a face, a corner
presses on a bracket plate and lifts the hubs, so gravity returns the bar to
the nearest face until something pushes it past the corner. Released 50 to
80 degrees round, it wedges and stays. A Toggle scores only while it sits on
a face and nothing on a robot touches it (`<SC4>`); the other robots count.

The bar is a body like the pieces, so loose Pins and Cups hit it as a held one
does. A Pin dropped on a bar lands on it, tips it under a degree, and slides
off into the field.

### Other robots

```python
OtherRobot("opponent 1", x, y, theta, length=18 * INCH, width=18 * INCH, height=18 * INCH,
           mass_lb=15.0, drive="four_motor_450", traction="none", path=(), volts=8.0, start_s=0.0)
```

The role is `"partner"`, `"opponent 1"` or `"opponent 2"`; contact reports
name it "the opponent 1 robot". Its drivetrain is the vexsim preset `drive`,
with `traction` wheels (`"middle"` or `"all"`) if asked, on its own battery,
under a box its size from 0.4 in off the floor. With no `path` it holds still
with its motors on brake. With one, it brakes while the world settles, waits
`start_s` after the Match starts (the world's settle ends), then chases a point
12 in ahead along the path from where it really is, at `volts`, slows over
the last 24 in, and brakes 2 in from the end or once past it
(`other_robots.GUESSES`).

Hit from behind at 8 V by the cascade robot, a 15 lb braking robot on
3.25 in omnis is pushed along; hit from the side it rolls farther on its
omnis and turns. The box cannot reach over a wall, so it never touches a
Toggle.

### Match Loads

`match_load("red right")` raises that Loader and stands a Cup holding a Pin
under it (`<SG11b>`). The tube is then 11.85 to 22.97 in up, so a robot can
drive under it and take the stack from below. Each alliance has 10 Cups, 10
alliance/yellow Pins and 1 yellow/yellow Pin. These raise `ValueError`:

- the alliance has none left;
- a piece already stands under the Loader;
- a robot part lower than a Cup is where the Cup would go.

The world does not check when loading is allowed. `<SG11a>` allows it in the
Driver Controlled Period of a Head-to-Head Match; `<RSC2d>` allows it at any
time in Skills, through the red Loaders.

### Not modelled

- A Toggle's spin after a part drags it.
- Match Loads dropped through a Loader's top.

## Cascade robot (`cascade.py`)

Built like the cascade-lift robots teams built for Override (sources in
`cascade.SOURCES`). Use it with `presets.override_cascade()`: 4 x 11 W blue
on 2.75 in omnis at 450 rpm.

| Part | Model |
| --- | --- |
| Frame | 15 x 12.5 in, 0.4 to 5.0 in off the floor. Wheels 11.5 in apart side to side, 10.5 in front to back. One sprung 2 in tracking wheel at the centre measures sideways (`track_perp`). |
| Bite | round notch in the frame's front, 5.23 in deep, past the front axles. Lips flare to 4.5 in either side of centre. A seated Goal's middle is 1.87 in behind the frame front, under the claw. |
| Lift | five-stage cascade: five 12.9 in rails a side, 8.02 in of travel each, 40.1 in at the claw. Three green motors turn a 1.0 in drum through 48:72. Each string is a damped spring, 77,500 N/m a run. Rubber bands push 8.4 N down to 5.6 N up. Full raise at 12 V: 1.17 s empty. Coasting it creeps down about 0.4 in/s; at 0 V, shorted, about 0.13 in/s. Slide friction 0.2. |
| Claw | pneumatic, over the bite. Each V jaw has a bottom band, a recess and a top band 0.30 in further in. On a stack, the bottom band holds the Cup below its rim and the top band the Pin above its flange. Bottom bands 4.4 in apart open, 3.1 in closed. A 12 mm piston closes in 130 ms and opens in 160 ms, 59 N a jaw at 100 psi. Rubber pads, friction 0.9. Each stroke fills one side of the piston from a 200 mL tank starting at 100 psi; grip scales with the psi left (`claw.air_psi`); 77 strokes reach 25 psi. The tank leaks: sealed, it falls from 100 to 90 psi in an hour (`GUESSES["Air leak"]`). Holds one Pin and one Cup (`<SG6>`). |
| Roller | one green motor through 12:36 chain, 67 rpm. Axle 13.5 in up, 1.75 in in from the back. Two rubber flaps a side, tips 2.5 in from the axle and 0.75 in behind the frame; 3 N bends a tip 30 degrees; friction 0.55. Back on the wall at a Toggle with the roller at 12 V, each flap turns the bar one face, top toward the wall: the first in under 0.65 s, 360 degrees in under 1.6 s. With the drive at 0 V the first flap pushes the robot out of reach. Run backward, it does not turn the Toggle. |
| Mass | 6.35 kg (14.0 lb), measured. VEX's masses for the 8 motors, battery and brain; 0.98 kg of lift that moves; 3.50 kg of frame and the rest, placed to put the CG 3.5 in up with the lift down (`cascade.GUESSES`, "Mass layout"). |
| Tipping | over the nose past 69.4 degrees with the lift down, 46.9 fully up; sideways past 38.9 fully up, where the wheels sink into the foam and the frame edge touches first. Braking from full speed with the lift up lifts the back wheels under 2 degrees. |
| Size and power | `robot.extent(chassis)`: inside 18 in at the start with the preload, 49.97 in tall fully up. `robot.motor_watts()`: 33 W lift, 11 W roller; with the 44 W drive that is the 88 W `<R11>` allows. |

`robot.components(chassis)` lists every solid piece with the lift down and
the claw open, and what each rests on. `robot.parts(chassis)` moves them
with the lift, claw and roller for collisions; mclib's planner draws the
same pieces. `robot.bumper_inside(chassis)` is the inside edge of 1.0 in
bumpers, drawn only. `tests/test_cascade.py` checks that each piece touches
its parent at every lift height and claw gap, and that no two overlap.

Where each number comes from: `cascade.SOURCES` (published), `CHOSEN` (the
robot's owner), `GUESSES` (everything else).

## Field shapes (`override_cad.py`)

The Pins, Cups and Goals collide with shapes cut from VEX's Override field
CAD (276-9250-000, 26 April 2026). Setup is in the [README](../README.md#setup).
Without them, `OverrideWorld` raises `override_cad.MissingCad`.

The CAD is older than manual 2.0. Its Pins, Cups and Goals agree with the
manual within 0.013 in; the Pins and Cups it places stand within 0.028 in
of A12. Where they disagree, vexsim uses the manual:

| | Manual 2.0 | CAD |
| --- | --- | --- |
| Loader width along the wall | 4.02 in | 4.42 in |
| Red Loader, bottom and top | 0 and 14.37 in | 0.31 and 14.68 in |
| Toggle bar length | 25.99 in | 25.84 in |
| Toggle bar across | 2.05 in, "approximately" | 2.03 x 2.19 in |

The manual gives no distance from the field centre to a Toggle's axle.
vexsim uses 70.08 in, from the VEX VR field; the CAD has 71.45 in.
`tests/test_override_cad.py` checks the shapes against the manual.

## MuJoCo checks (`mujoco_checks.py`)

`step()` raises `MujocoUnstable` when MuJoCo finds a NaN, an infinity or a
huge value, naming the body and the time. It raises `MujocoOutOfMemory`
when contacts or constraints overflow the 256 MB arena. MuJoCo alone only
logs these, resets an unstable world and drops what does not fit.

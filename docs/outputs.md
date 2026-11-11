# Outputs

## Log

`sim.log` (also `robot.log`) holds one row per 10 ms (`SimConfig.log_every`).
`log.to_csv(path)` writes every column.

| Group | Columns |
| --- | --- |
| Pose | `t`, `x_m`, `y_m`, `heading_rad`, `x_in`, `y_in`, `heading_deg` |
| Motion | `vel_fwd_mps`, `vel_lat_mps`, `yaw_rate_dps`, `speed_mps`, `speed_inps`, `accel_fwd_mps2`, `accel_lat_mps2`, `accel_fwd_g`, `yaw_accel_dps2` |
| Forces | `force_fwd_N`, `force_lat_N`, `yaw_moment_Nm`, `normal_total_N` |
| Load transfer | `tipping_margin`, `lateral_transfer_ratio`, `pitch_transfer_ratio`, `normal_balance_feasible`, `normal_pitch_residual_Nm`, `normal_roll_residual_Nm`, `any_wheel_slipping` |
| Battery | `battery_V`, `battery_A`, `battery_W`, `battery_soc`, `battery_loss_W`, `battery_rejected_W`, `battery_chemical_W`, `brownout` |
| Power | `motor_current_total_A`, `electrical_W`, `mechanical_W`, `copper_loss_W`, `friction_loss_W`, `gear_loss_W`, `slip_loss_W`, `rolling_loss_W`, `drag_loss_W`, `scrub_loss_W`, `drivetrain_efficiency` |
| Motors, summary | `max_motor_temp_C`, `min_motor_derate` |
| Energy | `kinetic_energy_J`, `energy_battery_J`, `energy_rejected_J`, `energy_slip_J` |
| Sensors | `imu_heading_deg`, `imu_rate_dps`, `imu_valid`, `odom_x_in`, `odom_y_in`, `odom_theta_deg`, `odom_error_in`, `gps_x_in`, `gps_y_in`, `distance_sensor_in`, `distance_sensor_valid` |

`scrub_loss_W` is part of `drag_loss_W`; do not add them.

Per motor, prefix `m_<name>` (`SimConfig.log_per_motor`): `_V`, `_cmdV`,
`_A`, `_Nm`, `_rpm`, `_W_in`, `_W_out`, `_eff`, `_tempC`, `_derate`,
`_ilimited`.

Per wheel, prefix `w_<name>` (`SimConfig.log_per_wheel`): `_rpm`, `_Fz_N`,
`_Fx_N`, `_Fy_N`, `_slip_ratio`, `_slip_angle_deg`, `_surface_mps`,
`_ground_mps`, `_slipping`, `_slip_measure`, `_torque_Nm`.

## Reports

`robot.report()` returns a dict; `robot.print_report()` prints it.

| Key | |
| --- | --- |
| `time_s`, `timed_out` | duration of the last move; whether it hit its timeout |
| `true_x_in`, `true_y_in`, `true_heading_deg` | where the robot is |
| `odom_x_in`, `odom_y_in`, `odom_heading_deg`, `odom_position_error_in`, `imu_heading_error_deg` | where its sensors say it is |
| `peak_speed_inps`, `peak_accel_g`, `peak_current_A`, `peak_power_W`, `avg_power_W` | peaks over the run |
| `min_battery_V`, `battery_mAh_used`, `hottest_motor_C` | battery and heat |
| `min_tipping_margin`, `wheel_slip_fraction`, `slip_energy_J` | traction |

`sim.energy_report()` splits the energy drawn from the battery, in joules:
`battery_J`, `battery_rejected_J`, `copper_J`, `motor_friction_J`,
`external_gear_J`, `wheel_shaft_J`, `tire_slip_J`, `rolling_J`,
`chassis_drag_J`, and `unaccounted_J` for what the ledger misses
(inductance, finite-step error). Also `battery_mAh`. Tire elastic storage is
not in the ledger.

## Contacts

`world.reported_contacts()` on an `OverrideWorld` or `arena.open_floor()`
world lists each hit: the part, the obstacle, where, peak force and drive
current.

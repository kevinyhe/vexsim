"""V5 battery model.

The V5 battery is a 4-cell LiFePO4 pack, 12.8 V nominal, ~1100 mAh.
Two things about it matter to a drivetrain sim:

1. Internal resistance.  Six motors pulling 2.5 A each is 15 A through roughly
   35 milliohms, which is half a volt of sag.  Sag lowers the voltage every
   motor sees, which lowers top speed and torque, which is why a robot that is
   quick on a fresh battery is slow on a flat one.
2. State of charge.  Open-circuit voltage falls as the pack empties.  A pack at
   30% is about 0.7 V lower than a full one before any sag at all.

The open-circuit curve and internal resistance are approximate assumptions,
not a measured pack identification. The finite source disconnects on empty
charge; terminal voltage and charge bookkeeping cannot create extra capacity.
Excess regeneration into a full pack is explicitly dissipated by an idealized
protection shunt. This conserves energy without claiming a measured V5 BMS
topology; power_rejected and energy_rejected expose that approximation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .units import clamp

# (state of charge 0-1, open circuit volts)
_DEFAULT_OCV: List[Tuple[float, float]] = [
    (0.00, 11.20),
    (0.05, 11.85),
    (0.10, 12.10),
    (0.20, 12.42),
    (0.30, 12.63),
    (0.40, 12.80),
    (0.50, 12.96),
    (0.60, 13.15),
    (0.70, 13.36),
    (0.80, 13.60),
    (0.90, 13.92),
    (1.00, 14.40),
]


@dataclass
class Battery:
    """A charge-tracking voltage source with internal resistance."""

    capacity_ah: float = 1.1            # 1100 mAh
    internal_resistance: float = 0.035  # ohms, whole pack including wiring
    soc: float = 1.0                    # state of charge, 0-1
    ocv_curve: List[Tuple[float, float]] = field(default_factory=lambda: list(_DEFAULT_OCV))
    min_voltage: float = 7.0            # brain browns out below roughly this

    # --- live outputs, refreshed every step ---
    voltage: float = 14.4               # terminal voltage under load
    current: float = 0.0                # total pack current, A
    power_out: float = 0.0              # W delivered to the loads
    power_loss: float = 0.0             # W burned in internal resistance
    power_rejected: float = 0.0         # W dumped because the pack is already full
    power_chemical: float = 0.0         # W leaving stored charge (negative charging)
    energy_rejected: float = 0.0        # J dissipated by the idealized regen shunt
    charge_used_ah: float = 0.0
    brownout: bool = False

    def __post_init__(self) -> None:
        self.soc = clamp(self.soc, 0.0, 1.0)
        self.voltage = self.terminal_voltage(0.0)
        self.brownout = self.voltage < self.min_voltage

    def open_circuit_voltage(self) -> float:
        soc = clamp(self.soc, 0.0, 1.0)
        curve = self.ocv_curve
        if soc <= curve[0][0]:
            return curve[0][1]
        for i in range(1, len(curve)):
            s0, v0 = curve[i - 1]
            s1, v1 = curve[i]
            if soc <= s1:
                t = (soc - s0) / (s1 - s0)
                return v0 + t * (v1 - v0)
        return curve[-1][1]

    def terminal_voltage(self, current: float) -> float:
        """Voltage available to the loads when they draw `current` amps."""
        if self.soc <= 0.0 and current >= 0.0:
            return 0.0
        return max(0.0, self.open_circuit_voltage() - current * self.internal_resistance)

    def discharge_current_limit(self, dt: float) -> float:
        """Largest average discharge current that this step's charge can fund."""
        charge_limit = max(0.0, self.soc * self.capacity_ah * 3600.0 / dt)
        short_circuit = (self.open_circuit_voltage() / self.internal_resistance
                         if self.internal_resistance > 0.0 else float("inf"))
        return min(charge_limit, short_circuit)

    def update(self, current: float, dt: float) -> None:
        """Book-keep one timestep at a known total draw."""
        if dt <= 0.0:
            raise ValueError("battery timestep must be positive")
        current = min(current, self.discharge_current_limit(dt))
        self.current = current
        ocv = self.open_circuit_voltage()
        self.voltage = self.terminal_voltage(current)
        self.power_loss = current * current * self.internal_resistance
        self.power_out = self.voltage * current
        self.brownout = self.voltage < self.min_voltage

        used = current * dt / 3600.0   # signed terminal amp-hours
        stored_used = max(used, -(1.0 - self.soc) * self.capacity_ah)
        self.power_rejected = ocv * (stored_used - used) * 3600.0 / dt
        self.power_chemical = ocv * stored_used * 3600.0 / dt
        self.energy_rejected += self.power_rejected * dt
        self.charge_used_ah += stored_used
        self.soc = clamp(self.soc - stored_used / self.capacity_ah, 0.0, 1.0)

    def reset(self, soc: float = 1.0) -> None:
        self.soc = clamp(soc, 0.0, 1.0)
        self.charge_used_ah = 0.0
        self.current = 0.0
        self.voltage = self.terminal_voltage(0.0)
        self.power_out = 0.0
        self.power_loss = 0.0
        self.power_rejected = 0.0
        self.power_chemical = 0.0
        self.energy_rejected = 0.0
        self.brownout = self.voltage < self.min_voltage


@dataclass
class InfiniteBattery(Battery):
    """A bench supply: no sag, no discharge.  Useful for isolating effects."""

    internal_resistance: float = 0.0

    def open_circuit_voltage(self) -> float:
        return 12.0

    def terminal_voltage(self, current: float) -> float:
        return 12.0

    def discharge_current_limit(self, dt: float) -> float:
        return float("inf")

    def update(self, current: float, dt: float) -> None:
        self.current = current
        self.voltage = 12.0
        self.power_out = 12.0 * current
        self.power_loss = 0.0
        self.power_rejected = 0.0
        self.power_chemical = self.power_out
        self.brownout = False

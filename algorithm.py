"""
VPP Orchestration Engine — Algorithm
=====================================
Core charge/discharge decision logic for distributed energy resources.

Decision Rules:
- Price HIGH + Battery has charge  → Discharge to grid (earn money)
- Price LOW  + Renewable available → Charge batteries (store cheap energy)
- Price MED  + Load high           → Discharge partially
- Always maintain minimum 20% battery reserve
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# ── Thresholds (tune these based on your grid) ────────────────────────────────
PRICE_HIGH    = 8.0   # Rs/kWh — above this, sell to grid
PRICE_LOW     = 3.0   # Rs/kWh — below this, buy from grid
MIN_SOC       = 0.20  # 20% minimum state of charge
MAX_SOC       = 0.95  # 95% maximum state of charge
MAX_CHARGE_RATE   = 5.0  # kW per battery unit
MAX_DISCHARGE_RATE = 5.0  # kW per battery unit


@dataclass
class GridState:
    """Current state of the grid and environment."""
    price: float          # Rs/kWh — current electricity price
    solar_forecast: float # 0.0 to 1.0 — solar generation forecast
    wind_forecast: float  # 0.0 to 1.0 — wind generation forecast
    local_load: float     # kW — current local demand
    frequency: float      # Hz — grid frequency (nominal 50 Hz in India)
    voltage: float        # V — grid voltage (nominal 230 V)


@dataclass
class BatteryUnit:
    """Represents one home battery unit."""
    id: str
    capacity_kwh: float   # Total capacity in kWh
    soc: float            # State of charge 0.0 to 1.0
    max_charge_kw: float  # Max charge rate
    max_discharge_kw: float  # Max discharge rate
    online: bool = True

    @property
    def energy_available(self) -> float:
        """kWh available above minimum reserve."""
        return max(0, (self.soc - MIN_SOC) * self.capacity_kwh)

    @property
    def energy_headroom(self) -> float:
        """kWh that can still be charged."""
        return max(0, (MAX_SOC - self.soc) * self.capacity_kwh)


@dataclass
class Decision:
    """Control decision for a battery unit."""
    battery_id: str
    action: str       # 'charge', 'discharge', 'idle'
    power_kw: float   # kW — positive = charge, negative = discharge
    reason: str       # Human-readable explanation


class VPPOrchestrator:
    """
    Main orchestration algorithm.
    
    Takes grid state + battery fleet → outputs control decisions.
    """

    def __init__(self, batteries: List[BatteryUnit]):
        self.batteries = batteries
        self.total_decisions = 0
        self.total_revenue = 0.0  # Rs earned from grid

    def decide(self, state: GridState) -> List[Decision]:
        """
        Main decision function — called every control cycle (e.g. every 5 min).
        Returns list of decisions for each battery.
        """
        decisions = []
        renewable_available = (state.solar_forecast + state.wind_forecast) / 2

        for battery in self.batteries:
            if not battery.online:
                decisions.append(Decision(battery.id, 'idle', 0, 'Offline'))
                continue

            decision = self._decide_single(battery, state, renewable_available)
            decisions.append(decision)

        self.total_decisions += 1
        return decisions

    def _decide_single(self, battery: BatteryUnit, state: GridState, renewable: float) -> Decision:
        """Decision logic for one battery unit."""

        # ── FREQUENCY RESPONSE (highest priority) ────────────────────────────
        # Grid frequency drops → discharge immediately to stabilise
        if state.frequency < 49.8 and battery.energy_available > 0:
            power = min(battery.max_discharge_kw, battery.energy_available * 12)
            return Decision(battery.id, 'discharge', -power,
                          f'Freq response: {state.frequency:.2f} Hz < 49.8 Hz')

        # Grid frequency high → absorb excess by charging
        if state.frequency > 50.2 and battery.energy_headroom > 0:
            power = min(battery.max_charge_kw, battery.energy_headroom * 12)
            return Decision(battery.id, 'charge', power,
                          f'Freq absorption: {state.frequency:.2f} Hz > 50.2 Hz')

        # ── PRICE-BASED ARBITRAGE ─────────────────────────────────────────────
        # High price → discharge and sell
        if state.price >= PRICE_HIGH and battery.energy_available > 0:
            power = min(battery.max_discharge_kw, battery.energy_available * 4)
            revenue = power * state.price / 12  # Rs earned this cycle
            self.total_revenue += revenue
            return Decision(battery.id, 'discharge', -power,
                          f'High price Rs{state.price:.1f}/kWh — selling to grid')

        # Low price + renewable available → charge
        if state.price <= PRICE_LOW and battery.energy_headroom > 0 and renewable > 0.3:
            power = min(battery.max_charge_kw, battery.energy_headroom * 4)
            return Decision(battery.id, 'charge', power,
                          f'Cheap price Rs{state.price:.1f}/kWh + renewable {renewable:.0%}')

        # ── LOAD BALANCING ────────────────────────────────────────────────────
        # Local load very high → help by discharging
        if state.local_load > 8.0 and battery.energy_available > 0:
            power = min(battery.max_discharge_kw * 0.5, battery.energy_available * 2)
            return Decision(battery.id, 'discharge', -power,
                          f'Peak load support: {state.local_load:.1f} kW')

        # Lots of solar → charge to absorb excess
        if state.solar_forecast > 0.8 and battery.energy_headroom > 0:
            power = min(battery.max_charge_kw * 0.7, battery.energy_headroom * 3)
            return Decision(battery.id, 'charge', power,
                          f'Solar absorption: {state.solar_forecast:.0%} forecast')

        return Decision(battery.id, 'idle', 0, 'Conditions nominal')

    def get_fleet_summary(self) -> dict:
        """Summary stats for the whole battery fleet."""
        online = [b for b in self.batteries if b.online]
        total_cap = sum(b.capacity_kwh for b in online)
        avg_soc = np.mean([b.soc for b in online]) if online else 0
        total_energy = sum(b.energy_available for b in online)

        return {
            'total_units': len(self.batteries),
            'online_units': len(online),
            'total_capacity_kwh': round(total_cap, 1),
            'avg_soc_percent': round(avg_soc * 100, 1),
            'available_energy_kwh': round(total_energy, 1),
            'total_revenue_rs': round(self.total_revenue, 2),
        }

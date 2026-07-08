"""
Generator data model and synthetic grid builders for FluxQ.

A Generator is the fundamental unit of the Unit Commitment Problem.
Each generator is modelled as a binary on/off decision variable per timestep,
dispatching at full rated capacity (P_max) when committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class Generator:
    """
    A single dispatchable generator unit.

    In the binary UC model, a generator either produces P_max MW (ON)
    or 0 MW (OFF) at each timestep. The extended model (Phase 1+) adds
    minimum-stable-generation (P_min) and ramp-rate limits.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. 'coal_01', 'gas_peaker_03'.
    P_max : float
        Rated power output (MW). This is the dispatch level when ON.
    c_fuel : float
        Variable/fuel cost rate ($/MWh).
    SU_cost : float
        Startup cost ($) — charged when transitioning OFF → ON.
    SD_cost : float
        Shutdown cost ($) — charged when transitioning ON → OFF.
    P_min : float
        Minimum stable generation (MW). Informational for binary model.
    MUT : int
        Minimum up time (timesteps). Generator must stay ON for at least
        MUT steps after starting. Default 1 (no constraint beyond binary).
    MDT : int
        Minimum down time (timesteps). Default 1.
    ramp_rate : float
        Maximum ramp rate (MW/timestep). Default inf (unconstrained).
    initial_state : int
        Commitment state at t = -1 (before planning horizon). 1=ON, 0=OFF.
    """

    name: str
    P_max: float
    c_fuel: float
    SU_cost: float
    SD_cost: float
    P_min: float = 0.0
    MUT: int = 1
    MDT: int = 1
    ramp_rate: float = float("inf")
    initial_state: int = 0

    def __post_init__(self) -> None:
        if self.P_max <= 0:
            raise ValueError(f"'{self.name}': P_max must be > 0, got {self.P_max}")
        if self.c_fuel < 0:
            raise ValueError(f"'{self.name}': c_fuel must be >= 0, got {self.c_fuel}")
        if self.SU_cost < 0:
            raise ValueError(f"'{self.name}': SU_cost must be >= 0, got {self.SU_cost}")
        if self.SD_cost < 0:
            raise ValueError(f"'{self.name}': SD_cost must be >= 0, got {self.SD_cost}")
        if self.MUT < 1:
            raise ValueError(f"'{self.name}': MUT must be >= 1, got {self.MUT}")
        if self.MDT < 1:
            raise ValueError(f"'{self.name}': MDT must be >= 1, got {self.MDT}")
        if self.initial_state not in (0, 1):
            raise ValueError(f"'{self.name}': initial_state must be 0 or 1")


# ---------------------------------------------------------------------------
# Pre-built generator sets
# ---------------------------------------------------------------------------

def make_toy_generators() -> List[Generator]:
    """
    3-generator toy system for Phase 0 brute-force verification.

    Inspired by the classic textbook UC example:
      - Coal baseload: cheap fuel, slow and expensive to start
      - Gas combined-cycle: mid-merit
      - Gas peaker: expensive fuel, fast and cheap to start

    Total capacity: 450 MW. Typical peak demand: 300–350 MW.
    """
    return [
        Generator(
            name="coal_baseload",
            P_max=200.0,    # MW
            c_fuel=20.0,    # $/MWh — cheapest fuel
            SU_cost=500.0,  # $ — expensive to start (cold start)
            SD_cost=300.0,  # $
            MUT=3,
            MDT=2,
        ),
        Generator(
            name="gas_combined",
            P_max=150.0,
            c_fuel=40.0,    # $/MWh — mid-merit
            SU_cost=200.0,
            SD_cost=100.0,
            MUT=2,
            MDT=1,
        ),
        Generator(
            name="gas_peaker",
            P_max=100.0,
            c_fuel=80.0,    # $/MWh — most expensive
            SU_cost=100.0,  # $ — cheapest to start (fast response)
            SD_cost=50.0,
            MUT=1,
            MDT=1,
        ),
    ]


def make_ieee14_generators() -> List[Generator]:
    """
    Approximate generator parameters for the IEEE 14-bus test case.

    Five generators matching the network topology of IEEE 14-bus.
    These are approximate defaults for testing without pandapower.
    For exact values, use ``loader.load_ieee14_generators()``.

    Bus mapping: G1 (bus 1, slack), G2 (bus 2), G3 (bus 3),
                 G6 (bus 6), G8 (bus 8).
    """
    return [
        Generator(
            name="G1_slack_bus1",
            P_max=332.0,
            c_fuel=20.0,
            SU_cost=400.0,
            SD_cost=200.0,
            MUT=4,
            MDT=2,
        ),
        Generator(
            name="G2_bus2",
            P_max=140.0,
            c_fuel=30.0,
            SU_cost=200.0,
            SD_cost=100.0,
            MUT=2,
            MDT=1,
        ),
        Generator(
            name="G3_bus3",
            P_max=100.0,
            c_fuel=40.0,
            SU_cost=150.0,
            SD_cost=75.0,
            MUT=2,
            MDT=1,
        ),
        Generator(
            name="G6_bus6",
            P_max=100.0,
            c_fuel=35.0,
            SU_cost=150.0,
            SD_cost=75.0,
            MUT=2,
            MDT=1,
        ),
        Generator(
            name="G8_bus8",
            P_max=100.0,
            c_fuel=45.0,
            SU_cost=120.0,
            SD_cost=60.0,
            MUT=1,
            MDT=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Demand profile utilities
# ---------------------------------------------------------------------------

def make_demand_profile(
    T: int,
    peak_demand: float,
    profile: str = "flat",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a demand profile for T timesteps.

    Parameters
    ----------
    T : int
        Number of timesteps.
    peak_demand : float
        Peak demand (MW). Returned profile never exceeds this value.
    profile : str
        One of:
          'flat'   — constant at 80% of peak
          'daily'  — sinusoidal day cycle (low at night, peaks at ~18:00)
          'step'   — low in first/last quarter, high in middle
          'random' — random walk around 70% of peak (seed for reproducibility)
    seed : int, optional
        Random seed for the 'random' profile.

    Returns
    -------
    np.ndarray, shape (T,)
        Demand in MW at each timestep.
    """
    if profile == "flat":
        return np.full(T, peak_demand * 0.80)

    elif profile == "daily":
        # Smooth daily load curve: trough ~3AM, peaks ~8AM and ~6PM
        t = np.linspace(0, 24, T, endpoint=False)
        morning_peak = np.exp(-0.5 * ((t - 8) / 2) ** 2)
        evening_peak = np.exp(-0.5 * ((t - 18) / 2) ** 2)
        demand = peak_demand * (0.55 + 0.25 * morning_peak + 0.20 * evening_peak)
        return np.clip(demand, 0.40 * peak_demand, peak_demand)

    elif profile == "step":
        demand = np.full(T, peak_demand * 0.60)
        demand[T // 4: 3 * T // 4] = peak_demand * 0.90
        return demand

    elif profile == "random":
        rng = np.random.default_rng(seed)
        demand = np.full(T, peak_demand * 0.70)
        walk = np.cumsum(rng.normal(0, peak_demand * 0.03, T))
        demand = demand + walk
        return np.clip(demand, 0.40 * peak_demand, peak_demand)

    else:
        valid = ["flat", "daily", "step", "random"]
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {valid}")

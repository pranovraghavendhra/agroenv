"""
Soil Water Balance Model
========================
Implements the FAO-56 single layer soil water balance.
Tracks root zone depletion, actual vs potential ET, and drainage.

Reference: Allen et al. (1998), FAO-56, Chapter 8.
"""

import json
import os
from dataclasses import dataclass, field


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "soils.json")


@dataclass
class SoilState:
    moisture_pct: float            # Current volumetric water content %
    depletion_mm: float            # Root zone depletion from field capacity
    drainage_mm_today: float       # Drainage below root zone today
    runoff_mm_today: float         # Surface runoff today
    cumulative_stress_days: int    # Days below critical moisture threshold
    waterlog_days: int             # Days above waterlogging threshold
    ks: float                      # Water stress coefficient (0=full stress, 1=no stress)
    raw_mm: float                  # Readily Available Water in mm


class SoilModel:
    """
    FAO-56 single-layer root zone water balance.
    Tracks depletion, stress, drainage, and runoff.
    """

    def __init__(self, soil_key: str = "loamy_soil", root_depth_m: float = 0.5):
        with open(DATA_PATH, "r") as f:
            all_soils = json.load(f)

        if soil_key not in all_soils:
            soil_key = "loamy_soil"

        self.soil = all_soils[soil_key]
        self.root_depth_m = root_depth_m

        # Soil hydraulic properties
        self.theta_fc = self.soil["field_capacity_pct"]     # Field capacity %
        self.theta_wp = self.soil["wilting_point_pct"]      # Wilting point %
        self.theta_sat = self.soil["saturation_pct"]        # Saturation %
        self.infiltration_rate = self.soil["infiltration_rate_mm_per_hr"]
        self.depth_mm = root_depth_m * 1000.0               # Root depth in mm

        # Total available water (TAW) in mm
        self.taw_mm = (self.theta_fc - self.theta_wp) / 100.0 * self.depth_mm

        # Depletion fraction for stress (p) — typical 0.5 for most crops
        self.p_depletion = 0.50

        # State
        self._moisture_pct: float = self.theta_fc  # Start at field capacity
        self._depletion_mm: float = 0.0
        self._cumulative_stress_days: int = 0
        self._waterlog_days: int = 0

    def reset(self, initial_moisture_pct: float = None) -> SoilState:
        """Reset soil to initial conditions."""
        if initial_moisture_pct is None:
            initial_moisture_pct = self.theta_fc
        self._moisture_pct = max(self.theta_wp, min(self.theta_sat, initial_moisture_pct))
        self._depletion_mm = max(0.0, (self.theta_fc - self._moisture_pct) / 100.0 * self.depth_mm)
        self._cumulative_stress_days = 0
        self._waterlog_days = 0
        return self._build_state()

    def update(
        self,
        et0_mm: float,
        kc: float,
        rainfall_mm: float,
        irrigation_mm: float,
    ) -> tuple[SoilState, dict]:
        """
        Advance soil water balance by one day.

        Args:
            et0_mm: Reference ET (FAO-56 Penman-Monteith)
            kc: Crop coefficient for current growth stage
            rainfall_mm: Daily rainfall
            irrigation_mm: Irrigation applied today

        Returns:
            (SoilState, info_dict)
        """
        # Potential crop ET
        etc_mm = et0_mm * kc

        # Water input: rainfall + irrigation
        infiltration_mm, runoff_mm = self._compute_infiltration(rainfall_mm, irrigation_mm)
        total_input_mm = infiltration_mm

        # Compute water stress coefficient (Ks) — FAO-56 Eq. 84
        raw_mm = self.p_depletion * self.taw_mm
        if self._depletion_mm <= raw_mm:
            ks = 1.0  # No stress
        elif self._depletion_mm >= self.taw_mm:
            ks = 0.0  # Full stress
        else:
            ks = (self.taw_mm - self._depletion_mm) / ((1 - self.p_depletion) * self.taw_mm)
            ks = max(0.0, min(1.0, ks))

        # Actual ET (reduced by stress)
        actual_et_mm = etc_mm * ks

        # Update depletion
        self._depletion_mm = self._depletion_mm + actual_et_mm - total_input_mm
        self._depletion_mm = max(0.0, min(self.taw_mm * 1.5, self._depletion_mm))

        # Compute drainage (excess above field capacity)
        drainage_mm = 0.0
        current_water_mm = (self.theta_fc / 100.0 * self.depth_mm) - self._depletion_mm
        max_water_mm = self.theta_sat / 100.0 * self.depth_mm
        if current_water_mm > self.theta_fc / 100.0 * self.depth_mm:
            excess = current_water_mm - self.theta_fc / 100.0 * self.depth_mm
            drainage_mm = min(excess, self.infiltration_rate * 24 * 0.1)
            self._depletion_mm = max(0.0, self._depletion_mm + drainage_mm)

        # Update moisture percentage
        depletion_pct = self._depletion_mm / self.depth_mm * 100.0
        self._moisture_pct = max(self.theta_wp, self.theta_fc - depletion_pct)
        self._moisture_pct = min(self.theta_sat, self._moisture_pct)

        # Track stress days
        critical_moisture = self.soil.get("critical_moisture_min_pct",
                                          self.theta_wp + (self.theta_fc - self.theta_wp) * 0.3)
        waterlog_threshold = self.soil.get("waterlog_threshold_pct",
                                           self.theta_fc + (self.theta_sat - self.theta_fc) * 0.8)

        if self._moisture_pct < critical_moisture:
            self._cumulative_stress_days += 1
        if self._moisture_pct > waterlog_threshold:
            self._waterlog_days += 1

        info = {
            "etc_mm": round(etc_mm, 2),
            "actual_et_mm": round(actual_et_mm, 2),
            "infiltration_mm": round(infiltration_mm, 2),
            "runoff_mm": round(runoff_mm, 2),
            "drainage_mm": round(drainage_mm, 2),
            "ks": round(ks, 3),
        }

        return self._build_state(drainage_mm, runoff_mm, ks, raw_mm), info

    def get_irrigation_recommendation_mm(self, et0_mm: float, kc: float) -> float:
        """
        Compute scientifically correct irrigation to refill to field capacity.
        This is the 'oracle' value used by the grader to assess agent decisions.
        """
        etc_mm = et0_mm * kc
        # Irrigate when depletion exceeds RAW threshold
        raw_mm = self.p_depletion * self.taw_mm
        if self._depletion_mm > raw_mm:
            # Refill to field capacity
            net_irrigation = self._depletion_mm
            # Add 15% for distribution uniformity (standard practice)
            gross_irrigation = net_irrigation / 0.85
            return round(max(0.0, min(gross_irrigation, 80.0)), 1)
        return 0.0

    def _compute_infiltration(self, rainfall_mm: float, irrigation_mm: float) -> tuple[float, float]:
        """Simple CN-based runoff estimation."""
        total_input = rainfall_mm + irrigation_mm
        max_infiltration_per_day = self.infiltration_rate * 24
        if total_input <= max_infiltration_per_day:
            return total_input, 0.0
        else:
            runoff = total_input - max_infiltration_per_day
            return max_infiltration_per_day, runoff

    def _build_state(
        self,
        drainage_mm: float = 0.0,
        runoff_mm: float = 0.0,
        ks: float = None,
        raw_mm: float = None
    ) -> SoilState:
        if ks is None:
            raw_mm = self.p_depletion * self.taw_mm
            if self._depletion_mm <= raw_mm:
                ks = 1.0
            elif self._depletion_mm >= self.taw_mm:
                ks = 0.0
            else:
                ks = (self.taw_mm - self._depletion_mm) / ((1 - self.p_depletion) * self.taw_mm)
                ks = max(0.0, min(1.0, ks))

        return SoilState(
            moisture_pct=round(self._moisture_pct, 2),
            depletion_mm=round(self._depletion_mm, 2),
            drainage_mm_today=round(drainage_mm, 2),
            runoff_mm_today=round(runoff_mm, 2),
            cumulative_stress_days=self._cumulative_stress_days,
            waterlog_days=self._waterlog_days,
            ks=round(ks, 3),
            raw_mm=round(raw_mm if raw_mm is not None else self.p_depletion * self.taw_mm, 2),
        )

"""
Crop Growth Model
==================
Implements:
- Growing Degree Day (GDD) accumulation
- Growth stage tracking
- Leaf Area Index (LAI) development
- NDVI estimation from LAI (Beer-Lambert law)
- Harvest maturity window
- Yield estimation with water/pest stress penalties

References:
- FAO-56 Allen et al. (1998)
- DSSAT crop model principles (Jones et al., 2003)
- NDVI-LAI relationship: Baret & Guyot (1991)
"""

import json
import os
import math
from dataclasses import dataclass
from typing import Optional


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crops.json")


@dataclass
class CropState:
    day: int
    growth_stage: str
    gdd_accumulated: float
    gdd_required_total: float
    gdd_progress_pct: float
    lai: float                     # Leaf Area Index (m²/m²)
    ndvi: float                    # Normalized Difference Vegetation Index (0–1)
    kc: float                      # Crop coefficient for ET calculation
    canopy_cover_pct: float        # Fraction of ground covered (%)
    biomass_relative: float        # Relative biomass accumulation (0–1)
    cumulative_water_stress: float # Integrated Ks deficit (0=no stress, high=severe)
    estimated_yield_pct: float     # Estimated final yield as % of maximum
    days_to_harvest_window: int    # Days until harvest window opens
    in_harvest_window: bool        # Whether GDD is within harvest window
    harvest_window_closing_days: int  # Days until window closes (0 = closed)


class CropModel:
    """
    Tracks crop development from sowing to harvest.
    Uses GDD for phenological development and LAI for canopy estimation.
    """

    def __init__(self, crop_key: str = "rice_kharif"):
        with open(DATA_PATH, "r") as f:
            all_crops = json.load(f)

        if crop_key not in all_crops:
            crop_key = "rice_kharif"

        self.crop = all_crops[crop_key]
        self.crop_key = crop_key
        self.stages = self.crop["growth_stages"]
        self.gdd_base = self.crop["gdd_base_temp_c"]
        self.gdd_total = self.crop["gdd_total_required"]
        self.gdd_window = self.crop["gdd_harvest_window"]
        self.max_yield = self.crop["max_yield_ton_per_ha"]
        self.total_days = self.crop["total_days"]

        # State
        self._day: int = 0
        self._gdd_accumulated: float = 0.0
        self._cumulative_water_stress: float = 0.0
        self._cumulative_pest_damage: float = 0.0
        self._stage_name: str = list(self.stages.keys())[0]

    def reset(self) -> CropState:
        """Reset crop to beginning of season."""
        self._day = 0
        self._gdd_accumulated = 0.0
        self._cumulative_water_stress = 0.0
        self._cumulative_pest_damage = 0.0
        self._stage_name = list(self.stages.keys())[0]
        return self._build_state()

    def update(
        self,
        tmax_c: float,
        tmin_c: float,
        ks: float,          # Soil water stress coefficient
        pest_damage_pct: float = 0.0,
    ) -> tuple[CropState, dict]:
        """
        Advance crop development by one day.

        Args:
            tmax_c: Maximum temperature
            tmin_c: Minimum temperature
            ks: Water stress coefficient from soil model (0–1)
            pest_damage_pct: Accumulated pest damage today

        Returns:
            (CropState, info_dict)
        """
        self._day += 1

        # GDD accumulation (modified to account for heat stress above ceiling)
        tmean = (tmax_c + tmin_c) / 2.0
        gdd_today = max(0.0, tmean - self.gdd_base)

        # GDD ceiling — extreme heat reduces effective GDD
        tmax_ceiling = 35.0  # Most Indian crops have reduced accumulation above this
        if tmax_c > tmax_ceiling:
            gdd_today *= max(0.3, 1.0 - (tmax_c - tmax_ceiling) * 0.05)

        # Water stress slows GDD accumulation slightly
        gdd_today *= (0.7 + 0.3 * ks)

        self._gdd_accumulated += gdd_today

        # Update growth stage
        self._stage_name = self._get_stage(self._day)
        kc = self.stages[self._stage_name]["kc"]

        # Water stress accumulation
        stress_today = max(0.0, 1.0 - ks)
        self._cumulative_water_stress += stress_today * 0.1  # scaled

        # Pest damage
        self._cumulative_pest_damage += pest_damage_pct * 0.01

        info = {
            "gdd_today": round(gdd_today, 2),
            "stress_today": round(stress_today, 3),
        }

        return self._build_state(), info

    def get_kc(self) -> float:
        """Current crop coefficient."""
        return self.stages[self._stage_name]["kc"]

    def estimate_final_yield(self) -> float:
        """
        Estimate final yield in tonnes/ha considering accumulated stresses.
        Uses Jensen multiplicative stress model (simplified).
        """
        water_stress_factor = max(0.0, 1.0 - self._cumulative_water_stress * 0.15)
        pest_factor = max(0.0, 1.0 - self._cumulative_pest_damage)

        # Stage-weighted stress (flowering/heading stress is most damaging)
        water_stress_factor = min(1.0, water_stress_factor)
        yield_fraction = water_stress_factor * pest_factor
        yield_fraction = max(0.0, min(1.0, yield_fraction))
        return round(self.max_yield * yield_fraction, 2)

    def _get_stage(self, day: int) -> str:
        """Determine growth stage from day number."""
        current_stage = list(self.stages.keys())[0]
        for stage_name, stage_data in self.stages.items():
            if stage_data["start_day"] <= day <= stage_data["end_day"]:
                return stage_name
            if day > stage_data["end_day"]:
                current_stage = stage_name
        return current_stage

    def _compute_lai(self) -> float:
        """
        Leaf Area Index estimation based on crop development stage.
        Follows a logistic growth curve peaking at heading/flowering.
        """
        stage_data = self.stages[self._stage_name]
        stages_list = list(self.stages.keys())
        stage_idx = stages_list.index(self._stage_name)
        n_stages = len(stages_list)

        # Peak LAI at ~60% of season (heading stage for most crops)
        peak_stage_fraction = 0.6
        current_fraction = stage_idx / max(1, n_stages - 1)

        if current_fraction <= peak_stage_fraction:
            # Rising phase: logistic growth
            x = current_fraction / peak_stage_fraction * 6 - 3
            lai = 5.5 / (1 + math.exp(-x))
        else:
            # Senescence phase: linear decline
            decline = (current_fraction - peak_stage_fraction) / (1 - peak_stage_fraction)
            lai = 5.5 * (1 - decline * 0.85)

        # Water stress reduces LAI
        if self._cumulative_water_stress > 0:
            lai *= max(0.5, 1.0 - self._cumulative_water_stress * 0.1)

        return round(max(0.1, min(6.0, lai)), 2)

    def _lai_to_ndvi(self, lai: float) -> float:
        """
        Convert LAI to NDVI using Beer-Lambert extinction law.
        NDVI = NDVImax * (1 - exp(-k * LAI))
        k = extinction coefficient ~0.5 for most crops.
        Reference: Baret & Guyot (1991), Remote Sensing of Environment.
        """
        ndvi_max = 0.92
        k = 0.50  # extinction coefficient
        ndvi = ndvi_max * (1 - math.exp(-k * lai))
        return round(max(0.0, min(1.0, ndvi)), 3)

    def _build_state(self) -> CropState:
        lai = self._compute_lai()
        ndvi = self._lai_to_ndvi(lai)
        kc = self.stages[self._stage_name]["kc"]
        canopy_cover = min(100.0, lai / 6.0 * 100.0)
        biomass_rel = min(1.0, self._gdd_accumulated / self.gdd_total)

        # Harvest window calculation
        gdd_window_open = self.gdd_window[0]
        gdd_window_close = self.gdd_window[1]
        in_window = gdd_window_open <= self._gdd_accumulated <= gdd_window_close

        if self._gdd_accumulated < gdd_window_open:
            # Estimate days to window based on current GDD rate
            remaining_gdd = gdd_window_open - self._gdd_accumulated
            remaining_days = max(0, self.total_days - self._day)
            total_remaining_gdd = max(1, self.gdd_total - self._gdd_accumulated)
            days_to_window = int(remaining_gdd / max(1, total_remaining_gdd) * max(1, remaining_days))
        else:
            days_to_window = 0

        if in_window:
            gdd_remaining_in_window = gdd_window_close - self._gdd_accumulated
            # Rough estimate: 10 GDD/day average
            window_close_days = max(0, int(gdd_remaining_in_window / 10))
        elif self._gdd_accumulated > gdd_window_close:
            window_close_days = 0
        else:
            window_close_days = self.total_days - self._day

        estimated_yield_pct = round(self.estimate_final_yield() / self.max_yield * 100, 1)

        return CropState(
            day=self._day,
            growth_stage=self._stage_name,
            gdd_accumulated=round(self._gdd_accumulated, 1),
            gdd_required_total=self.gdd_total,
            gdd_progress_pct=round(self._gdd_accumulated / self.gdd_total * 100, 1),
            lai=lai,
            ndvi=ndvi,
            kc=kc,
            canopy_cover_pct=round(canopy_cover, 1),
            biomass_relative=round(biomass_rel, 3),
            cumulative_water_stress=round(self._cumulative_water_stress, 3),
            estimated_yield_pct=estimated_yield_pct,
            days_to_harvest_window=days_to_window,
            in_harvest_window=in_window,
            harvest_window_closing_days=window_close_days,
        )

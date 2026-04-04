"""
IPM Pest Population Dynamics Model
====================================
Implements Integrated Pest Management thresholds used by Indian
state agricultural departments (ICAR standards).

Models:
- Population growth using logistic model with weather-driven rates
- Economic Injury Level (EIL) and Economic Threshold (ET) from ICAR
- Pesticide resistance buildup over repeated applications
- Natural enemy dynamics (predator-prey)

Reference: ICAR Pest Management Guidelines; Stern et al. (1959) EIL concept.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ICAR Economic Injury Levels and Economic Thresholds for Indian crops
PEST_PROFILES = {
    # Rice pests
    "brown_planthopper": {
        "crop": "rice_kharif",
        "scientific_name": "Nilaparvata lugens",
        "economic_threshold": 5,       # per hill
        "economic_injury_level": 10,   # per hill
        "favorable_temp_c": (25, 32),
        "favorable_humidity_pct": (80, 95),
        "base_growth_rate": 0.18,
        "max_population": 50,
        "natural_enemy_suppression": 0.25,
        "pesticide_options": ["imidacloprid", "buprofezin", "thiamethoxam"],
        "resistance_risk": "high",
    },
    "stem_borer": {
        "crop": "rice_kharif",
        "scientific_name": "Scirpophaga incertulas",
        "economic_threshold": 5,       # deadheart% or whitehead%
        "economic_injury_level": 10,
        "favorable_temp_c": (25, 35),
        "favorable_humidity_pct": (70, 90),
        "base_growth_rate": 0.12,
        "max_population": 30,
        "natural_enemy_suppression": 0.20,
        "pesticide_options": ["chlorantraniliprole", "fipronil", "cartap"],
        "resistance_risk": "medium",
    },
    # Wheat pests
    "aphid": {
        "crop": "wheat_rabi",
        "scientific_name": "Rhopalosiphum padi",
        "economic_threshold": 50,      # per tiller
        "economic_injury_level": 100,
        "favorable_temp_c": (12, 22),
        "favorable_humidity_pct": (60, 80),
        "base_growth_rate": 0.22,
        "max_population": 300,
        "natural_enemy_suppression": 0.35,
        "pesticide_options": ["dimethoate", "thiamethoxam", "imidacloprid"],
        "resistance_risk": "medium",
    },
    # Cotton pests
    "whitefly": {
        "crop": "cotton_kharif",
        "scientific_name": "Bemisia tabaci",
        "economic_threshold": 6,       # adults per leaf
        "economic_injury_level": 12,
        "favorable_temp_c": (28, 38),
        "favorable_humidity_pct": (50, 75),
        "base_growth_rate": 0.25,
        "max_population": 60,
        "natural_enemy_suppression": 0.20,
        "pesticide_options": ["imidacloprid", "spiromesifen", "pyriproxyfen"],
        "resistance_risk": "very_high",
    },
    "bollworm": {
        "crop": "cotton_kharif",
        "scientific_name": "Helicoverpa armigera",
        "economic_threshold": 2,       # larvae per plant
        "economic_injury_level": 4,
        "favorable_temp_c": (25, 35),
        "favorable_humidity_pct": (55, 80),
        "base_growth_rate": 0.10,
        "max_population": 20,
        "natural_enemy_suppression": 0.30,
        "pesticide_options": ["chlorantraniliprole", "spinosad", "indoxacarb"],
        "resistance_risk": "high",
    },
    # Tomato pests
    "fruit_borer": {
        "crop": "tomato_rabi",
        "scientific_name": "Helicoverpa armigera",
        "economic_threshold": 1,       # larva per plant
        "economic_injury_level": 2,
        "favorable_temp_c": (20, 32),
        "favorable_humidity_pct": (60, 85),
        "base_growth_rate": 0.10,
        "max_population": 15,
        "natural_enemy_suppression": 0.25,
        "pesticide_options": ["chlorantraniliprole", "spinosad", "emamectin_benzoate"],
        "resistance_risk": "high",
    },
    "early_blight": {
        "crop": "tomato_rabi",
        "scientific_name": "Alternaria solani",
        "economic_threshold": 10,      # % leaf area affected
        "economic_injury_level": 25,
        "favorable_temp_c": (24, 30),
        "favorable_humidity_pct": (80, 95),
        "base_growth_rate": 0.15,
        "max_population": 100,
        "natural_enemy_suppression": 0.05,
        "pesticide_options": ["mancozeb", "chlorothalonil", "azoxystrobin"],
        "resistance_risk": "low",
    },
}

PESTICIDE_EFFICACY = {
    "imidacloprid":         {"contact_kill_pct": 85, "residual_days": 14, "resistance_buildup": 0.08},
    "buprofezin":           {"contact_kill_pct": 70, "residual_days": 21, "resistance_buildup": 0.04},
    "thiamethoxam":         {"contact_kill_pct": 88, "residual_days": 12, "resistance_buildup": 0.07},
    "chlorantraniliprole":  {"contact_kill_pct": 92, "residual_days": 14, "resistance_buildup": 0.03},
    "fipronil":             {"contact_kill_pct": 80, "residual_days": 18, "resistance_buildup": 0.05},
    "cartap":               {"contact_kill_pct": 75, "residual_days": 10, "resistance_buildup": 0.04},
    "dimethoate":           {"contact_kill_pct": 72, "residual_days": 7,  "resistance_buildup": 0.06},
    "spiromesifen":         {"contact_kill_pct": 78, "residual_days": 21, "resistance_buildup": 0.03},
    "pyriproxyfen":         {"contact_kill_pct": 65, "residual_days": 28, "resistance_buildup": 0.02},
    "spinosad":             {"contact_kill_pct": 85, "residual_days": 7,  "resistance_buildup": 0.03},
    "indoxacarb":           {"contact_kill_pct": 80, "residual_days": 10, "resistance_buildup": 0.04},
    "mancozeb":             {"contact_kill_pct": 60, "residual_days": 10, "resistance_buildup": 0.01},
    "chlorothalonil":       {"contact_kill_pct": 65, "residual_days": 12, "resistance_buildup": 0.01},
    "azoxystrobin":         {"contact_kill_pct": 75, "residual_days": 14, "resistance_buildup": 0.02},
    "emamectin_benzoate":   {"contact_kill_pct": 90, "residual_days": 10, "resistance_buildup": 0.05},
    "neem_oil":             {"contact_kill_pct": 45, "residual_days": 5,  "resistance_buildup": 0.00},
    "none":                 {"contact_kill_pct": 0,  "residual_days": 0,  "resistance_buildup": 0.00},
}


@dataclass
class PestState:
    pest_name: str
    population: float          # Current population (scale: per unit as per EIL definition)
    at_threshold: bool         # >= Economic Threshold
    above_eil: bool            # >= Economic Injury Level
    resistance_index: float    # 0.0 (no resistance) to 1.0 (full resistance)
    days_since_spray: int
    natural_enemy_population: float
    damage_accumulated_pct: float  # Cumulative yield damage caused


class PestModel:
    """
    Tracks multiple pest species simultaneously.
    Uses logistic population growth with weather-dependent rates.
    """

    def __init__(self, crop_key: str, seed: Optional[int] = None):
        self.crop_key = crop_key
        self.rng = random.Random(seed or 42)

        # Get relevant pests for this crop
        self.active_pests = {
            k: v for k, v in PEST_PROFILES.items() if v["crop"] == crop_key
        }

        # State per pest
        self._populations: dict[str, float] = {}
        self._resistance: dict[str, float] = {}
        self._days_since_spray: dict[str, int] = {}
        self._natural_enemies: dict[str, float] = {}
        self._damage_pct: dict[str, float] = {}
        self._residual_active: dict[str, dict] = {}  # pesticide residual tracking

    def reset(self) -> list[PestState]:
        """Initialize pest populations at low background levels."""
        for pest_name, profile in self.active_pests.items():
            eil = profile["economic_injury_level"]
            # Start at 10-20% of economic threshold (background level)
            et = profile["economic_threshold"]
            self._populations[pest_name] = self.rng.uniform(et * 0.10, et * 0.20)
            self._resistance[pest_name] = self.rng.uniform(0.0, 0.05)
            self._days_since_spray[pest_name] = 30  # fresh start
            self._natural_enemies[pest_name] = profile["natural_enemy_suppression"] * 2
            self._damage_pct[pest_name] = 0.0
            self._residual_active[pest_name] = {}
        return self._build_states()

    def update(
        self,
        temperature_c: float,
        humidity_pct: float,
        growth_stage: str,
        spray_actions: dict[str, str],  # {pest_name: pesticide_name}
    ) -> tuple[list[PestState], dict]:
        """
        Advance pest dynamics by one day.

        Args:
            temperature_c: Mean air temperature
            humidity_pct: Relative humidity
            growth_stage: Current crop growth stage
            spray_actions: Dict of pest_name -> pesticide applied (or 'none')

        Returns:
            (List of PestState, info_dict)
        """
        info = {}
        total_unnecessary_sprays = 0
        total_correct_sprays = 0

        for pest_name, profile in self.active_pests.items():
            pop = self._populations[pest_name]
            resistance = self._resistance[pest_name]

            # Apply spray if instructed
            if pest_name in spray_actions and spray_actions[pest_name] != "none":
                pesticide = spray_actions[pest_name]
                efficacy_data = PESTICIDE_EFFICACY.get(pesticide, PESTICIDE_EFFICACY["none"])

                # Check if this was warranted (at or above threshold)
                et = profile["economic_threshold"]
                if pop >= et:
                    total_correct_sprays += 1
                else:
                    total_unnecessary_sprays += 1

                # Apply kill effect (reduced by resistance)
                effective_kill = efficacy_data["contact_kill_pct"] / 100.0 * (1 - resistance * 0.7)
                pop = pop * (1 - effective_kill)

                # Resistance buildup
                resistance = min(1.0, resistance + efficacy_data["resistance_buildup"])

                # Store residual
                self._residual_active[pest_name] = {
                    "pesticide": pesticide,
                    "days_remaining": efficacy_data["residual_days"],
                    "daily_kill": effective_kill * 0.15,
                }
                self._days_since_spray[pest_name] = 0
            else:
                self._days_since_spray[pest_name] += 1

            # Apply residual pesticide effect
            if self._residual_active.get(pest_name, {}).get("days_remaining", 0) > 0:
                residual = self._residual_active[pest_name]
                pop = max(0.0, pop * (1 - residual["daily_kill"]))
                residual["days_remaining"] -= 1

            # Weather-driven growth rate
            growth_rate = self._weather_growth_modifier(
                temperature_c, humidity_pct, profile
            )

            # Logistic population growth
            k = profile["max_population"]
            r = profile["base_growth_rate"] * growth_rate
            if pop < k:
                pop = pop + r * pop * (1 - pop / k)

            # Natural enemy suppression (Lotka-Volterra simplified)
            enemy_pop = self._natural_enemies[pest_name]
            enemy_suppression = enemy_pop * profile["natural_enemy_suppression"] * 0.05
            pop = max(0.0, pop - enemy_suppression)

            # Natural enemies grow when pest pop is high
            if pop > profile["economic_threshold"]:
                self._natural_enemies[pest_name] = min(
                    5.0, enemy_pop + 0.02 * (pop / profile["economic_injury_level"])
                )
            else:
                self._natural_enemies[pest_name] = max(
                    profile["natural_enemy_suppression"] * 0.5,
                    enemy_pop - 0.01
                )

            # Accumulate yield damage
            eil = profile["economic_injury_level"]
            if pop > profile["economic_threshold"]:
                damage_today = min(0.5, (pop - profile["economic_threshold"]) / eil * 0.02)
                self._damage_pct[pest_name] = min(40.0, self._damage_pct[pest_name] + damage_today)

            self._populations[pest_name] = round(max(0.0, pop), 2)
            self._resistance[pest_name] = round(resistance, 4)

        info["unnecessary_sprays"] = total_unnecessary_sprays
        info["correct_sprays"] = total_correct_sprays
        return self._build_states(), info

    def get_spray_recommendation(self) -> dict[str, str]:
        """Returns the oracle IPM recommendation (what an expert would do)."""
        recommendations = {}
        for pest_name, profile in self.active_pests.items():
            pop = self._populations.get(pest_name, 0)
            et = profile["economic_threshold"]
            resistance = self._resistance.get(pest_name, 0)

            if pop >= et:
                # Choose pesticide with lowest resistance risk and adequate efficacy
                options = profile["pesticide_options"]
                # Prefer option with lower resistance buildup
                best = min(
                    options,
                    key=lambda p: PESTICIDE_EFFICACY[p]["resistance_buildup"]
                )
                recommendations[pest_name] = best
            else:
                recommendations[pest_name] = "none"
        return recommendations

    def _weather_growth_modifier(
        self, temp_c: float, humidity_pct: float, profile: dict
    ) -> float:
        """
        Weather suitability multiplier for pest growth.
        Returns 0.2 (unfavorable) to 1.5 (highly favorable).
        """
        tmin, tmax = profile["favorable_temp_c"]
        hmin, hmax = profile["favorable_humidity_pct"]

        # Temperature suitability (bell curve)
        tmid = (tmin + tmax) / 2
        t_score = max(0.2, 1.0 - abs(temp_c - tmid) / ((tmax - tmin) / 2) * 0.8)

        # Humidity suitability
        hmid = (hmin + hmax) / 2
        h_score = max(0.2, 1.0 - abs(humidity_pct - hmid) / ((hmax - hmin) / 2) * 0.6)

        return min(1.5, t_score * h_score * 1.2)

    def _build_states(self) -> list[PestState]:
        states = []
        for pest_name, profile in self.active_pests.items():
            pop = self._populations.get(pest_name, 0)
            states.append(PestState(
                pest_name=pest_name,
                population=round(pop, 2),
                at_threshold=pop >= profile["economic_threshold"],
                above_eil=pop >= profile["economic_injury_level"],
                resistance_index=round(self._resistance.get(pest_name, 0), 3),
                days_since_spray=self._days_since_spray.get(pest_name, 30),
                natural_enemy_population=round(self._natural_enemies.get(pest_name, 0), 3),
                damage_accumulated_pct=round(self._damage_pct.get(pest_name, 0), 2),
            ))
        return states

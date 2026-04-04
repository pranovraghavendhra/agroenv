"""
Weather Engine — IMD-Calibrated Synthetic Weather Generator
============================================================
Uses FAO-56 Penman-Monteith reference evapotranspiration (ET₀) formula.
Weather is synthetic but statistically calibrated to IMD historical records
for three Indian agro-climatic zones.

Reference: Allen et al. (1998), FAO Irrigation and Drainage Paper 56.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional
import json
import os


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "weather_seeds.json")


@dataclass
class DailyWeather:
    day: int
    date_str: str          # "Day-N of season"
    tmax_c: float
    tmin_c: float
    tmean_c: float
    humidity_pct: float
    wind_speed_ms: float
    sunshine_hrs: float
    solar_radiation_mj_m2: float
    rainfall_mm: float
    et0_mm: float          # FAO-56 Penman-Monteith reference ET
    is_rainy: bool


class WeatherEngine:
    """
    Generates daily weather sequences for Indian agricultural regions.
    Uses monthly climatological normals + stochastic perturbation.
    ET₀ computed via FAO-56 Penman-Monteith equation.
    """

    def __init__(self, region_key: str = "maharashtra_pune", seed: Optional[int] = None):
        self.seed = seed if seed is not None else random.randint(0, 99999)
        self.rng = random.Random(self.seed)

        with open(DATA_PATH, "r") as f:
            all_regions = json.load(f)

        if region_key not in all_regions:
            region_key = "maharashtra_pune"

        self.region = all_regions[region_key]
        self.lat_rad = math.radians(self.region["lat"])
        self._weather_cache: list[DailyWeather] = []

    def generate_season(self, start_month: int, total_days: int) -> list[DailyWeather]:
        """Pre-generate full season weather. start_month is 1-indexed."""
        self._weather_cache = []
        for day_idx in range(total_days):
            month_idx = ((start_month - 1 + day_idx // 30) % 12)
            doy = (((start_month - 1) * 30) + day_idx + 1) % 365
            weather = self._generate_day(day_idx + 1, month_idx, doy)
            self._weather_cache.append(weather)
        return self._weather_cache

    def get_day(self, day: int) -> Optional[DailyWeather]:
        """Get weather for a specific day (1-indexed). Returns None if out of range."""
        if 1 <= day <= len(self._weather_cache):
            return self._weather_cache[day - 1]
        return None

    def get_forecast(self, current_day: int, horizon_days: int = 7) -> list[dict]:
        """
        Returns weather forecast for next N days with realistic uncertainty.
        Uncertainty increases with forecast horizon (mimics NWP models).
        """
        forecast = []
        for i in range(1, horizon_days + 1):
            target_day = current_day + i
            if target_day <= len(self._weather_cache):
                actual = self._weather_cache[target_day - 1]
                noise_factor = 1.0 + (i * 0.04)  # 4% error per day horizon
                forecast.append({
                    "day_ahead": i,
                    "tmax_c": round(actual.tmax_c + self.rng.gauss(0, i * 0.3), 1),
                    "tmin_c": round(actual.tmin_c + self.rng.gauss(0, i * 0.2), 1),
                    "rain_prob_pct": min(100, max(0, round(
                        (80 if actual.is_rainy else 15) + self.rng.gauss(0, i * 5), 0
                    ))),
                    "expected_rain_mm": round(
                        actual.rainfall_mm * self.rng.uniform(0.5, 1.5) if actual.is_rainy else
                        max(0, self.rng.gauss(0, 0.5)), 1
                    ),
                    "et0_forecast_mm": round(actual.et0_mm * noise_factor, 2),
                    "humidity_pct": round(actual.humidity_pct + self.rng.gauss(0, i * 1.5), 1),
                    "forecast_confidence": round(max(0.3, 1.0 - i * 0.08), 2)
                })
        return forecast

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _generate_day(self, day_num: int, month_idx: int, doy: int) -> DailyWeather:
        r = self.region
        tmax_mean = r["monthly_avg_tmax_c"][month_idx]
        tmin_mean = r["monthly_avg_tmin_c"][month_idx]
        rh_mean = r["monthly_avg_rh_pct"][month_idx]
        rain_days = r["rain_days_per_month"][month_idx]
        monthly_rain = r["monthly_avg_rain_mm"][month_idx]
        wind_mean = r["monthly_avg_wind_ms"][month_idx]
        sun_mean = r["monthly_avg_sunshine_hrs"][month_idx]

        # Stochastic daily variation
        tmax = tmax_mean + self.rng.gauss(0, 2.0)
        tmin = tmin_mean + self.rng.gauss(0, 1.5)
        if tmin >= tmax:
            tmin = tmax - 4.0
        tmean = (tmax + tmin) / 2.0

        humidity = max(20.0, min(98.0, rh_mean + self.rng.gauss(0, 6.0)))
        wind = max(0.5, wind_mean + self.rng.gauss(0, 0.8))
        sunshine = max(0.0, min(12.0, sun_mean + self.rng.gauss(0, 1.2)))

        # Rainfall generation (Markov chain-like)
        rain_prob = rain_days / 30.0
        is_rainy = self.rng.random() < rain_prob
        if is_rainy and monthly_rain > 0:
            # Exponential distribution for rainfall amount
            mean_rain_per_rainy_day = monthly_rain / max(1, rain_days)
            rainfall = self.rng.expovariate(1.0 / mean_rain_per_rainy_day)
            rainfall = round(min(rainfall, mean_rain_per_rainy_day * 6.0), 1)
        else:
            rainfall = 0.0
            is_rainy = False

        # Solar radiation from sunshine hours (Angstrom formula)
        ra = self._extraterrestrial_radiation(doy)
        solar_rad = (0.25 + 0.50 * (sunshine / 12.0)) * ra

        et0 = self._penman_monteith(tmean, tmax, tmin, humidity, wind, solar_rad, doy)

        return DailyWeather(
            day=day_num,
            date_str=f"Season Day {day_num}",
            tmax_c=round(tmax, 1),
            tmin_c=round(tmin, 1),
            tmean_c=round(tmean, 1),
            humidity_pct=round(humidity, 1),
            wind_speed_ms=round(wind, 2),
            sunshine_hrs=round(sunshine, 1),
            solar_radiation_mj_m2=round(solar_rad, 2),
            rainfall_mm=rainfall,
            et0_mm=round(et0, 2),
            is_rainy=is_rainy,
        )

    def _extraterrestrial_radiation(self, doy: int) -> float:
        """
        Extraterrestrial radiation (Ra) in MJ/m²/day.
        FAO-56 Equation 21.
        """
        dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
        declination = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
        ws = math.acos(-math.tan(self.lat_rad) * math.tan(declination))
        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
            ws * math.sin(self.lat_rad) * math.sin(declination) +
            math.cos(self.lat_rad) * math.cos(declination) * math.sin(ws)
        )
        return max(0.0, ra)

    def _penman_monteith(
        self, tmean: float, tmax: float, tmin: float,
        rh: float, wind: float, rs: float, doy: int
    ) -> float:
        """
        FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀).
        Returns ET₀ in mm/day.
        Reference: Allen et al. (1998), FAO Paper 56, Equation 6.
        """
        # Saturation vapour pressure (kPa)
        es_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
        es_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
        es = (es_tmax + es_tmin) / 2.0

        # Actual vapour pressure from relative humidity
        ea = (rh / 100.0) * es

        # Slope of saturation vapour pressure curve (kPa/°C)
        delta = (4098 * 0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)

        # Psychrometric constant (kPa/°C) — assumed elevation 400m
        gamma = 0.0665  # kPa/°C at ~400m elevation

        # Net radiation
        rns = (1 - 0.23) * rs  # net shortwave (albedo=0.23 for grass ref)
        rnl = self._net_longwave(tmax, tmin, ea, rs, doy)
        rn = rns - rnl
        rn = max(0.0, rn)

        # Soil heat flux (negligible for daily)
        g = 0.0

        # Wind speed at 2m height (assume measured at 10m, convert)
        u2 = wind * (4.87 / math.log(67.8 * 10 - 5.42))

        # PM equation
        numerator = (0.408 * delta * (rn - g) + gamma * (900 / (tmean + 273)) * u2 * (es - ea))
        denominator = delta + gamma * (1 + 0.34 * u2)

        et0 = numerator / denominator
        return max(0.0, et0)

    def _net_longwave(self, tmax: float, tmin: float, ea: float, rs: float, doy: int) -> float:
        """Net longwave radiation. FAO-56 Equation 39."""
        sigma = 4.903e-9  # Stefan-Boltzmann MJ/m²/day/K⁴
        tmax_k = tmax + 273.16
        tmin_k = tmin + 273.16
        ra = self._extraterrestrial_radiation(doy)
        rso = (0.75 + 2e-5 * 400) * ra  # clear-sky radiation at 400m
        cloud_factor = max(0.25, min(1.0, 1.35 * (rs / max(rso, 0.1)) - 0.35))
        humidity_factor = 0.34 - 0.14 * math.sqrt(max(0.0, ea))
        rnl = sigma * ((tmax_k ** 4 + tmin_k ** 4) / 2) * humidity_factor * cloud_factor
        return max(0.0, rnl)

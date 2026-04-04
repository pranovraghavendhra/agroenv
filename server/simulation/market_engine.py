"""
Market Price Engine
====================
Simulates Indian agricultural commodity prices with:
- AGMARKNET-calibrated baseline prices (MSP + market premium)
- Seasonal price cycles (harvest glut, off-season premium)
- Stochastic volatility (realistic for Indian mandi prices)
- Quality-based price adjustment
- Yield-quality relationship

Data calibrated from:
- AGMARKNET historical data (agmarknet.gov.in)
- CACP MSP recommendations 2023-24
"""

import json
import math
import random
import os
from dataclasses import dataclass


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_prices.json")


@dataclass
class MarketSnapshot:
    crop_key: str
    current_price_inr_per_quintal: float
    msp_inr_per_quintal: float
    price_vs_msp_pct: float          # % above/below MSP
    quality_grade: str               # A / B / C
    adjusted_price_inr_per_quintal: float
    estimated_revenue_inr_per_ha: float
    market_trend: str                # "rising" / "falling" / "stable"
    days_to_peak_price: int          # Approx days until best selling time
    glut_risk_pct: float             # Risk of price crash due to surplus


class MarketEngine:
    """
    Simulates mandi (wholesale market) prices for Indian crops.
    Uses seasonal indices + GBM-like stochastic component.
    """

    def __init__(self, crop_key: str = "rice_kharif", seed: int = 42):
        self.crop_key = crop_key
        self.rng = random.Random(seed)

        with open(DATA_PATH, "r") as f:
            self.market_data = json.load(f)

        if crop_key not in self.market_data:
            crop_key = "rice_kharif"
            self.crop_key = crop_key

        self.data = self.market_data[crop_key]
        self.msp = self.data["base_msp_inr_per_quintal"]
        self.volatility = self.data["annual_volatility_pct"] / 100.0

        # Current price state
        self._current_price = self.msp * (1 + self.data["market_premium_over_msp_pct"] / 100.0)
        self._price_history: list[float] = []
        self._current_month: int = 0

    def reset(self, start_month: int) -> MarketSnapshot:
        """Reset market to season start."""
        self._current_month = (start_month - 1) % 12
        monthly_idx = self.data["monthly_price_index"][self._current_month]
        self._current_price = self.msp * (1 + self.data["market_premium_over_msp_pct"] / 100.0) * monthly_idx
        self._price_history = [self._current_price]
        return self._build_snapshot("B")

    def update(self, day: int, season_day: int, quality_grade: str = "B") -> MarketSnapshot:
        """
        Advance market price by one day.

        Args:
            day: Calendar day (for seasonal adjustment)
            season_day: Day number within crop season
            quality_grade: Harvest quality A/B/C

        Returns:
            MarketSnapshot
        """
        # Update month every 30 days
        self._current_month = (self._current_month + (season_day % 30 == 0)) % 12

        # Seasonal price index
        monthly_idx = self.data["monthly_price_index"][self._current_month]
        base_price = self.msp * (1 + self.data["market_premium_over_msp_pct"] / 100.0) * monthly_idx

        # Daily stochastic walk (GBM)
        daily_vol = self.volatility / math.sqrt(252)
        drift = self.rng.gauss(0, daily_vol)
        self._current_price = self._current_price * (1 + drift)

        # Mean reversion toward seasonal base (prevents drift too far)
        mean_reversion = 0.05
        self._current_price = self._current_price + mean_reversion * (base_price - self._current_price)
        self._current_price = max(self.msp * 0.70, self._current_price)  # Floor at 70% MSP

        self._price_history.append(round(self._current_price, 2))
        return self._build_snapshot(quality_grade)

    def get_price_outlook(self) -> dict:
        """
        Returns market outlook for next 30 days (used by agent for harvest timing).
        """
        upcoming_prices = []
        current = self._current_price
        for d in range(1, 31):
            future_month = (self._current_month + d // 30) % 12
            idx = self.data["monthly_price_index"][future_month]
            base = self.msp * (1 + self.data["market_premium_over_msp_pct"] / 100.0) * idx
            # Simplified projection
            projected = current * 0.95 + base * 0.05 + self.rng.gauss(0, base * 0.02)
            upcoming_prices.append(round(projected, 0))

        trend_3d = upcoming_prices[2] - self._current_price
        return {
            "price_3d_ahead": upcoming_prices[2],
            "price_7d_ahead": upcoming_prices[6],
            "price_15d_ahead": upcoming_prices[14],
            "trend": "rising" if trend_3d > 50 else ("falling" if trend_3d < -50 else "stable"),
            "peak_price_estimate": max(upcoming_prices),
            "days_to_peak_estimate": upcoming_prices.index(max(upcoming_prices)) + 1,
        }

    def compute_revenue(self, yield_ton_per_ha: float, quality_grade: str) -> dict:
        """
        Compute final revenue from yield + quality + current price.
        """
        price = self._current_price

        # Quality adjustment
        if quality_grade == "A":
            price *= (1 + self.data["quality_premium_grade_a_pct"] / 100.0)
        elif quality_grade == "C":
            price *= (1 - self.data["quality_discount_grade_c_pct"] / 100.0)

        # Harvest glut penalty (if harvesting at same time as everyone)
        glut_discount = self.data["peak_harvest_glut_discount_pct"] / 100.0
        revenue_per_ha = yield_ton_per_ha * 10 * price  # 1 tonne = 10 quintals

        return {
            "price_per_quintal": round(price, 2),
            "yield_quintal_per_ha": round(yield_ton_per_ha * 10, 1),
            "revenue_inr_per_ha": round(revenue_per_ha, 0),
            "quality_grade": quality_grade,
        }

    def _build_snapshot(self, quality_grade: str) -> MarketSnapshot:
        # Trend from last 5 days
        if len(self._price_history) >= 5:
            trend_val = self._price_history[-1] - self._price_history[-5]
            trend = "rising" if trend_val > 30 else ("falling" if trend_val < -30 else "stable")
        else:
            trend = "stable"

        # Adjusted price for quality
        adj_price = self._current_price
        if quality_grade == "A":
            adj_price *= (1 + self.data["quality_premium_grade_a_pct"] / 100.0)
        elif quality_grade == "C":
            adj_price *= (1 - self.data["quality_discount_grade_c_pct"] / 100.0)

        # Rough revenue estimate (assuming 80% of max yield)
        estimated_yield = 0.80 * 5.0  # tonne/ha placeholder
        estimated_revenue = estimated_yield * 10 * adj_price

        outlook = self.get_price_outlook()

        return MarketSnapshot(
            crop_key=self.crop_key,
            current_price_inr_per_quintal=round(self._current_price, 2),
            msp_inr_per_quintal=self.msp,
            price_vs_msp_pct=round((self._current_price / self.msp - 1) * 100, 1),
            quality_grade=quality_grade,
            adjusted_price_inr_per_quintal=round(adj_price, 2),
            estimated_revenue_inr_per_ha=round(estimated_revenue, 0),
            market_trend=trend,
            days_to_peak_price=outlook["days_to_peak_estimate"],
            glut_risk_pct=round(self.data["peak_harvest_glut_discount_pct"], 1),
        )

"""
Grader: Full Season Optimizer (Hard)
======================================
Deterministic scoring. Primarily revenue-driven with sustainability bonuses.
"""

from .base_task import BaseGrader, EpisodeRecord


BREAK_EVEN_INR = 80_000.0
MAX_REVENUE_INR = 280_000.0
MAX_YIELD_TON_PER_HA = 25.0
INPUT_COST_INR = 15_000.0


class SeasonGrader(BaseGrader):

    @property
    def task_name(self) -> str:
        return "season_optimizer"

    def grade(self, episode: EpisodeRecord) -> tuple[float, dict]:
        """
        Scoring breakdown:
          A. Net revenue outcome         — 45%
          B. Yield achievement           — 25%
          C. Harvest timing quality      — 15%
          D. Sustainability (water/pest) — 15%
        """
        if not episode.observations:
            return 0.0, {"error": "No observations recorded"}

        final_obs = episode.observations[-1]

        # ---- A: Net Revenue (45%) ----
        gross_revenue = episode.final_revenue_inr_per_ha
        net_revenue = gross_revenue - INPUT_COST_INR - episode.total_cost_inr
        net_revenue = max(0.0, net_revenue)

        if net_revenue >= MAX_REVENUE_INR * 0.75:
            revenue_score = 1.0
        elif net_revenue >= BREAK_EVEN_INR * 1.5:
            revenue_score = 0.80
        elif net_revenue >= BREAK_EVEN_INR:
            t = (net_revenue - BREAK_EVEN_INR) / (BREAK_EVEN_INR * 0.5)
            revenue_score = 0.55 + 0.25 * t
        elif net_revenue >= BREAK_EVEN_INR * 0.5:
            t = (net_revenue - BREAK_EVEN_INR * 0.5) / (BREAK_EVEN_INR * 0.5)
            revenue_score = 0.20 + 0.35 * t
        else:
            revenue_score = max(0.0, net_revenue / (BREAK_EVEN_INR * 0.5) * 0.20)

        # ---- B: Yield Achievement (25%) ----
        actual_yield = episode.final_yield_ton_per_ha
        yield_fraction = actual_yield / MAX_YIELD_TON_PER_HA

        if yield_fraction >= 0.85:
            yield_score = 1.0
        elif yield_fraction >= 0.65:
            yield_score = 0.75 + (yield_fraction - 0.65) / 0.20 * 0.25
        elif yield_fraction >= 0.45:
            yield_score = 0.40 + (yield_fraction - 0.45) / 0.20 * 0.35
        else:
            yield_score = max(0.0, yield_fraction / 0.45 * 0.40)

        # ---- C: Harvest Timing (15%) ----
        if episode.harvest_day is None:
            # Never harvested — worst case
            harvest_score = 0.0
        else:
            crop_final = final_obs.crop
            # Was harvest done within window?
            if crop_final.in_harvest_window or crop_final.harvest_window_closing_days == 0:
                # Check if done at good GDD progress
                harvest_score = 0.80

                # Bonus for price timing (captured in revenue already, small extra here)
                harvest_idx = min(episode.harvest_day - 1, len(episode.observations) - 1)
                last_market = episode.observations[harvest_idx].market if episode.harvest_day and episode.observations else None
                if last_market and last_market.market_trend == "rising":
                    harvest_score = 1.0
                elif last_market and last_market.market_trend == "stable":
                    harvest_score = 0.90
            elif episode.harvest_day < 70:
                # Very early harvest
                harvest_score = max(0.0, 0.30 - (70 - episode.harvest_day) * 0.02)
            else:
                harvest_score = 0.40  # Late but better than never

        # ---- D: Sustainability (15%) ----
        # Water use efficiency
        et_crop_total = sum(
            info.get("etc_mm", 0) for info in episode.step_infos
        )
        water_use_ratio = (
            episode.total_irrigation_mm / max(1, et_crop_total)
        )
        if 0.85 <= water_use_ratio <= 1.20:
            water_efficiency = 1.0
        elif 0.65 <= water_use_ratio < 0.85:
            water_efficiency = 0.70
        elif 1.20 < water_use_ratio <= 1.60:
            water_efficiency = 0.65
        else:
            water_efficiency = 0.25

        # Pest resistance sustainability
        max_resistance = max(
            (p.resistance_index for p in final_obs.pests), default=0.0
        )
        resistance_sustainability = max(0.0, 1.0 - max_resistance * 1.5)

        sustainability_score = 0.60 * water_efficiency + 0.40 * resistance_sustainability

        # ---- Final Score ----
        final_score = (
            0.45 * revenue_score +
            0.25 * yield_score +
            0.15 * harvest_score +
            0.15 * sustainability_score
        )
        final_score = self._clamp(final_score)

        breakdown = {
            "revenue_score": round(revenue_score, 3),
            "yield_score": round(yield_score, 3),
            "harvest_timing_score": round(harvest_score, 3),
            "sustainability_score": round(sustainability_score, 3),
            "final_score": round(final_score, 3),
            "gross_revenue_inr": round(gross_revenue, 0),
            "net_revenue_inr": round(net_revenue, 0),
            "final_yield_ton_per_ha": round(actual_yield, 2),
            "yield_pct_of_max": round(yield_fraction * 100, 1),
            "harvest_day": episode.harvest_day,
            "total_irrigation_mm": round(episode.total_irrigation_mm, 1),
            "water_use_ratio": round(water_use_ratio, 3),
            "max_resistance_index": round(max_resistance, 3),
            "passed": final_score >= 0.55,
        }

        return final_score, breakdown

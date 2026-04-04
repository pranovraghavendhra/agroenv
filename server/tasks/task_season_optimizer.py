"""
Task 3: Full Season Harvest Optimizer (Hard)
==============================================
The agent manages a complete tomato growing season (110 days) in Andhra Pradesh.
Must simultaneously optimize: irrigation, pest management, AND harvest timing
to maximize net revenue per hectare.

This is the hardest task because:
1. Long horizon (110 days) with decisions compounding
2. Must balance 4 competing objectives simultaneously
3. Harvest timing is critical and irreversible (cannot un-harvest)
4. Market price volatility adds uncertainty to harvest decision
5. Input budget is tight (₹15,000/ha) — every rupee must be justified
6. Pest-water-market interactions create complex tradeoffs

Success criteria:
  Net revenue ≥ ₹80,000/ha (approx. break-even for smallholder tomato)
  Yield ≥ 60% of maximum potential
  Episode score ≥ 0.55.

Why this is 'Hard':
  - Frontier models score ~0.38 baseline (below break-even)
  - Optimal agents score ~0.85
  - Many locally-optimal traps (e.g. heavy irrigation → waterlogging → disease)

Real world context:
  Tomato in AP is a high-value, high-risk crop.
  Price can swing from ₹500 to ₹4000/quintal in the same season.
  Water stress during fruit set causes blossom drop — irreversible.
  Early Blight is the #1 yield killer if not managed correctly.
"""
from __future__ import annotations

from ..models import AgroAction, AgroObservation

from .base_task import BaseTask, TaskConfig, EpisodeRecord
from ..simulation.pest_model import PESTICIDE_EFFICACY


BREAK_EVEN_REVENUE_INR_PER_HA = 80_000.0   # Approx. cost of cultivation for tomato in AP
MAX_REVENUE_INR_PER_HA = 280_000.0          # Top end for excellent management
HARVEST_COST_INR_PER_HA = 8_000.0           # Labour cost to harvest
INPUT_BUDGET_INR_PER_HA = 15_000.0


class SeasonOptimizerTask(BaseTask):

    @property
    def config(self) -> TaskConfig:
        return TaskConfig(
            task_name="season_optimizer",
            crop_key="tomato_rabi",
            soil_key="red_laterite_soil",
            region_key="andhra_guntur",
            max_steps=110,
            description=(
                "Manage a complete tomato Rabi season (110 days) in Andhra Pradesh on red laterite soil. "
                "Make daily decisions on irrigation, pest/disease management (fruit borer + early blight), "
                "and crucially — when to harvest. "
                "Market prices fluctuate significantly. Harvest too early: low yield, low quality. "
                "Harvest too late: crop deteriorates, price may crash. "
                "Budget: ₹15,000/ha. Goal: maximize net revenue per hectare. "
                "Break-even is ₹80,000/ha. Top farmers achieve ₹2,00,000+/ha."
            ),
            success_criteria=(
                "Net revenue ≥ ₹80,000/ha. "
                "Final yield ≥ 60% of maximum (15 tonnes/ha). "
                "Harvest within GDD harvest window. "
                "Episode score ≥ 0.55."
            ),
            difficulty="hard",
        )

    def compute_step_reward(
        self,
        action: AgroAction,
        obs_before: AgroObservation,
        obs_after: AgroObservation,
        step_info: dict,
    ) -> tuple[float, str]:
        """
        Reward is a dense composite signal over the full season.
        Design philosophy: every day of good management earns a small positive reward.
        Major events (waterlogging, missed pests, wrong harvest) give large negative/positive.
        """
        reward = 0.0
        messages = []

        # ---- 1. Irrigation quality signal (0.15 max per step) ----
        soil = obs_before.soil
        ks = obs_after.soil.ks
        if ks >= 0.90:
            reward += 0.05
        elif ks < 0.60:
            reward -= 0.10
            messages.append("⚠ Severe water stress — yield damage likely")

        if obs_after.soil.waterlog_days > obs_before.soil.waterlog_days:
            reward -= 0.12
            messages.append("✗ Waterlogging event — root oxygen stress for tomato")

        if obs_after.soil.drainage_mm_today > 15:
            reward -= 0.06
            messages.append(f"⚠ {obs_after.soil.drainage_mm_today:.0f}mm wasted to drainage")

        # ---- 2. Pest management signal (0.15 max per step) ----
        for pest_obs in obs_before.pests:
            pest_name = pest_obs.pest_name
            spray_map = {d.pest_name: d.pesticide for d in action.spray_decisions}
            pesticide = spray_map.get(pest_name, "none")
            sprayed = pesticide != "none"

            if pest_obs.at_threshold:
                if sprayed:
                    reward += 0.08
                    efficacy = PESTICIDE_EFFICACY.get(pesticide, {})
                    if efficacy.get("resistance_buildup", 1.0) <= 0.03:
                        reward += 0.03
                else:
                    reward -= 0.10
                    messages.append(f"✗ Missed spray on {pest_name} at threshold")
            elif not pest_obs.at_threshold and sprayed:
                reward -= 0.08
                messages.append(f"✗ Unnecessary spray on {pest_name}")

        # Blight spreading during humid days is especially costly for tomato
        for pest_obs in obs_before.pests:
            if pest_obs.pest_name == "early_blight" and pest_obs.above_eil:
                if obs_before.weather_today.humidity_pct > 85:
                    reward -= 0.05
                    messages.append("✗ Blight above EIL in high humidity — critical window")

        # ---- 3. Fruit set protection bonus ----
        crop = obs_before.crop
        if crop.growth_stage in ("flowering", "fruit_set"):
            # Water stress during fruit set is irreversible
            if obs_before.soil.ks < 0.75:
                reward -= 0.12
                messages.append("✗ CRITICAL: Water stress during fruit set — blossom drop risk")
            else:
                reward += 0.04

        # ---- 4. Harvest timing signal ----
        if action.harvest_now:
            if crop.in_harvest_window:
                # Bonus for harvesting at good GDD
                gdd = crop.gdd_accumulated
                gdd_optimal_center = (obs_before.crop.gdd_progress_pct / 100.0)
                market = obs_before.market

                # Price timing bonus
                if market.market_trend == "rising":
                    reward += 0.30
                    messages.append("✓ Excellent harvest timing: price rising")
                elif market.market_trend == "stable":
                    reward += 0.20
                    messages.append("✓ Good harvest timing: stable price")
                else:
                    reward += 0.10
                    messages.append("⚠ Harvested while price falling")

                # Yield quality at harvest
                if crop.estimated_yield_pct >= 80:
                    reward += 0.15
                    messages.append("✓ Excellent yield at harvest")
                elif crop.estimated_yield_pct >= 60:
                    reward += 0.08
                else:
                    reward -= 0.10
                    messages.append("✗ Poor yield at harvest — too much accumulated stress")

            elif not crop.in_harvest_window and crop.days_to_harvest_window > 0:
                reward -= 0.40
                messages.append(f"✗ Premature harvest: {crop.days_to_harvest_window} days before window")

            elif crop.harvest_window_closing_days == 0:
                reward -= 0.20
                messages.append("✗ Harvest after optimal window — quality degraded")

        # ---- 5. Resource management ----
        budget_remaining = obs_before.resources.budget_remaining_inr
        if budget_remaining < 1000 and obs_before.day < 80:
            reward -= 0.05
            messages.append("⚠ Budget nearly exhausted before season end")

        reward = max(-1.0, min(1.0, reward))
        feedback = " | ".join(messages) if messages else "Routine management"
        return round(reward, 4), feedback

    def is_done(self, obs: AgroObservation, step: int, episode: EpisodeRecord) -> bool:
        # Episode ends if: harvest was triggered, season complete, or budget exhausted
        if episode.harvest_day is not None:
            return True
        if step >= self.config.max_steps:
            return True
        if obs.resources.budget_remaining_inr < 0:
            return True
        return False

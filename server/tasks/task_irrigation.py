"""
Task 1: Irrigation Scheduling (Easy)
======================================
The agent manages irrigation for a 14-day window during the critical
tillering/flowering stage of rice cultivation in Maharashtra.

Objective:
  Maintain optimal soil moisture while minimising water waste.
  Correct decisions: irrigate when soil moisture drops below RAW threshold;
  skip irrigation when soil is adequately moist or rain is forecast.

Success criteria:
  Cumulative step score >= 0.65 over 14 days.

Why this is 'Easy':
  - Short horizon (14 days only)
  - Single decision per step (irrigate yes/no + how much)
  - Clear signal from soil moisture and ET₀
  - No pest or harvest decisions
  - Oracle answer is computable from FAO-56 directly

Difficulty for frontier LLMs: Medium-Easy. GPT-4 scores ~0.72 baseline.
"""
from __future__ import annotations

from ..models import AgroAction, AgroObservation

from dataclasses import dataclass
from .base_task import BaseTask, TaskConfig, EpisodeRecord


IRRIGATION_COST_INR_PER_MM_PER_HA = 8.0   # ₹8 per mm per hectare (drip/sprinkler)
FLOOD_COST_INR_PER_MM_PER_HA = 3.5        # Cheaper but less efficient
OVER_IRRIGATION_PENALTY_THRESHOLD_MM = 60  # Daily irrigation above this is wasteful


class IrrigationTask(BaseTask):

    @property
    def config(self) -> TaskConfig:
        return TaskConfig(
            task_name="irrigation_scheduling",
            crop_key="rice_kharif",
            soil_key="loamy_soil",
            region_key="maharashtra_pune",
            max_steps=14,
            description=(
                "Manage irrigation for a rice crop over 14 days during the tillering stage. "
                "Use soil moisture readings, ET₀ estimates, and 7-day weather forecasts "
                "to decide when to irrigate and how much water to apply. "
                "Goal: maintain soil moisture in the optimal range (60–80% of field capacity) "
                "while minimising water waste and cost."
            ),
            success_criteria=(
                "Maintain soil moisture above wilting stress threshold on ≥10 of 14 days. "
                "Avoid waterlogging (>95% saturation). "
                "Total irrigation water use within 20% of FAO-56 optimal. "
                "Episode score ≥ 0.65."
            ),
            difficulty="easy",
        )

    def compute_step_reward(
        self,
        action: AgroAction,
        obs_before: AgroObservation,
        obs_after: AgroObservation,
        step_info: dict,
    ) -> tuple[float, str]:
        """
        Reward breakdown for irrigation task:
        - Correct irrigation decision: +0.30
        - Wrong direction (over/under): -0.20 to -0.30
        - Efficient method bonus: +0.05
        - Water waste penalty: -0.10 per 10mm over-application
        """
        reward = 0.0
        messages = []

        soil_before = obs_before.soil
        soil_after = obs_after.soil
        soil_fc = soil_before.field_capacity_pct
        soil_wp = soil_before.wilting_point_pct
        raw_mm = soil_before.raw_mm

        # --- Core decision correctness ---
        depletion_before = soil_before.depletion_mm

        # Oracle: should we have irrigated?
        oracle_irrigate = depletion_before > raw_mm * 0.85
        rain_incoming = any(
            d.rain_prob_pct > 60 and d.expected_rain_mm > 10
            for d in obs_before.weather_forecast[:3]
        )

        if oracle_irrigate and not rain_incoming:
            if action.irrigate and action.irrigation_amount_mm > 5:
                reward += 0.30
                messages.append("✓ Correct: irrigated when soil was depleted")

                # Efficiency: check amount accuracy vs optimal
                oracle_amount = step_info.get("oracle_irrigation_mm", 0)
                if oracle_amount > 0:
                    amount_error_pct = abs(action.irrigation_amount_mm - oracle_amount) / oracle_amount
                    if amount_error_pct < 0.15:
                        reward += 0.10
                        messages.append("✓ Excellent amount precision (within 15% of optimal)")
                    elif amount_error_pct < 0.35:
                        reward += 0.05
                        messages.append("✓ Good amount precision")

                # Over-irrigation penalty
                if action.irrigation_amount_mm > OVER_IRRIGATION_PENALTY_THRESHOLD_MM:
                    excess = action.irrigation_amount_mm - OVER_IRRIGATION_PENALTY_THRESHOLD_MM
                    penalty = min(0.20, excess / 100.0 * 0.20)
                    reward -= penalty
                    messages.append(f"⚠ Over-irrigation penalty: {excess:.0f}mm excess")

            else:
                reward -= 0.25
                messages.append("✗ Missed irrigation: soil depletion exceeded RAW threshold")

        elif not oracle_irrigate or rain_incoming:
            if not action.irrigate or action.irrigation_amount_mm < 5:
                reward += 0.25
                messages.append("✓ Correct: skipped irrigation (soil adequate or rain forecast)")

                if rain_incoming and not action.irrigate:
                    reward += 0.05
                    messages.append("✓ Smart: rain forecast correctly identified")
            else:
                reward -= 0.15
                messages.append("✗ Unnecessary irrigation: soil was adequate")

                # Waterlogging penalty
                if soil_after.moisture_pct > soil_before.field_capacity_pct * 0.95:
                    reward -= 0.10
                    messages.append("✗ Waterlogging risk from over-irrigation")

        # --- Drainage penalty (wasted water) ---
        if soil_after.drainage_mm_today > 10:
            reward -= 0.08
            messages.append(f"⚠ {soil_after.drainage_mm_today:.1f}mm lost to drainage (waste)")

        # --- Irrigation method efficiency bonus ---
        if action.irrigate and action.irrigation_amount_mm > 0:
            if action.irrigation_method == "drip":
                reward += 0.05
                messages.append("✓ Efficient drip irrigation selected")
            elif action.irrigation_method == "flood":
                reward -= 0.03
                messages.append("⚠ Flood irrigation selected (less efficient)")

        # Clamp to [-1, 1]
        reward = max(-1.0, min(1.0, reward))
        feedback = " | ".join(messages) if messages else "No action feedback"
        return round(reward, 4), feedback

    def is_done(self, obs: AgroObservation, step: int, episode: EpisodeRecord) -> bool:
        return step >= self.config.max_steps

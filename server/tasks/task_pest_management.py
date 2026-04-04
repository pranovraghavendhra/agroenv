"""
Task 2: Pest & Disease Management (Medium)
============================================
The agent manages a cotton crop through a 30-day pest pressure window
(flowering to boll development stage) with whitefly and bollworm pressure.

Objective:
  Apply pesticides only when pest populations exceed ICAR Economic Thresholds.
  Balance immediate control with long-term resistance management.
  Avoid spraying when natural enemies are providing sufficient biocontrol.

Success criteria:
  Keep pest damage below 8% cumulative yield loss.
  Avoid resistance index > 0.40 on any pest.
  Keep unnecessary spray rate below 30%.
  Episode score ≥ 0.60.

Why this is 'Medium':
  - Must understand Economic Threshold vs Economic Injury Level distinction
  - Must track pesticide resistance across multiple applications
  - Must balance biological control vs chemical intervention
  - Natural enemy dynamics add uncertainty
  - Two pests simultaneously with different thresholds

Difficulty for frontier LLMs: Medium. GPT-4 scores ~0.58 baseline.
"""
from __future__ import annotations

from ..models import AgroAction, AgroObservation

from .base_task import BaseTask, TaskConfig, EpisodeRecord
from ..simulation.pest_model import PEST_PROFILES, PESTICIDE_EFFICACY


SPRAY_COST_INR_PER_EVENT = 850.0     # ₹850 per spray event per hectare (labour + chemical)
MAX_SPRAY_EVENTS = 6                  # Agronomic limit


class PestManagementTask(BaseTask):

    @property
    def config(self) -> TaskConfig:
        return TaskConfig(
            task_name="pest_management",
            crop_key="cotton_kharif",
            soil_key="black_cotton_soil",
            region_key="andhra_guntur",
            max_steps=30,
            description=(
                "Manage a cotton crop through the critical flowering-to-boll stage. "
                "Two pest species are active: whitefly (Bemisia tabaci) and bollworm (Helicoverpa armigera). "
                "Use ICAR Economic Thresholds to decide when to spray, which pesticide to use, "
                "and how to rotate chemicals to prevent resistance buildup. "
                "Natural enemies are present — unnecessary sprays kill them too. "
                "Budget: ₹5,000/ha. Each spray event costs ₹850."
            ),
            success_criteria=(
                "Cumulative pest damage < 8% of potential yield. "
                "Pesticide resistance index < 0.40 on both pests. "
                "Unnecessary spray rate < 30%. "
                "Episode score ≥ 0.60."
            ),
            difficulty="medium",
        )

    def compute_step_reward(
        self,
        action: AgroAction,
        obs_before: AgroObservation,
        obs_after: AgroObservation,
        step_info: dict,
    ) -> tuple[float, str]:
        """
        Reward breakdown for pest management:
        - Correct spray at threshold: +0.25
        - Correct no-spray (below threshold): +0.15
        - Spray above EIL (too late): +0.10 (better late than never)
        - Unnecessary spray: -0.20
        - Spray kills natural enemies unnecessarily: -0.10
        - Good pesticide rotation: +0.05
        - Building resistance: -0.10
        """
        reward = 0.0
        messages = []

        spray_decisions = {d.pest_name: d.pesticide for d in action.spray_decisions}
        unnecessary = step_info.get("unnecessary_sprays", 0)
        correct = step_info.get("correct_sprays", 0)

        # Score each pest's spray decision
        for pest_obs in obs_before.pests:
            pest_name = pest_obs.pest_name
            pesticide = spray_decisions.get(pest_name, "none")
            sprayed = pesticide != "none"

            if pest_obs.at_threshold or pest_obs.above_eil:
                if sprayed:
                    reward += 0.25
                    messages.append(f"✓ Correct spray on {pest_name} (at threshold)")

                    # Reward good pesticide choice (lower resistance risk)
                    efficacy = PESTICIDE_EFFICACY.get(pesticide, {})
                    resistance_risk = efficacy.get("resistance_buildup", 0.10)
                    if resistance_risk <= 0.03:
                        reward += 0.05
                        messages.append(f"✓ Low-resistance-risk pesticide chosen for {pest_name}")
                    elif resistance_risk >= 0.07:
                        reward -= 0.05
                        messages.append(f"⚠ High-resistance-risk pesticide for {pest_name}")

                    # Penalty for spraying the same pesticide repeatedly (resistance rotation)
                    if pest_obs.days_since_spray < 10 and pest_obs.resistance_index > 0.15:
                        reward -= 0.08
                        messages.append(f"⚠ No pesticide rotation for {pest_name} — resistance building")
                else:
                    if pest_obs.above_eil:
                        reward -= 0.20
                        messages.append(f"✗ Failed to spray {pest_name} above EIL — yield damage occurring")
                    else:
                        reward -= 0.10
                        messages.append(f"✗ Missed spray on {pest_name} at Economic Threshold")

            else:
                # Below threshold
                if sprayed:
                    reward -= 0.20
                    messages.append(f"✗ Unnecessary spray on {pest_name} (below threshold)")

                    # Extra penalty if natural enemies were active
                    if pest_obs.natural_enemy_population > 0.5:
                        reward -= 0.08
                        messages.append(f"✗ Natural enemies destroyed for {pest_name}")
                else:
                    reward += 0.15
                    messages.append(f"✓ Correct: no spray on {pest_name} (below threshold)")

                    # Bonus for recognizing biocontrol
                    if pest_obs.natural_enemy_population > 0.8:
                        reward += 0.05
                        messages.append(f"✓ Natural enemies providing control for {pest_name}")

        # Resistance penalty from after-state
        for pest_after in obs_after.pests:
            if pest_after.resistance_index > 0.35:
                reward -= 0.05
                messages.append(f"⚠ High resistance index for {pest_after.pest_name}: {pest_after.resistance_index:.2f}")

        # Budget constraint
        spray_cost = len([d for d in action.spray_decisions if d.pesticide != "none"]) * SPRAY_COST_INR_PER_EVENT
        if obs_before.resources.budget_remaining_inr < spray_cost:
            reward -= 0.15
            messages.append("✗ Budget exceeded for spray action")

        reward = max(-1.0, min(1.0, reward))
        feedback = " | ".join(messages) if messages else "No pest action taken"
        return round(reward, 4), feedback

    def is_done(self, obs: AgroObservation, step: int, episode: EpisodeRecord) -> bool:
        # End early if all pests have above 30% damage accumulated (crop ruined)
        if all(p.damage_accumulated_pct > 30 for p in obs.pests):
            return True
        return step >= self.config.max_steps

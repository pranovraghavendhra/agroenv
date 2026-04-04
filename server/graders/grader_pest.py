"""
Grader: Pest & Disease Management (Medium)
============================================
Deterministic scoring of completed pest management episode.
"""

from .base_task import BaseGrader, EpisodeRecord


class PestGrader(BaseGrader):

    @property
    def task_name(self) -> str:
        return "pest_management"

    def grade(self, episode: EpisodeRecord) -> tuple[float, dict]:
        """
        Scoring breakdown:
          A. Pest damage control         — 35%
          B. IPM compliance              — 30%
          C. Resistance management       — 20%
          D. Biocontrol preservation     — 15%
        """
        if not episode.observations:
            return 0.0, {"error": "No observations recorded"}

        final_obs = episode.observations[-1]

        # ---- A: Pest Damage Control (35%) ----
        total_damage_pct = sum(p.damage_accumulated_pct for p in final_obs.pests)
        n_pests = max(1, len(final_obs.pests))
        avg_damage = total_damage_pct / n_pests

        if avg_damage < 3.0:
            damage_score = 1.0
        elif avg_damage < 8.0:
            damage_score = 1.0 - (avg_damage - 3.0) / 5.0 * 0.5
        elif avg_damage < 20.0:
            damage_score = 0.50 - (avg_damage - 8.0) / 12.0 * 0.40
        else:
            damage_score = max(0.0, 0.10 - (avg_damage - 20.0) / 20.0 * 0.10)

        # ---- B: IPM Compliance (30%) ----
        total_sprays = episode.total_spray_events
        correct_sprays = episode.correct_sprays
        unnecessary_sprays = episode.unnecessary_sprays

        if total_sprays == 0:
            # Check if any pest exceeded EIL — if yes, bad (missed treatment)
            max_damage = max((p.damage_accumulated_pct for p in final_obs.pests), default=0)
            ipm_score = 0.50 if max_damage > 8 else 0.85
        else:
            correct_rate = correct_sprays / total_sprays
            unnecessary_rate = unnecessary_sprays / total_sprays
            ipm_score = correct_rate - unnecessary_rate * 0.5
            ipm_score = max(0.0, min(1.0, ipm_score))

        # ---- C: Resistance Management (20%) ----
        max_resistance = max(
            (p.resistance_index for p in final_obs.pests), default=0.0
        )
        if max_resistance < 0.15:
            resistance_score = 1.0
        elif max_resistance < 0.30:
            resistance_score = 0.80
        elif max_resistance < 0.45:
            resistance_score = 0.50
        elif max_resistance < 0.65:
            resistance_score = 0.25
        else:
            resistance_score = 0.05

        # ---- D: Biocontrol Preservation (15%) ----
        avg_natural_enemies = sum(
            p.natural_enemy_population for p in final_obs.pests
        ) / max(1, len(final_obs.pests))
        # Higher natural enemy population = better biocontrol preservation
        if avg_natural_enemies > 0.8:
            biocontrol_score = 1.0
        elif avg_natural_enemies > 0.5:
            biocontrol_score = 0.75
        elif avg_natural_enemies > 0.3:
            biocontrol_score = 0.50
        else:
            biocontrol_score = 0.20

        # ---- Final Score ----
        final_score = (
            0.35 * damage_score +
            0.30 * ipm_score +
            0.20 * resistance_score +
            0.15 * biocontrol_score
        )
        final_score = self._clamp(final_score)

        breakdown = {
            "damage_control_score": round(damage_score, 3),
            "ipm_compliance_score": round(ipm_score, 3),
            "resistance_management_score": round(resistance_score, 3),
            "biocontrol_preservation_score": round(biocontrol_score, 3),
            "final_score": round(final_score, 3),
            "avg_pest_damage_pct": round(avg_damage, 2),
            "total_spray_events": total_sprays,
            "correct_sprays": correct_sprays,
            "unnecessary_sprays": unnecessary_sprays,
            "max_resistance_index": round(max_resistance, 3),
            "avg_natural_enemy_population": round(avg_natural_enemies, 3),
            "passed": final_score >= 0.60,
        }

        return final_score, breakdown

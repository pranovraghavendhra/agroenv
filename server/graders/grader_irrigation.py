"""
Grader: Irrigation Scheduling (Easy)
======================================
Deterministic final scoring of a completed irrigation episode.
Score components are independently verifiable.
"""

from .base_task import BaseGrader, EpisodeRecord


class IrrigationGrader(BaseGrader):

    @property
    def task_name(self) -> str:
        return "irrigation_scheduling"

    def grade(self, episode: EpisodeRecord) -> tuple[float, dict]:
        """
        Scoring breakdown:
          A. Water management efficiency   — 40%
          B. Stress avoidance              — 35%
          C. Waste minimization            — 15%
          D. Method efficiency             — 10%
        """
        if not episode.observations:
            return 0.0, {"error": "No observations recorded"}

        # ---- A: Water Management Efficiency (40%) ----
        # Compare total irrigation applied vs FAO-56 optimal
        # Oracle optimal is tracked in step_infos
        total_oracle = sum(
            info.get("oracle_irrigation_mm", 0)
            for info in episode.step_infos
        )
        total_applied = episode.total_irrigation_mm

        if total_oracle > 0:
            efficiency_ratio = total_applied / total_oracle
            if 0.85 <= efficiency_ratio <= 1.20:
                water_score = 1.0
            elif 0.70 <= efficiency_ratio < 0.85 or 1.20 < efficiency_ratio <= 1.40:
                water_score = 0.75
            elif 0.50 <= efficiency_ratio < 0.70 or 1.40 < efficiency_ratio <= 1.70:
                water_score = 0.45
            else:
                water_score = 0.15
        else:
            # No irrigation was needed this period
            water_score = 1.0 if total_applied < 15 else 0.50

        # ---- B: Stress Avoidance (35%) ----
        final_obs = episode.observations[-1]
        stress_days = final_obs.soil.cumulative_stress_days
        waterlog_days = final_obs.soil.waterlog_days
        total_days = len(episode.observations)

        stress_day_fraction = stress_days / max(1, total_days)
        waterlog_fraction = waterlog_days / max(1, total_days)

        stress_score = max(0.0, 1.0 - stress_day_fraction * 3.0 - waterlog_fraction * 4.0)

        # ---- C: Waste Minimization (15%) ----
        total_drainage = sum(
            obs.soil.drainage_mm_today for obs in episode.observations
        )
        total_runoff = sum(
            info.get("runoff_mm", 0) for info in episode.step_infos
        )
        total_waste = total_drainage + total_runoff

        if total_waste < 10:
            waste_score = 1.0
        elif total_waste < 30:
            waste_score = 0.75
        elif total_waste < 60:
            waste_score = 0.45
        else:
            waste_score = 0.10

        # ---- D: Method Efficiency (10%) ----
        methods_used = [
            a.irrigation_method for a in episode.actions if a.irrigate
        ]
        if not methods_used:
            method_score = 0.8  # neutral if no irrigation needed
        else:
            method_score_map = {"drip": 1.0, "sprinkler": 0.85, "furrow": 0.60, "flood": 0.40, "none": 0.0}
            method_score = sum(method_score_map.get(m, 0.5) for m in methods_used) / len(methods_used)

        # ---- Final Weighted Score ----
        final_score = (
            0.40 * water_score +
            0.35 * stress_score +
            0.15 * waste_score +
            0.10 * method_score
        )
        final_score = self._clamp(final_score)

        breakdown = {
            "water_efficiency_score": round(water_score, 3),
            "stress_avoidance_score": round(stress_score, 3),
            "waste_minimization_score": round(waste_score, 3),
            "method_efficiency_score": round(method_score, 3),
            "final_score": round(final_score, 3),
            "total_irrigation_mm": round(episode.total_irrigation_mm, 1),
            "oracle_irrigation_mm": round(total_oracle, 1),
            "stress_days": stress_days,
            "waterlog_days": waterlog_days,
            "total_drainage_mm": round(total_drainage, 1),
            "efficiency_ratio": round(efficiency_ratio if total_oracle > 0 else 0.0, 3),
            "passed": final_score >= 0.65,
        }

        return final_score, breakdown

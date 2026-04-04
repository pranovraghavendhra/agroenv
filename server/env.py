from __future__ import annotations
"""
AgroEnv Core Environment
=========================
Orchestrates all simulation subsystems into a single coherent episode.
This is the central engine — step(), reset(), state() all flow through here.

Architecture:
  AgroEnv owns one instance of each simulation engine.
  On reset(): all engines re-initialize, weather is pre-generated for full season.
  On step(): engines advance in order: weather → soil → crop → pest → market → reward.
  Episode record is updated every step for the grader.
"""

import uuid
import json
import os
from typing import Optional

from .models import (
    AgroAction, AgroObservation, StepResult, ResetRequest, ResetResponse,
    StateResponse, DailyWeatherObs, WeatherForecastDay, SoilObs, CropObs,
    PestObs, MarketObs, ResourceObs,
)
from .config import (
    MAX_STEPS, SEASON_BUDGET_INR, WATER_ALLOCATION_MM,
    IRRIGATION_COST, IRRIGATION_EFFICIENCY, SPRAY_COST_INR,
    HARVEST_COST_INR, TASK_DEFAULTS,
)
from .simulation import (
    WeatherEngine, SoilModel, CropModel, PestModel, MarketEngine,
)
from .tasks import TASK_REGISTRY, EpisodeRecord
from .graders import GRADER_REGISTRY

# Load crop data for total_days lookup
_CROP_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crops.json")
with open(_CROP_DATA_PATH) as f:
    _CROP_DATA = json.load(f)

_SOIL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "soils.json")
with open(_SOIL_DATA_PATH) as f:
    _SOIL_DATA = json.load(f)


class AgroEnv:
    """
    Main AgroEnv environment class.
    One instance per active episode. Stateful — do not share across requests.
    """

    def __init__(self):
        self._episode_id: Optional[str] = None
        self._task_name: Optional[str] = None
        self._crop_key: Optional[str] = None
        self._soil_key: Optional[str] = None
        self._region_key: Optional[str] = None
        self._seed: Optional[int] = None
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0

        # Engines (initialized on reset)
        self._weather: Optional[WeatherEngine] = None
        self._soil: Optional[SoilModel] = None
        self._crop: Optional[CropModel] = None
        self._pest: Optional[PestModel] = None
        self._market: Optional[MarketEngine] = None

        # Task + grader
        self._task = None
        self._grader = None

        # Episode state
        self._record: Optional[EpisodeRecord] = None
        self._last_obs: Optional[AgroObservation] = None

        # Resource tracking
        self._budget_remaining: float = 0.0
        self._water_remaining: float = 0.0
        self._cumulative_irrigation_mm: float = 0.0
        self._spray_events: int = 0
        self._irrigation_events: int = 0
        self._total_cost: float = 0.0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, request: ResetRequest) -> ResetResponse:
        """Initialize a new episode."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._task_name = request.task
        self._seed = request.seed if request.seed is not None else hash(self._episode_id) % 100000

        # Apply task defaults if user didn't specify custom crop/soil/region
        defaults = TASK_DEFAULTS[request.task]
        self._crop_key = request.crop if request.crop != "rice_kharif" or request.task == "irrigation_scheduling" else defaults["crop"]
        self._soil_key = request.soil if request.soil != "loamy_soil" or request.task == "irrigation_scheduling" else defaults["soil"]
        self._region_key = request.region if request.region != "maharashtra_pune" or request.task == "irrigation_scheduling" else defaults["region"]

        # Use task defaults unconditionally for pest/season tasks
        if request.task != "irrigation_scheduling":
            self._crop_key = defaults["crop"]
            self._soil_key = defaults["soil"]
            self._region_key = defaults["region"]

        # Initialize resource trackers
        self._budget_remaining = SEASON_BUDGET_INR[request.task]
        self._water_remaining = WATER_ALLOCATION_MM[request.task]
        self._cumulative_irrigation_mm = 0.0
        self._spray_events = 0
        self._irrigation_events = 0
        self._total_cost = 0.0
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

        # Initialize all engines
        self._weather = WeatherEngine(region_key=self._region_key, seed=self._seed)
        self._soil = SoilModel(soil_key=self._soil_key, root_depth_m=0.5)
        self._crop = CropModel(crop_key=self._crop_key)
        self._pest = PestModel(crop_key=self._crop_key, seed=self._seed)
        self._market = MarketEngine(crop_key=self._crop_key, seed=self._seed)

        # Pre-generate full season weather (allows 7-day forecast from day 1)
        crop_info = _CROP_DATA[self._crop_key]
        total_days = MAX_STEPS[request.task]
        start_month = crop_info["sowing_month"]
        self._weather.generate_season(start_month=start_month, total_days=total_days + 14)

        # Reset all engines to initial state
        soil_state = self._soil.reset()
        crop_state = self._crop.reset()
        pest_states = self._pest.reset()
        market_snap = self._market.reset(start_month=start_month)

        # Initialize task and grader
        task_cls = TASK_REGISTRY[request.task]
        self._task = task_cls()
        grader_cls = GRADER_REGISTRY[request.task]
        self._grader = grader_cls()

        # Initialize episode record
        self._record = EpisodeRecord(
            task_name=request.task,
            crop_key=self._crop_key,
        )

        # Build initial observation (day 0 → use day 1 weather)
        weather_day = self._weather.get_day(1)
        obs = self._build_observation(
            day=0,
            weather_day=weather_day,
            soil_state=soil_state,
            crop_state=crop_state,
            pest_states=pest_states,
            market_snap=market_snap,
            last_action_result="Episode started. Make your first decision.",
        )
        self._last_obs = obs
        self._record.observations.append(obs)

        return ResetResponse(
            observation=obs,
            task_description=self._task.config.description,
            success_criteria=self._task.config.success_criteria,
            max_steps=MAX_STEPS[request.task],
            episode_id=self._episode_id,
        )

    def step(self, action: AgroAction) -> StepResult:
        """Advance the environment by one day."""
        if self._done:
            return StepResult(
                observation=self._last_obs,
                reward=0.0,
                done=True,
                info={"error": "Episode is already done. Call reset() to start a new episode."},
            )

        self._step_count += 1
        day = self._step_count

        # --- Get today's weather ---
        weather_day = self._weather.get_day(day)
        if weather_day is None:
            self._done = True
            return StepResult(
                observation=self._last_obs,
                reward=0.0,
                done=True,
                info={"error": "Season complete — no more weather data."},
            )

        # --- Compute irrigation efficiency ---
        actual_irrigation_mm = 0.0
        irrigation_cost_today = 0.0

        if action.irrigate and action.irrigation_amount_mm > 0:
            method = action.irrigation_method if action.irrigation_method != "none" else "flood"
            efficiency = IRRIGATION_EFFICIENCY.get(method, 0.60)
            gross_mm = action.irrigation_amount_mm
            actual_irrigation_mm = gross_mm * efficiency

            # Cost calculation
            cost_per_mm = IRRIGATION_COST.get(method, 4.0)
            irrigation_cost_today = gross_mm * cost_per_mm

            # Check water budget
            if actual_irrigation_mm > self._water_remaining:
                actual_irrigation_mm = self._water_remaining
                irrigation_cost_today = actual_irrigation_mm / efficiency * cost_per_mm

            self._water_remaining = max(0.0, self._water_remaining - actual_irrigation_mm)
            self._cumulative_irrigation_mm += actual_irrigation_mm
            self._irrigation_events += 1
            self._budget_remaining -= irrigation_cost_today
            self._total_cost += irrigation_cost_today

        # --- Soil update ---
        crop_kc = self._crop.get_kc()
        soil_state, soil_info = self._soil.update(
            et0_mm=weather_day.et0_mm,
            kc=crop_kc,
            rainfall_mm=weather_day.rainfall_mm,
            irrigation_mm=actual_irrigation_mm,
        )
        soil_info["oracle_irrigation_mm"] = self._soil.get_irrigation_recommendation_mm(
            weather_day.et0_mm, crop_kc
        )

        # --- Crop update ---
        total_pest_damage = sum(
            p.damage_accumulated_pct for p in self._pest._build_states()
        ) if self._pest.active_pests else 0.0

        crop_state, crop_info = self._crop.update(
            tmax_c=weather_day.tmax_c,
            tmin_c=weather_day.tmin_c,
            ks=soil_state.ks,
            pest_damage_pct=total_pest_damage * 0.01,
        )

        # --- Pest update ---
        spray_map = {d.pest_name: d.pesticide for d in action.spray_decisions}
        spray_cost_today = 0.0

        for pest_name, pesticide in spray_map.items():
            if pesticide != "none":
                self._spray_events += 1
                spray_cost_today += SPRAY_COST_INR
                self._budget_remaining -= SPRAY_COST_INR
                self._total_cost += SPRAY_COST_INR

        pest_states, pest_info = self._pest.update(
            temperature_c=weather_day.tmean_c,
            humidity_pct=weather_day.humidity_pct,
            growth_stage=crop_state.growth_stage,
            spray_actions=spray_map,
        )

        # --- Market update ---
        market_snap = self._market.update(
            day=day,
            season_day=day,
            quality_grade="B",
        )

        # --- Harvest handling ---
        harvest_cost_today = 0.0
        if action.harvest_now and self._record.harvest_day is None:
            self._record.harvest_day = day
            self._record.harvest_gdd = crop_state.gdd_accumulated
            harvest_cost_today = HARVEST_COST_INR
            self._budget_remaining -= HARVEST_COST_INR
            self._total_cost += HARVEST_COST_INR

            # Compute final yield and revenue
            final_yield = self._crop.estimate_final_yield()
            quality = self._determine_quality(crop_state, soil_state, pest_states)
            revenue_data = self._market.compute_revenue(final_yield, quality)
            self._record.final_yield_ton_per_ha = final_yield
            self._record.final_revenue_inr_per_ha = revenue_data["revenue_inr_per_ha"]

        # --- Build new observation ---
        price_outlook = self._market.get_price_outlook()
        new_obs = self._build_observation(
            day=day,
            weather_day=weather_day,
            soil_state=soil_state,
            crop_state=crop_state,
            pest_states=pest_states,
            market_snap=market_snap,
            price_outlook=price_outlook,
            irrigation_cost_today=irrigation_cost_today,
            spray_cost_today=spray_cost_today,
        )

        # --- Compute step reward ---
        step_info = {
            **soil_info,
            **crop_info,
            **pest_info,
            "irrigation_cost_today": irrigation_cost_today,
            "spray_cost_today": spray_cost_today,
        }

        reward, feedback = self._task.compute_step_reward(
            action=action,
            obs_before=self._last_obs,
            obs_after=new_obs,
            step_info=step_info,
        )

        new_obs.last_action_result = feedback
        new_obs.episode_reward_so_far = self._total_reward + reward

        # --- Update record ---
        self._record.days.append(day)
        self._record.actions.append(action)
        self._record.observations.append(new_obs)
        self._record.step_rewards.append(reward)
        self._record.step_infos.append(step_info)
        self._record.total_irrigation_mm = self._cumulative_irrigation_mm
        self._record.total_spray_events = self._spray_events
        self._record.total_cost_inr = self._total_cost
        self._record.correct_sprays += pest_info.get("correct_sprays", 0)
        self._record.unnecessary_sprays += pest_info.get("unnecessary_sprays", 0)

        self._total_reward += reward
        self._last_obs = new_obs

        # --- Check done ---
        done = self._task.is_done(new_obs, self._step_count, self._record)

        # If done and no harvest yet for season_optimizer — compute end-of-season yield
        if done and self._record.harvest_day is None and self._task_name == "season_optimizer":
            final_yield = self._crop.estimate_final_yield() * 0.85  # penalty for not harvesting
            self._record.final_yield_ton_per_ha = final_yield
            self._record.final_revenue_inr_per_ha = 0.0  # not sold

        if done and self._record.harvest_day is None and self._task_name != "season_optimizer":
            # For other tasks, estimate yield at end
            self._record.final_yield_ton_per_ha = self._crop.estimate_final_yield()

        self._done = done

        return StepResult(
            observation=new_obs,
            reward=round(reward, 4),
            done=done,
            info={
                **step_info,
                "episode_id": self._episode_id,
                "step": self._step_count,
                "total_reward": round(self._total_reward, 4),
                "budget_remaining": round(self._budget_remaining, 2),
                "water_remaining_mm": round(self._water_remaining, 2),
            },
        )

    def state(self) -> StateResponse:
        """Return lightweight episode state (no full observation)."""
        return StateResponse(
            episode_id=self._episode_id or "none",
            task=self._task_name or "irrigation_scheduling",
            day=self._step_count,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            steps_taken=self._step_count,
            crop=self._crop_key or "unknown",
            region=self._region_key or "unknown",
            soil=self._soil_key or "unknown",
        )

    def grade(self) -> tuple[float, dict]:
        """Run the grader on the completed episode. Can be called anytime."""
        if self._grader is None or self._record is None:
            return 0.0, {"error": "No episode to grade"}
        return self._grader.grade(self._record)

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        day: int,
        weather_day,
        soil_state,
        crop_state,
        pest_states,
        market_snap,
        price_outlook: dict = None,
        last_action_result: str = None,
        irrigation_cost_today: float = 0.0,
        spray_cost_today: float = 0.0,
    ) -> AgroObservation:
        """Assemble full typed observation from all subsystem states."""

        soil_profile = _SOIL_DATA[self._soil_key]
        crop_info = _CROP_DATA[self._crop_key]

        # Weather today
        weather_obs = DailyWeatherObs(
            tmax_c=weather_day.tmax_c,
            tmin_c=weather_day.tmin_c,
            tmean_c=weather_day.tmean_c,
            humidity_pct=weather_day.humidity_pct,
            rainfall_mm=weather_day.rainfall_mm,
            solar_radiation_mj_m2=weather_day.solar_radiation_mj_m2,
            et0_mm=weather_day.et0_mm,
            wind_speed_ms=weather_day.wind_speed_ms,
        )

        # 7-day weather forecast
        forecast_raw = self._weather.get_forecast(day, horizon_days=7)
        forecast = [
            WeatherForecastDay(
                day_ahead=f["day_ahead"],
                tmax_c=f["tmax_c"],
                tmin_c=f["tmin_c"],
                rain_prob_pct=f["rain_prob_pct"],
                expected_rain_mm=f["expected_rain_mm"],
                et0_forecast_mm=f["et0_forecast_mm"],
                humidity_pct=f["humidity_pct"],
                forecast_confidence=f["forecast_confidence"],
            )
            for f in forecast_raw
        ]

        # Soil observation
        soil_obs = SoilObs(
            moisture_pct=soil_state.moisture_pct,
            field_capacity_pct=soil_profile["field_capacity_pct"],
            wilting_point_pct=soil_profile["wilting_point_pct"],
            depletion_mm=soil_state.depletion_mm,
            ks=soil_state.ks,
            drainage_mm_today=soil_state.drainage_mm_today,
            cumulative_stress_days=soil_state.cumulative_stress_days,
            waterlog_days=soil_state.waterlog_days,
            raw_mm=soil_state.raw_mm,
        )

        # Crop observation
        crop_obs = CropObs(
            crop_name=crop_info["name"],
            growth_stage=crop_state.growth_stage,
            day_of_season=crop_state.day,
            total_season_days=crop_info["total_days"],
            gdd_accumulated=crop_state.gdd_accumulated,
            gdd_progress_pct=crop_state.gdd_progress_pct,
            ndvi=crop_state.ndvi,
            lai=crop_state.lai,
            canopy_cover_pct=crop_state.canopy_cover_pct,
            kc=crop_state.kc,
            estimated_yield_pct=crop_state.estimated_yield_pct,
            in_harvest_window=crop_state.in_harvest_window,
            days_to_harvest_window=crop_state.days_to_harvest_window,
            harvest_window_closing_days=crop_state.harvest_window_closing_days,
        )

        # Pest observations
        pest_obs_list = []
        for ps in pest_states:
            from .simulation.pest_model import PEST_PROFILES
            profile = PEST_PROFILES.get(ps.pest_name, {})
            pest_obs_list.append(PestObs(
                pest_name=ps.pest_name,
                population=ps.population,
                economic_threshold=profile.get("economic_threshold", 5),
                economic_injury_level=profile.get("economic_injury_level", 10),
                at_threshold=ps.at_threshold,
                above_eil=ps.above_eil,
                resistance_index=ps.resistance_index,
                days_since_spray=ps.days_since_spray,
                natural_enemy_population=ps.natural_enemy_population,
                damage_accumulated_pct=ps.damage_accumulated_pct,
            ))

        # Market observation
        if price_outlook is None:
            price_outlook = self._market.get_price_outlook()

        market_obs = MarketObs(
            current_price_inr_per_quintal=market_snap.current_price_inr_per_quintal,
            msp_inr_per_quintal=market_snap.msp_inr_per_quintal,
            price_vs_msp_pct=market_snap.price_vs_msp_pct,
            market_trend=market_snap.market_trend,
            days_to_peak_price=market_snap.days_to_peak_price,
            glut_risk_pct=market_snap.glut_risk_pct,
            price_3d_ahead=price_outlook.get("price_3d_ahead", market_snap.current_price_inr_per_quintal),
            price_7d_ahead=price_outlook.get("price_7d_ahead", market_snap.current_price_inr_per_quintal),
            price_15d_ahead=price_outlook.get("price_15d_ahead", market_snap.current_price_inr_per_quintal),
        )

        # Resource observation
        resource_obs = ResourceObs(
            budget_remaining_inr=round(self._budget_remaining, 2),
            water_available_mm=round(self._water_remaining, 2),
            irrigation_events_used=self._irrigation_events,
            spray_events_used=self._spray_events,
            cumulative_irrigation_mm=round(self._cumulative_irrigation_mm, 2),
            cost_irrigation_today_inr=round(irrigation_cost_today, 2),
            cost_spray_today_inr=round(spray_cost_today, 2),
        )

        return AgroObservation(
            task=self._task_name,
            day=day,
            weather_today=weather_obs,
            weather_forecast=forecast,
            soil=soil_obs,
            crop=crop_obs,
            pests=pest_obs_list,
            market=market_obs,
            resources=resource_obs,
            last_action_result=last_action_result,
            episode_reward_so_far=self._total_reward,
            info_message=self._generate_info_message(soil_state, crop_state, pest_states),
        )

    def _determine_quality(self, crop_state, soil_state, pest_states) -> str:
        """Determine harvest quality grade based on accumulated stresses."""
        stress_score = crop_state.cumulative_water_stress
        pest_damage = sum(p.damage_accumulated_pct for p in pest_states)

        if stress_score < 0.5 and pest_damage < 5 and crop_state.in_harvest_window:
            return "A"
        elif stress_score < 1.5 and pest_damage < 15:
            return "B"
        else:
            return "C"

    def _generate_info_message(self, soil_state, crop_state, pest_states) -> str:
        """Generate critical advisory messages for the agent."""
        messages = []

        if soil_state.ks < 0.50:
            messages.append("⚠️ CRITICAL: Severe water stress (Ks={:.2f}) — immediate irrigation required".format(soil_state.ks))
        if soil_state.waterlog_days > 0 and soil_state.moisture_pct > 90:
            messages.append("⚠️ WARNING: Waterlogging detected — halt irrigation")
        if crop_state.in_harvest_window:
            messages.append("🌾 HARVEST WINDOW OPEN — GDD {:.0f} within optimal range".format(crop_state.gdd_accumulated))
        if crop_state.harvest_window_closing_days <= 5 and crop_state.harvest_window_closing_days > 0:
            messages.append("⏰ URGENT: Harvest window closes in {} days!".format(crop_state.harvest_window_closing_days))
        for ps in pest_states:
            if ps.above_eil:
                messages.append("🐛 CRITICAL: {} above Economic Injury Level — yield loss active".format(ps.pest_name))
            elif ps.at_threshold:
                messages.append("⚠️ {} at Economic Threshold — spray warranted".format(ps.pest_name))

        return " | ".join(messages) if messages else None

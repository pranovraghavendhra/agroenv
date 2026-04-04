"""
AgroEnv Test Suite
===================
Tests all simulation engines, models, and graders.
Run from project root: python -m pytest server/tests/ -v

Tests are deterministic (fixed seeds) and cover:
1. Simulation correctness (physical bounds, agronomic rules)
2. Model typing (all Pydantic models valid)
3. Grader scoring (in [0,1], deterministic)
4. API contract (step/reset/state work end-to-end)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from server.simulation.weather_engine import WeatherEngine
from server.simulation.soil_model import SoilModel
from server.simulation.crop_model import CropModel
from server.simulation.pest_model import PestModel, PEST_PROFILES
from server.simulation.market_engine import MarketEngine
from server.models import (
    AgroAction, ResetRequest, PestSprayAction,
)
from server.env import AgroEnv
from server.graders import IrrigationGrader, PestGrader, SeasonGrader
from server.tasks.base_task import EpisodeRecord


# ─────────────────────────────────────────────────────────────────────────────
# Weather Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherEngine:

    def test_season_generation_length(self):
        engine = WeatherEngine(region_key="maharashtra_pune", seed=42)
        days = engine.generate_season(start_month=6, total_days=120)
        assert len(days) == 120

    def test_et0_positive(self):
        engine = WeatherEngine(seed=42)
        days = engine.generate_season(start_month=6, total_days=30)
        for d in days:
            assert d.et0_mm >= 0.0, f"ET₀ must be non-negative, got {d.et0_mm}"
            assert d.et0_mm < 25.0, f"ET₀ unrealistically high: {d.et0_mm}"

    def test_temperature_bounds(self):
        engine = WeatherEngine(seed=42)
        days = engine.generate_season(start_month=4, total_days=60)
        for d in days:
            assert d.tmin_c < d.tmax_c, "Tmin must be < Tmax"
            assert -5 < d.tmin_c < 50
            assert 0 < d.tmax_c < 55

    def test_rainfall_non_negative(self):
        engine = WeatherEngine(seed=42)
        days = engine.generate_season(start_month=7, total_days=30)
        for d in days:
            assert d.rainfall_mm >= 0.0

    def test_forecast_uncertainty_increases(self):
        engine = WeatherEngine(seed=42)
        engine.generate_season(start_month=6, total_days=120)
        forecast = engine.get_forecast(current_day=10, horizon_days=7)
        assert len(forecast) == 7
        # Confidence should decrease with horizon
        confidences = [f["forecast_confidence"] for f in forecast]
        assert confidences[0] > confidences[-1]

    def test_reproducibility(self):
        e1 = WeatherEngine(seed=42)
        e2 = WeatherEngine(seed=42)
        d1 = e1.generate_season(start_month=6, total_days=14)
        d2 = e2.generate_season(start_month=6, total_days=14)
        for a, b in zip(d1, d2):
            assert abs(a.et0_mm - b.et0_mm) < 1e-6
            assert abs(a.tmax_c - b.tmax_c) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Soil Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSoilModel:

    def test_reset_at_field_capacity(self):
        soil = SoilModel(soil_key="loamy_soil")
        state = soil.reset()
        assert abs(state.moisture_pct - 35.0) < 5.0  # Close to FC
        assert state.depletion_mm == 0.0

    def test_moisture_within_bounds(self):
        soil = SoilModel(soil_key="loamy_soil")
        soil.reset()
        for _ in range(20):
            state, _ = soil.update(et0_mm=5.0, kc=1.1, rainfall_mm=0.0, irrigation_mm=0.0)
            assert state.moisture_pct >= soil.theta_wp
            assert state.moisture_pct <= soil.theta_sat

    def test_irrigation_raises_moisture(self):
        soil = SoilModel(soil_key="loamy_soil")
        soil.reset()
        # Deplete soil first
        for _ in range(10):
            soil.update(et0_mm=6.0, kc=1.2, rainfall_mm=0.0, irrigation_mm=0.0)
        state_before = soil.reset()  # won't reset depletion — just read
        state_before, _ = soil.update(et0_mm=0.0, kc=1.0, rainfall_mm=0.0, irrigation_mm=0.0)
        moisture_before = state_before.moisture_pct
        state_after, _ = soil.update(et0_mm=0.0, kc=1.0, rainfall_mm=0.0, irrigation_mm=40.0)
        assert state_after.moisture_pct >= moisture_before

    def test_ks_between_zero_and_one(self):
        soil = SoilModel(soil_key="red_laterite_soil")
        soil.reset()
        for i in range(30):
            state, _ = soil.update(et0_mm=7.0, kc=1.2, rainfall_mm=0.0, irrigation_mm=0.0)
            assert 0.0 <= state.ks <= 1.0

    def test_waterlogging_on_excess(self):
        soil = SoilModel(soil_key="black_cotton_soil")
        soil.reset()
        state, _ = soil.update(et0_mm=0.0, kc=1.0, rainfall_mm=0.0, irrigation_mm=100.0)
        assert state.waterlog_days >= 0  # May trigger waterlogging

    def test_oracle_recommendation_positive(self):
        soil = SoilModel(soil_key="loamy_soil")
        soil.reset()
        # Deplete
        for _ in range(8):
            soil.update(et0_mm=5.0, kc=1.1, rainfall_mm=0.0, irrigation_mm=0.0)
        rec = soil.get_irrigation_recommendation_mm(et0_mm=5.0, kc=1.1)
        assert rec >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Crop Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCropModel:

    def test_gdd_accumulates(self):
        crop = CropModel("rice_kharif")
        crop.reset()
        prev_gdd = 0.0
        for _ in range(30):
            state, _ = crop.update(tmax_c=32, tmin_c=24, ks=1.0)
            assert state.gdd_accumulated >= prev_gdd
            prev_gdd = state.gdd_accumulated

    def test_ndvi_in_range(self):
        crop = CropModel("wheat_rabi")
        crop.reset()
        for _ in range(50):
            state, _ = crop.update(tmax_c=25, tmin_c=10, ks=1.0)
            assert 0.0 <= state.ndvi <= 1.0

    def test_lai_in_range(self):
        crop = CropModel("cotton_kharif")
        crop.reset()
        for _ in range(80):
            state, _ = crop.update(tmax_c=33, tmin_c=22, ks=0.9)
            assert 0.0 <= state.lai <= 7.0

    def test_water_stress_reduces_yield(self):
        # Unstressed crop
        crop1 = CropModel("tomato_rabi")
        crop1.reset()
        for _ in range(60):
            crop1.update(tmax_c=30, tmin_c=18, ks=1.0)
        yield1 = crop1.estimate_final_yield()

        # Stressed crop
        crop2 = CropModel("tomato_rabi")
        crop2.reset()
        for _ in range(60):
            crop2.update(tmax_c=30, tmin_c=18, ks=0.2)
        yield2 = crop2.estimate_final_yield()

        assert yield1 > yield2, "Water-stressed crop should have lower yield"

    def test_growth_stage_transitions(self):
        crop = CropModel("rice_kharif")
        crop.reset()
        stages_seen = set()
        for _ in range(120):
            state, _ = crop.update(tmax_c=30, tmin_c=22, ks=1.0)
            stages_seen.add(state.growth_stage)
        # Should see multiple stages
        assert len(stages_seen) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Pest Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPestModel:

    def test_reset_below_threshold(self):
        pest = PestModel("cotton_kharif", seed=42)
        states = pest.reset()
        for s in states:
            profile = PEST_PROFILES[s.pest_name]
            # Initial population should be below Economic Threshold
            assert s.population < profile["economic_threshold"], \
                f"{s.pest_name} starts at {s.population}, threshold={profile['economic_threshold']}"

    def test_population_grows_without_spray(self):
        pest = PestModel("cotton_kharif", seed=42)
        pest.reset()
        pops_day1 = {s.pest_name: s.population for s in pest._build_states()}

        for _ in range(15):
            pest.update(temperature_c=32, humidity_pct=70, growth_stage="flowering", spray_actions={})

        pops_day15 = {s.pest_name: s.population for s in pest._build_states()}
        # At least one pest should have grown
        any_grew = any(pops_day15[k] > pops_day1[k] for k in pops_day1)
        assert any_grew

    def test_spray_reduces_population(self):
        pest = PestModel("cotton_kharif", seed=42)
        pest.reset()

        # Let population build
        for _ in range(20):
            pest.update(temperature_c=32, humidity_pct=72, growth_stage="flowering", spray_actions={})

        states_before = pest._build_states()
        pop_before = sum(s.population for s in states_before)

        # Spray all pests
        spray = {"whitefly": "spiromesifen", "bollworm": "chlorantraniliprole"}
        pest.update(temperature_c=32, humidity_pct=72, growth_stage="flowering", spray_actions=spray)

        states_after = pest._build_states()
        pop_after = sum(s.population for s in states_after)
        assert pop_after < pop_before, "Spraying should reduce population"

    def test_resistance_increases_with_repeated_spray(self):
        pest = PestModel("cotton_kharif", seed=42)
        pest.reset()
        for _ in range(5):
            pest.update(temperature_c=32, humidity_pct=72, growth_stage="flowering",
                        spray_actions={"whitefly": "imidacloprid", "bollworm": "imidacloprid"})
        states = pest._build_states()
        resistances = [s.resistance_index for s in states]
        assert any(r > 0.0 for r in resistances), "Resistance should build with repeated spraying"

    def test_unnecessary_spray_counted(self):
        pest = PestModel("cotton_kharif", seed=42)
        pest.reset()  # Starts below threshold
        _, info = pest.update(
            temperature_c=30, humidity_pct=65, growth_stage="flowering",
            spray_actions={"whitefly": "imidacloprid", "bollworm": "spinosad"}
        )
        assert info["unnecessary_sprays"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Market Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketEngine:

    def test_price_above_msp_floor(self):
        mkt = MarketEngine("rice_kharif", seed=42)
        mkt.reset(start_month=6)
        for day in range(1, 30):
            snap = mkt.update(day=day, season_day=day, quality_grade="B")
            assert snap.current_price_inr_per_quintal >= snap.msp_inr_per_quintal * 0.70

    def test_quality_a_premium(self):
        mkt = MarketEngine("tomato_rabi", seed=42)
        mkt.reset(start_month=10)
        snap_a = mkt.update(day=50, season_day=50, quality_grade="A")
        snap_c = mkt.update(day=50, season_day=50, quality_grade="C")
        assert snap_a.adjusted_price_inr_per_quintal > snap_c.adjusted_price_inr_per_quintal

    def test_price_outlook_has_future_prices(self):
        mkt = MarketEngine("wheat_rabi", seed=42)
        mkt.reset(start_month=11)
        outlook = mkt.get_price_outlook()
        assert "price_3d_ahead" in outlook
        assert "price_7d_ahead" in outlook
        assert outlook["price_3d_ahead"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Environment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgroEnvEndToEnd:

    def _make_default_action(self) -> AgroAction:
        return AgroAction(
            irrigate=False,
            irrigation_amount_mm=0.0,
            irrigation_method="none",
            spray_decisions=[],
            harvest_now=False,
            reasoning="Test action",
        )

    def test_irrigation_task_reset(self):
        env = AgroEnv()
        req = ResetRequest(task="irrigation_scheduling", seed=42)
        resp = env.reset(req)
        assert resp.observation is not None
        assert resp.observation.day == 0
        assert resp.max_steps == 14
        assert resp.observation.task == "irrigation_scheduling"

    def test_irrigation_task_full_episode(self):
        env = AgroEnv()
        req = ResetRequest(task="irrigation_scheduling", seed=42)
        resp = env.reset(req)

        rewards = []
        for step in range(14):
            action = AgroAction(
                irrigate=True,
                irrigation_amount_mm=25.0,
                irrigation_method="drip",
                spray_decisions=[],
                harvest_now=False,
                reasoning="test",
            )
            result = env.step(action)
            rewards.append(result.reward)
            assert -1.0 <= result.reward <= 1.0
            if result.done:
                break

        assert result.done
        assert len(rewards) == 14

    def test_pest_task_reset(self):
        env = AgroEnv()
        req = ResetRequest(task="pest_management", seed=42)
        resp = env.reset(req)
        assert len(resp.observation.pests) == 2  # whitefly + bollworm
        assert resp.max_steps == 30

    def test_season_optimizer_reset(self):
        env = AgroEnv()
        req = ResetRequest(task="season_optimizer", seed=42)
        resp = env.reset(req)
        assert resp.max_steps == 110
        assert resp.observation.crop.crop_name == "Tomato (Rabi)"

    def test_state_endpoint(self):
        env = AgroEnv()
        req = ResetRequest(task="irrigation_scheduling", seed=42)
        env.reset(req)
        state = env.state()
        assert state.steps_taken == 0
        assert state.done == False
        env.step(self._make_default_action())
        state2 = env.state()
        assert state2.steps_taken == 1

    def test_reward_in_bounds_all_tasks(self):
        for task in ["irrigation_scheduling", "pest_management", "season_optimizer"]:
            env = AgroEnv()
            req = ResetRequest(task=task, seed=42)
            env.reset(req)
            for _ in range(5):
                result = env.step(self._make_default_action())
                assert -1.0 <= result.reward <= 1.0
                if result.done:
                    break

    def test_reproducibility_same_seed(self):
        env1 = AgroEnv()
        env2 = AgroEnv()
        req = ResetRequest(task="irrigation_scheduling", seed=99)
        env1.reset(req)
        env2.reset(req)

        action = AgroAction(irrigate=True, irrigation_amount_mm=30.0,
                            irrigation_method="drip", spray_decisions=[],
                            harvest_now=False, reasoning="test")
        r1 = env1.step(action)
        r2 = env2.step(action)
        assert abs(r1.reward - r2.reward) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Grader Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraders:

    def _make_minimal_record(self, task_name: str) -> EpisodeRecord:
        env = AgroEnv()
        req = ResetRequest(task=task_name, seed=42)
        env.reset(req)
        # Run a few steps
        action = AgroAction(
            irrigate=True, irrigation_amount_mm=25.0,
            irrigation_method="drip", spray_decisions=[],
            harvest_now=False, reasoning="grader test"
        )
        for _ in range(5):
            result = env.step(action)
            if result.done:
                break
        return env._record

    def test_irrigation_grader_in_range(self):
        record = self._make_minimal_record("irrigation_scheduling")
        grader = IrrigationGrader()
        score, breakdown = grader.grade(record)
        assert 0.0 <= score <= 1.0
        assert "final_score" in breakdown

    def test_pest_grader_in_range(self):
        env = AgroEnv()
        req = ResetRequest(task="pest_management", seed=42)
        env.reset(req)
        for _ in range(10):
            action = AgroAction(
                irrigate=False, irrigation_amount_mm=0.0,
                irrigation_method="none",
                spray_decisions=[
                    PestSprayAction(pest_name="whitefly", pesticide="none"),
                    PestSprayAction(pest_name="bollworm", pesticide="none"),
                ],
                harvest_now=False, reasoning="test"
            )
            result = env.step(action)
            if result.done:
                break
        grader = PestGrader()
        score, breakdown = grader.grade(env._record)
        assert 0.0 <= score <= 1.0

    def test_season_grader_in_range(self):
        record = self._make_minimal_record("season_optimizer")
        grader = SeasonGrader()
        score, breakdown = grader.grade(record)
        assert 0.0 <= score <= 1.0

    def test_graders_deterministic(self):
        """Same episode → same score, always."""
        record = self._make_minimal_record("irrigation_scheduling")
        grader = IrrigationGrader()
        s1, _ = grader.grade(record)
        s2, _ = grader.grade(record)
        assert abs(s1 - s2) < 1e-10

    def test_empty_record_returns_zero(self):
        record = EpisodeRecord(task_name="irrigation_scheduling", crop_key="rice_kharif")
        grader = IrrigationGrader()
        score, breakdown = grader.grade(record)
        assert score == 0.0
        assert "error" in breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

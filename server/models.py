"""
AgroEnv Typed Models
=====================
All Pydantic models defining the agent-environment contract.
These are the strict typed interfaces enforced at every API boundary.

Design principles:
- Every field has a docstring-style description (Field(..., description=...))
- Observation is information-rich but agent must interpret it intelligently
- Action is constrained to realistic agronomic choices only
- No free-form text in actions — forces structured decision making
"""

from __future__ import annotations
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
import math


# ---------------------------------------------------------------------------
# Enums / Literal Types
# ---------------------------------------------------------------------------

CropKey = Literal["rice_kharif", "wheat_rabi", "cotton_kharif", "tomato_rabi"]
SoilKey = Literal["black_cotton_soil", "alluvial_soil", "red_laterite_soil", "loamy_soil"]
RegionKey = Literal["maharashtra_pune", "punjab_ludhiana", "andhra_guntur"]
IrrigationMethod = Literal["drip", "sprinkler", "flood", "furrow", "none"]
PesticideChoice = Literal[
    "imidacloprid", "buprofezin", "thiamethoxam", "chlorantraniliprole",
    "fipronil", "cartap", "dimethoate", "spiromesifen", "pyriproxyfen",
    "spinosad", "indoxacarb", "mancozeb", "chlorothalonil", "azoxystrobin",
    "emamectin_benzoate", "neem_oil", "none"
]
TaskName = Literal["irrigation_scheduling", "pest_management", "season_optimizer"]
QualityGrade = Literal["A", "B", "C"]


# ---------------------------------------------------------------------------
# Weather Sub-Models
# ---------------------------------------------------------------------------

class DailyWeatherObs(BaseModel):
    tmax_c: float = Field(..., description="Maximum temperature today (°C)")
    tmin_c: float = Field(..., description="Minimum temperature today (°C)")
    tmean_c: float = Field(..., description="Mean temperature today (°C)")
    humidity_pct: float = Field(..., description="Relative humidity (%)")
    rainfall_mm: float = Field(..., description="Rainfall today (mm)")
    solar_radiation_mj_m2: float = Field(..., description="Solar radiation (MJ/m²/day)")
    et0_mm: float = Field(..., description="FAO-56 reference ET₀ (mm/day) — water demand of reference grass")
    wind_speed_ms: float = Field(..., description="Wind speed at 2m (m/s)")


class WeatherForecastDay(BaseModel):
    day_ahead: int = Field(..., description="Days ahead from today")
    tmax_c: float = Field(..., description="Forecast max temperature (°C)")
    tmin_c: float = Field(..., description="Forecast min temperature (°C)")
    rain_prob_pct: float = Field(..., description="Probability of rain (%)")
    expected_rain_mm: float = Field(..., description="Expected rainfall if rain occurs (mm)")
    et0_forecast_mm: float = Field(..., description="Forecast reference ET₀ (mm/day)")
    humidity_pct: float = Field(..., description="Forecast relative humidity (%)")
    forecast_confidence: float = Field(..., description="Confidence in forecast (0–1, decreases with horizon)")


# ---------------------------------------------------------------------------
# Soil Sub-Models
# ---------------------------------------------------------------------------

class SoilObs(BaseModel):
    moisture_pct: float = Field(..., description="Current soil volumetric moisture content (%)")
    field_capacity_pct: float = Field(..., description="Field capacity of this soil type (%) — upper limit for irrigation")
    wilting_point_pct: float = Field(..., description="Permanent wilting point (%) — plant cannot extract water below this")
    depletion_mm: float = Field(..., description="Root zone depletion from field capacity (mm) — 0 means fully recharged")
    ks: float = Field(..., description="Water stress coefficient (0=full stress, 1=no stress) — affects crop ET and growth")
    drainage_mm_today: float = Field(..., description="Water lost to deep drainage today (mm) — indicates over-irrigation")
    cumulative_stress_days: int = Field(..., description="Total days crop has experienced moisture stress this season")
    waterlog_days: int = Field(..., description="Total days soil has been waterlogged this season")
    raw_mm: float = Field(..., description="Readily Available Water (mm) — depletion threshold before stress begins")


# ---------------------------------------------------------------------------
# Crop Sub-Models
# ---------------------------------------------------------------------------

class CropObs(BaseModel):
    crop_name: str = Field(..., description="Crop variety name")
    growth_stage: str = Field(..., description="Current phenological stage (e.g. 'tillering', 'flowering')")
    day_of_season: int = Field(..., description="Day number within the crop season (1 = sowing day)")
    total_season_days: int = Field(..., description="Total duration of this crop's season")
    gdd_accumulated: float = Field(..., description="Growing Degree Days accumulated so far")
    gdd_progress_pct: float = Field(..., description="Season progress by GDD (%)")
    ndvi: float = Field(..., description="Simulated satellite NDVI (0–1): derived from Leaf Area Index via Beer-Lambert law. Higher = healthier canopy")
    lai: float = Field(..., description="Leaf Area Index (m²/m²) — canopy density indicator")
    canopy_cover_pct: float = Field(..., description="Ground area covered by crop canopy (%)")
    kc: float = Field(..., description="Current crop coefficient (Kc) — multiplied by ET₀ to get crop water demand")
    estimated_yield_pct: float = Field(..., description="Estimated final yield as % of max possible, given stresses so far")
    in_harvest_window: bool = Field(..., description="True if GDD is within optimal harvest window")
    days_to_harvest_window: int = Field(..., description="Days estimated until harvest window opens (0 = already open)")
    harvest_window_closing_days: int = Field(..., description="Days until optimal harvest window closes (0 = closed/missed)")


# ---------------------------------------------------------------------------
# Pest Sub-Models
# ---------------------------------------------------------------------------

class PestObs(BaseModel):
    pest_name: str = Field(..., description="Pest/disease identifier (e.g. 'brown_planthopper')")
    population: float = Field(..., description="Current population (scale: per-unit as per ICAR EIL definition)")
    economic_threshold: float = Field(..., description="ICAR Economic Threshold — action required above this level")
    economic_injury_level: float = Field(..., description="ICAR Economic Injury Level — yield loss begins above this")
    at_threshold: bool = Field(..., description="Population has reached/exceeded Economic Threshold — spray warranted")
    above_eil: bool = Field(..., description="Population above Economic Injury Level — active yield loss occurring")
    resistance_index: float = Field(..., description="Pesticide resistance buildup (0=none, 1=full resistance) — affects efficacy")
    days_since_spray: int = Field(..., description="Days since last pesticide application for this pest")
    natural_enemy_population: float = Field(..., description="Population of natural enemies/predators (biological control indicator)")
    damage_accumulated_pct: float = Field(..., description="Cumulative yield damage caused by this pest this season (%)")


# ---------------------------------------------------------------------------
# Market Sub-Models
# ---------------------------------------------------------------------------

class MarketObs(BaseModel):
    current_price_inr_per_quintal: float = Field(..., description="Current mandi price (INR/quintal)")
    msp_inr_per_quintal: float = Field(..., description="Government Minimum Support Price (INR/quintal)")
    price_vs_msp_pct: float = Field(..., description="Current price as % above/below MSP")
    market_trend: str = Field(..., description="Price trend: 'rising', 'falling', or 'stable'")
    days_to_peak_price: int = Field(..., description="Estimated days until price peaks in this season")
    glut_risk_pct: float = Field(..., description="Risk of harvest glut price crash if everyone harvests now (%)")
    price_3d_ahead: float = Field(..., description="Projected price 3 days from now (INR/quintal)")
    price_7d_ahead: float = Field(..., description="Projected price 7 days from now (INR/quintal)")
    price_15d_ahead: float = Field(..., description="Projected price 15 days from now (INR/quintal)")


# ---------------------------------------------------------------------------
# Budget / Resource Tracker
# ---------------------------------------------------------------------------

class ResourceObs(BaseModel):
    budget_remaining_inr: float = Field(..., description="Remaining seasonal budget (INR/ha)")
    water_available_mm: float = Field(..., description="Available water in irrigation reservoir (mm equivalent)")
    irrigation_events_used: int = Field(..., description="Number of irrigation events used this season")
    spray_events_used: int = Field(..., description="Number of pesticide spray events used this season")
    cumulative_irrigation_mm: float = Field(..., description="Total irrigation water applied this season (mm)")
    cost_irrigation_today_inr: float = Field(..., description="Cost of today's irrigation action (INR/ha)")
    cost_spray_today_inr: float = Field(..., description="Cost of today's spray action (INR/ha)")


# ---------------------------------------------------------------------------
# Main Observation Model
# ---------------------------------------------------------------------------

class AgroObservation(BaseModel):
    """
    Complete observation returned to agent at each step.
    Contains all information the agent needs to make a decision.
    """
    task: TaskName = Field(..., description="Current task type")
    day: int = Field(..., description="Current day of season (1-indexed)")

    weather_today: DailyWeatherObs
    weather_forecast: List[WeatherForecastDay] = Field(..., description="7-day weather forecast with uncertainty")

    soil: SoilObs
    crop: CropObs
    pests: List[PestObs] = Field(default_factory=list, description="Status of all tracked pest species")
    market: MarketObs

    resources: ResourceObs

    last_action_result: Optional[str] = Field(None, description="Feedback on the previous action taken")
    episode_reward_so_far: float = Field(0.0, description="Cumulative reward earned this episode")
    info_message: Optional[str] = Field(None, description="Advisory message from the environment (e.g. critical warning)")


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class PestSprayAction(BaseModel):
    """Spray decision for a specific pest."""
    pest_name: str = Field(..., description="Name of pest to treat")
    pesticide: PesticideChoice = Field(..., description="Pesticide to apply. Use 'none' to skip")


class AgroAction(BaseModel):
    """
    Agent's decision for the current day.
    All fields are optional — agent only specifies what it wants to do.
    Omitted fields default to 'do nothing'.
    """

    # Irrigation decision
    irrigate: bool = Field(False, description="Whether to irrigate today")
    irrigation_amount_mm: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Amount of irrigation to apply (mm). Only used if irrigate=True. Range: 0–100mm"
    )
    irrigation_method: IrrigationMethod = Field(
        "none",
        description="Irrigation delivery method. Affects water use efficiency"
    )

    # Pest management decisions
    spray_decisions: List[PestSprayAction] = Field(
        default_factory=list,
        description="Spray decisions per pest. Omit pests to skip spraying them"
    )

    # Harvest decision (only valid when in_harvest_window=True)
    harvest_now: bool = Field(
        False,
        description="Harvest the crop today. Only scores positively within the harvest window"
    )

    # Agent reasoning (logged but NOT scored — prevents reward hacking via verbose justification)
    reasoning: str = Field(
        "",
        max_length=500,
        description="Agent's explanation for this decision. Not scored. Used for debugging/analysis"
    )

    @field_validator("irrigation_amount_mm")
    @classmethod
    def validate_irrigation(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return round(max(0.0, min(100.0, v)), 1)

    @field_validator("reasoning")
    @classmethod
    def clean_reasoning(cls, v: str) -> str:
        return v.strip()[:500]


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: AgroObservation
    reward: float = Field(..., description="Reward for this step (-1.0 to +1.0)")
    done: bool = Field(..., description="Whether the episode has ended")
    info: dict = Field(default_factory=dict, description="Diagnostic info (not available to agent during scoring)")


# ---------------------------------------------------------------------------
# Reset Request/Response
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: TaskName = Field("irrigation_scheduling", description="Task to initialize")
    crop: CropKey = Field("rice_kharif", description="Crop to simulate")
    soil: SoilKey = Field("loamy_soil", description="Soil type")
    region: RegionKey = Field("maharashtra_pune", description="Agro-climatic region")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    observation: AgroObservation
    task_description: str
    success_criteria: str
    max_steps: int
    episode_id: str


# ---------------------------------------------------------------------------
# State Response
# ---------------------------------------------------------------------------

class StateResponse(BaseModel):
    episode_id: str
    task: TaskName
    day: int
    done: bool
    total_reward: float
    steps_taken: int
    crop: str
    region: str
    soil: str


# ---------------------------------------------------------------------------
# Error Model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

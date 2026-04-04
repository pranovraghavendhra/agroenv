# 🌾 AgroEnv: Precision Agriculture Advisor

> An OpenEnv-compliant reinforcement learning environment simulating real Indian smallholder farming decisions. Agents manage irrigation, pest control, and harvest timing using genuine agronomic science — not toy heuristics.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

---

## Why AgroEnv?

**The problem:** 500 million smallholder farmers globally make daily decisions — when to irrigate, whether to spray pesticides, when to harvest — with incomplete information and limited agronomic knowledge. Wrong decisions cost 20–40% of potential yield.

**What AgroEnv does:** Provides a scientifically rigorous simulation of these decisions so AI agents can be trained and evaluated against real agronomic ground truth. Every reward function is grounded in peer-reviewed science:

| Science Layer | Standard Used |
|---|---|
| Crop water demand | FAO-56 Penman-Monteith ET₀ (Allen et al., 1998) |
| Growth development | GDD Growing Degree Days + LAI/NDVI via Beer-Lambert |
| Pest management | ICAR Economic Injury Level / Economic Threshold |
| Market prices | AGMARKNET historical data calibration (2023-24) |
| Soil water balance | FAO-56 single-layer root zone depletion |
| Weather simulation | IMD climatological normals (Maharashtra, Punjab, AP) |

**Honest baseline scores:** GPT-4o-mini scores ~0.68 on irrigation, ~0.55 on pest management, ~0.38 on the full season — below pass threshold on the hardest task. There is genuine room for better agents.

---

## Architecture

```
agroenv/
├── server/
│   ├── main.py                     # FastAPI HTTP server (OpenEnv API)
│   ├── env.py                      # Core episode orchestrator
│   ├── models.py                   # Pydantic typed API contract
│   ├── config.py                   # Constants and defaults
│   ├── simulation/
│   │   ├── weather_engine.py       # FAO-56 ET₀ + IMD-calibrated weather
│   │   ├── soil_model.py           # Root zone water balance
│   │   ├── crop_model.py           # GDD, LAI, NDVI, yield estimation
│   │   ├── pest_model.py           # IPM population dynamics
│   │   └── market_engine.py        # AGMARKNET price simulation
│   ├── tasks/
│   │   ├── task_irrigation.py      # Task 1: Easy (14 days)
│   │   ├── task_pest_management.py # Task 2: Medium (30 days)
│   │   └── task_season_optimizer.py# Task 3: Hard (110 days)
│   ├── graders/
│   │   ├── grader_irrigation.py    # Deterministic grader T1
│   │   ├── grader_pest.py          # Deterministic grader T2
│   │   └── grader_season.py        # Deterministic grader T3
│   ├── data/
│   │   ├── crops.json              # 4 Indian crop varieties (real parameters)
│   │   ├── soils.json              # 4 Indian soil profiles (real hydraulics)
│   │   ├── weather_seeds.json      # IMD normals for 3 agro-climatic zones
│   │   └── market_prices.json      # AGMARKNET-calibrated price data
│   └── tests/
│       └── test_all.py             # Full test suite
├── inference.py                    # Baseline LLM agent (OpenAI client)
├── openenv.yaml                    # OpenEnv spec metadata
├── Dockerfile                      # Production container
└── README.md
```

---

## The Three Tasks

### Task 1: Irrigation Scheduling `[EASY]`

**Real-world scenario:** A rice farmer in Maharashtra must decide each morning whether to irrigate, how much water to apply, and which delivery method to use — while monitoring a 7-day weather forecast to avoid wasting water before rain.

**Duration:** 14 days (critical tillering stage)  
**Crop:** IR-64 Rice — Maharashtra Kharif season  
**Pass threshold:** Score ≥ 0.65

**What the agent sees:**
- Soil moisture (%) and depletion from field capacity (mm)
- FAO-56 reference ET₀ and crop water demand (ET₀ × Kc)
- 7-day weather forecast with rain probability and uncertainty
- Soil water stress coefficient (Ks) — 1.0 = no stress, 0.0 = full stress
- Water budget remaining (mm) and cost per irrigation event

**What the agent decides:**
- `irrigate`: true/false
- `irrigation_amount_mm`: 0–100mm
- `irrigation_method`: drip / sprinkler / flood / furrow

**Grader scoring (deterministic):**
- Water management efficiency vs FAO-56 oracle (40%)
- Stress avoidance (days Ks < critical threshold) (35%)
- Waste minimization (drainage + runoff) (15%)
- Method efficiency (drip > sprinkler > flood) (10%)

**Why frontier LLMs find this non-trivial:** The agent must learn that "depletion > RAW threshold" means irrigate, and must correctly identify rain forecasts to avoid unnecessary irrigation. Over-irrigation causes drainage waste and score penalties.

---

### Task 2: Pest & Disease Management `[MEDIUM]`

**Real-world scenario:** A cotton farmer in Andhra Pradesh manages two simultaneous pest species — whitefly (*Bemisia tabaci*) and bollworm (*Helicoverpa armigera*) — using ICAR Integrated Pest Management guidelines. Spraying too early kills natural enemies; spraying too late means yield loss.

**Duration:** 30 days (flowering to boll development)  
**Crop:** Bt-Cotton RCH-2 — Andhra Pradesh  
**Pass threshold:** Score ≥ 0.60

**What the agent sees:**
- Per-pest population vs ICAR Economic Threshold and Economic Injury Level
- Pesticide resistance index (0 = naive, 1 = fully resistant)
- Natural enemy population (biocontrol indicator)
- Days since last spray (rotation timing)
- Cumulative yield damage per pest (%)

**What the agent decides:**
- `spray_decisions`: list of `{pest_name, pesticide}` — one per pest species
- Pesticide choice from 16 ICAR-registered options with different efficacy/resistance profiles

**Grader scoring (deterministic):**
- Pest damage control (avg % damage) (35%)
- IPM compliance (correct vs unnecessary spray rate) (30%)
- Resistance management (max resistance index) (20%)
- Biocontrol preservation (natural enemy population) (15%)

**Why this is harder:** The agent must track resistance across multiple applications, understand that different pesticides build resistance at different rates, and recognize when natural enemies are already providing adequate suppression.

---

### Task 3: Full Season Optimizer `[HARD]`

**Real-world scenario:** A tomato farmer in Andhra Pradesh on red laterite soil manages an entire Rabi season (110 days). Market prices fluctuate from ₹500 to ₹4,000/quintal in the same season. Water stress during fruit set causes irreversible blossom drop. Early Blight can destroy 30% yield in a week of high humidity.

**Duration:** 110 days (full season)  
**Crop:** Pusa Ruby Tomato — Andhra Pradesh  
**Pass threshold:** Score ≥ 0.55

**What the agent sees:** Everything from Tasks 1 and 2, plus:
- AGMARKNET-calibrated market prices (INR/quintal) and 15-day price outlook
- Price trend (rising/stable/falling)
- Market sentiment (glut risk %)
- Harvest window status (in/out of optimal GDD range)
- Net revenue projection given current yield trajectory

**What the agent decides:**
- Irrigation (as Task 1)
- Pest spray decisions (as Task 2)
- `harvest_now`: whether to harvest today (irreversible)

**Grader scoring (deterministic):**
- Net revenue per hectare vs break-even (₹80,000/ha) (45%)
- Yield achievement (% of 25 tonnes/ha max) (25%)
- Harvest timing quality (within GDD window, price trend) (15%)
- Sustainability (water efficiency, resistance management) (15%)

**Why this is genuinely hard:** Frontier models score ~0.38 baseline — below break-even. The harvest timing decision is irreversible, interacts with market volatility, and must be balanced against continuing to build yield mass. Local optima traps: aggressive irrigation → waterlogging → Early Blight → total crop loss.

---

## Observation Space

```json
{
  "task": "season_optimizer",
  "day": 45,
  "weather_today": {
    "tmax_c": 31.4, "tmin_c": 21.2, "tmean_c": 26.3,
    "humidity_pct": 72.1, "rainfall_mm": 0.0,
    "solar_radiation_mj_m2": 18.4,
    "et0_mm": 5.12,
    "wind_speed_ms": 2.8
  },
  "weather_forecast": [
    {
      "day_ahead": 1, "tmax_c": 32.1, "tmin_c": 22.0,
      "rain_prob_pct": 15, "expected_rain_mm": 0.0,
      "et0_forecast_mm": 5.28, "humidity_pct": 68.0,
      "forecast_confidence": 0.92
    }
    // ... 6 more days
  ],
  "soil": {
    "moisture_pct": 18.4,
    "field_capacity_pct": 22.0,
    "wilting_point_pct": 10.0,
    "depletion_mm": 22.1,
    "ks": 0.61,
    "drainage_mm_today": 0.0,
    "cumulative_stress_days": 3,
    "waterlog_days": 0,
    "raw_mm": 18.0
  },
  "crop": {
    "crop_name": "Tomato (Rabi)",
    "growth_stage": "fruit_set",
    "day_of_season": 45,
    "total_season_days": 110,
    "gdd_accumulated": 892.4,
    "gdd_progress_pct": 63.7,
    "ndvi": 0.821,
    "lai": 3.82,
    "canopy_cover_pct": 63.7,
    "kc": 1.20,
    "estimated_yield_pct": 88.2,
    "in_harvest_window": false,
    "days_to_harvest_window": 28,
    "harvest_window_closing_days": 52
  },
  "pests": [
    {
      "pest_name": "fruit_borer",
      "population": 0.4,
      "economic_threshold": 1.0,
      "economic_injury_level": 2.0,
      "at_threshold": false,
      "above_eil": false,
      "resistance_index": 0.031,
      "days_since_spray": 8,
      "natural_enemy_population": 0.72,
      "damage_accumulated_pct": 0.8
    }
  ],
  "market": {
    "current_price_inr_per_quintal": 2140.0,
    "msp_inr_per_quintal": 1500.0,
    "price_vs_msp_pct": 42.7,
    "market_trend": "rising",
    "days_to_peak_price": 12,
    "glut_risk_pct": 25.0,
    "price_3d_ahead": 2280.0,
    "price_7d_ahead": 2450.0,
    "price_15d_ahead": 2600.0
  },
  "resources": {
    "budget_remaining_inr": 9200.0,
    "water_available_mm": 312.0,
    "irrigation_events_used": 8,
    "spray_events_used": 2,
    "cumulative_irrigation_mm": 188.0,
    "cost_irrigation_today_inr": 0.0,
    "cost_spray_today_inr": 0.0
  },
  "last_action_result": "✓ Correct: skipped irrigation (soil adequate) | ✓ No spray on fruit_borer (below threshold)",
  "episode_reward_so_far": 3.841,
  "info_message": "⚠️ CRITICAL: Water stress (Ks=0.61) — irrigation recommended"
}
```

## Action Space

```json
{
  "irrigate": true,
  "irrigation_amount_mm": 28.0,
  "irrigation_method": "drip",
  "spray_decisions": [
    {"pest_name": "fruit_borer", "pesticide": "none"},
    {"pest_name": "early_blight", "pesticide": "mancozeb"}
  ],
  "harvest_now": false,
  "reasoning": "Ks=0.61 below 0.75 threshold during fruit set. No rain forecast. Depletion 22mm > RAW 18mm. Drip chosen for efficiency."
}
```

**Action field constraints:**
| Field | Type | Range/Options |
|---|---|---|
| `irrigate` | bool | — |
| `irrigation_amount_mm` | float | 0.0 – 100.0 mm |
| `irrigation_method` | enum | drip / sprinkler / flood / furrow / none |
| `spray_decisions` | list | One entry per active pest; pesticide from 16 ICAR options |
| `harvest_now` | bool | Positive reward only when `in_harvest_window=true` |
| `reasoning` | string | Max 500 chars. **Not scored.** |

---

## Reward Design

**Dense per-step rewards** in range `[-1.0, +1.0]`:

| Event | Reward |
|---|---|
| Correct irrigation (depleted soil, no rain forecast) | +0.30 |
| Irrigation amount within 15% of FAO-56 oracle | +0.10 |
| Correct skip (adequate soil, rain incoming) | +0.25 |
| Drip/sprinkler method chosen | +0.05 |
| Over-irrigation (drainage waste) | -0.08 to -0.20 |
| Missed irrigation above RAW threshold | -0.25 |
| Waterlogging event | -0.10 to -0.12 |
| Correct spray at ICAR threshold | +0.25 |
| Low-resistance-risk pesticide chosen | +0.05 |
| Unnecessary spray (below threshold) | -0.20 |
| Natural enemies destroyed by unnecessary spray | -0.08 |
| Harvest within GDD window, rising price | +0.30 |
| Premature harvest (before window) | -0.40 |
| Water stress during fruit set (tomato) | -0.12 |

**Final episode score (0.0–1.0):** Computed by deterministic grader from episode record. Independent of step rewards.

---

## API Reference

### `POST /reset`

Start a new episode. Session is maintained via cookie.

```json
// Request body (all optional — defaults applied per task)
{
  "task": "irrigation_scheduling",  // irrigation_scheduling | pest_management | season_optimizer
  "crop": "rice_kharif",
  "soil": "loamy_soil",
  "region": "maharashtra_pune",
  "seed": 42
}
```

```json
// Response
{
  "observation": { /* full AgroObservation */ },
  "task_description": "Manage rice irrigation for 14 days...",
  "success_criteria": "Maintain soil moisture above wilting...",
  "max_steps": 14,
  "episode_id": "a3f8c1d2"
}
```

### `POST /step`

Take one action, advance simulation by one day.

```json
// Request body: AgroAction (see Action Space above)
// Response: StepResult
{
  "observation": { /* updated AgroObservation */ },
  "reward": 0.3500,
  "done": false,
  "info": { "step": 5, "total_reward": 1.42, "budget_remaining": 8200 }
}
```

### `GET /state`

Lightweight episode state — no full observation.

```json
{
  "episode_id": "a3f8c1d2",
  "task": "season_optimizer",
  "day": 45,
  "done": false,
  "total_reward": 3.841,
  "steps_taken": 45,
  "crop": "tomato_rabi",
  "region": "andhra_guntur",
  "soil": "red_laterite_soil"
}
```

### `POST /grade`

Run deterministic grader on current episode.

```json
{
  "score": 0.7234,
  "passed": true,
  "breakdown": {
    "water_efficiency_score": 0.85,
    "stress_avoidance_score": 0.92,
    "waste_minimization_score": 0.70,
    "method_efficiency_score": 1.00,
    "total_irrigation_mm": 94.0,
    "oracle_irrigation_mm": 88.0
  },
  "episode_id": "a3f8c1d2",
  "steps_completed": 14,
  "done": true
}
```

---

## Setup & Running

### Quick Start (Docker)

```bash
git clone https://huggingface.co/spaces/<your-username>/agroenv
cd agroenv
docker build -t agroenv .
docker run -p 7860:7860 agroenv
```

Visit `http://localhost:7860/docs` for interactive API documentation.

### Local Development

```bash
cd agroenv
pip install -r server/requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload
```

### Running the Inference Script

```bash
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export AGROENV_URL="http://localhost:7860"

# Start the server first, then:
python inference.py
```

### Running Tests

```bash
cd agroenv
python -m pytest server/tests/ -v
```

---

## Baseline Scores (Seed=42)

| Task | Difficulty | Baseline Score | Pass Threshold | Status |
|---|---|---|---|---|
| Irrigation Scheduling | Easy | ~0.68 | 0.65 | ✅ PASS |
| Pest Management | Medium | ~0.55 | 0.60 | ❌ FAIL |
| Season Optimizer | Hard | ~0.38 | 0.55 | ❌ FAIL |
| **Average** | | **~0.54** | — | — |

Baseline uses `gpt-4o-mini` with `temperature=0.2`. A well-tuned agent should score 0.80+ on all tasks.

---

## Data Sources & Honesty

**What is real:**
- FAO-56 Penman-Monteith formula — exact equations from Allen et al. (1998)
- ICAR Economic Thresholds — from published ICAR IPM guidelines
- MSP prices — CACP 2023-24 recommendations
- IMD weather normals — monthly climatological averages for real stations
- Soil hydraulic properties — from NBSS&LUP soil surveys

**What is simulated (and why):**
- Daily weather: synthetic stochastic generation calibrated to IMD normals. Real IMD API requires registration and has rate limits incompatible with a competition environment.
- Market prices: stochastic walk calibrated to AGMARKNET historical volatility. Real mandi prices change by the hour — a static API would give stale data.
- NDVI: derived from LAI via Beer-Lambert law — scientifically valid approximation. Real satellite NDVI requires Sentinel Hub credentials and adds 300MB+ dependencies.

---

## Scientific References

1. Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *FAO Irrigation and Drainage Paper No. 56: Crop Evapotranspiration.* FAO, Rome.
2. Stern, V.M., Smith, R.F., van den Bosch, R., Hagen, K.S. (1959). The integrated control concept. *Hilgardia* 29(2): 81–101.
3. Baret, F. & Guyot, G. (1991). Potentials and limits of vegetation indices for LAI and APAR assessment. *Remote Sensing of Environment* 35(2-3): 161–173.
4. ICAR (2022). *Package of Practices for Crop Production.* Indian Council of Agricultural Research, New Delhi.
5. CACP (2023). *Price Policy for Kharif and Rabi Crops 2023-24.* Commission for Agricultural Costs and Prices, GoI.
6. AGMARKNET (2024). *Agricultural Marketing Information Network.* NIC / Ministry of Agriculture, GoI.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

Built for the OpenEnv competition. Crop parameters, soil data, and weather normals compiled from public Indian government datasets (ICAR, IMD, AGMARKNET, CACP). All simulation models are deterministic, reproducible, and grounded in peer-reviewed agronomic science.

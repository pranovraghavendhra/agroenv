"""
AgroEnv Configuration
======================
Central constants used across all modules.
Modify here to change global environment behaviour.
"""

# --- Episode Limits ---
MAX_STEPS = {
    "irrigation_scheduling": 14,
    "pest_management": 30,
    "season_optimizer": 110,
}

# --- Reward Bounds ---
REWARD_MIN = -1.0
REWARD_MAX = 1.0

# --- Budget (INR per hectare per season) ---
SEASON_BUDGET_INR = {
    "irrigation_scheduling": 10_000.0,
    "pest_management": 5_000.0,
    "season_optimizer": 15_000.0,
}

# --- Water Allocation (mm per season) ---
WATER_ALLOCATION_MM = {
    "irrigation_scheduling": 300.0,
    "pest_management": 200.0,
    "season_optimizer": 500.0,
}

# --- Irrigation Costs (INR per mm per ha) ---
IRRIGATION_COST = {
    "drip":      8.0,
    "sprinkler": 6.0,
    "furrow":    4.0,
    "flood":     3.5,
    "none":      0.0,
}

# --- Irrigation Efficiency (fraction of water reaching roots) ---
IRRIGATION_EFFICIENCY = {
    "drip":      0.92,
    "sprinkler": 0.80,
    "furrow":    0.65,
    "flood":     0.55,
    "none":      1.00,
}

# --- Spray Costs (INR per event per ha, inclusive of labour + chemical) ---
SPRAY_COST_INR = 850.0

# --- Harvest Cost (INR per ha, labour) ---
HARVEST_COST_INR = 8_000.0

# --- Score Thresholds for Pass/Fail ---
PASS_THRESHOLD = {
    "irrigation_scheduling": 0.65,
    "pest_management":       0.60,
    "season_optimizer":      0.55,
}

# --- Default Task Configurations ---
TASK_DEFAULTS = {
    "irrigation_scheduling": {
        "crop": "rice_kharif",
        "soil": "loamy_soil",
        "region": "maharashtra_pune",
    },
    "pest_management": {
        "crop": "cotton_kharif",
        "soil": "black_cotton_soil",
        "region": "andhra_guntur",
    },
    "season_optimizer": {
        "crop": "tomato_rabi",
        "soil": "red_laterite_soil",
        "region": "andhra_guntur",
    },
}

# --- API ---
API_VERSION = "1.0.0"
ENV_NAME = "agroenv"
ENV_DISPLAY_NAME = "AgroEnv: Precision Agriculture Advisor"

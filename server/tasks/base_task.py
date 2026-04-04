"""
Base Task and Grader Abstract Classes
======================================
All tasks and graders must inherit from these bases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class TaskConfig:
    task_name: str
    crop_key: str
    soil_key: str
    region_key: str
    max_steps: int
    description: str
    success_criteria: str
    difficulty: str  # "easy" | "medium" | "hard"


@dataclass
class EpisodeRecord:
    """Full episode history for grader evaluation."""
    task_name: str
    crop_key: str
    days: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    step_rewards: list = field(default_factory=list)
    step_infos: list = field(default_factory=list)
    final_yield_ton_per_ha: float = 0.0
    final_revenue_inr_per_ha: float = 0.0
    total_irrigation_mm: float = 0.0
    total_spray_events: int = 0
    total_cost_inr: float = 0.0
    cumulative_stress_days: int = 0
    harvest_day: Optional[int] = None
    harvest_gdd: Optional[float] = None
    unnecessary_sprays: int = 0
    correct_sprays: int = 0


class BaseTask(ABC):
    """Abstract base for all AgroEnv tasks."""

    @property
    @abstractmethod
    def config(self) -> TaskConfig:
        pass

    @abstractmethod
    def compute_step_reward(
        self,
        action: Any,
        obs_before: Any,
        obs_after: Any,
        step_info: dict,
    ) -> tuple[float, str]:
        pass

    @abstractmethod
    def is_done(self, obs: Any, step: int, episode: EpisodeRecord) -> bool:
        pass


class BaseGrader(ABC):
    """Abstract base for all AgroEnv task graders."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass

    @abstractmethod
    def grade(self, episode: EpisodeRecord) -> tuple[float, dict]:
        pass

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

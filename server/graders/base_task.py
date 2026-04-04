"""
Base Grader Abstract Class (graders package copy)
===================================================
Shared base for all AgroEnv task graders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


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


class BaseGrader(ABC):
    """Abstract base for all AgroEnv task graders."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass

    @abstractmethod
    def grade(self, episode: EpisodeRecord) -> tuple[float, dict]:
        """
        Grade a completed episode.
        Returns:
            (score, breakdown_dict)
            score: float in [0.0, 1.0]
            breakdown_dict: detailed scoring components
        """
        pass

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

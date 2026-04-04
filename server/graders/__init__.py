from .base_task import BaseGrader, EpisodeRecord
from .grader_irrigation import IrrigationGrader
from .grader_pest import PestGrader
from .grader_season import SeasonGrader

GRADER_REGISTRY = {
    "irrigation_scheduling": IrrigationGrader,
    "pest_management": PestGrader,
    "season_optimizer": SeasonGrader,
}

__all__ = [
    "BaseGrader", "EpisodeRecord",
    "IrrigationGrader", "PestGrader", "SeasonGrader",
    "GRADER_REGISTRY",
]

from .base_task import BaseTask, BaseGrader, TaskConfig, EpisodeRecord
from .task_irrigation import IrrigationTask
from .task_pest_management import PestManagementTask
from .task_season_optimizer import SeasonOptimizerTask

TASK_REGISTRY = {
    "irrigation_scheduling": IrrigationTask,
    "pest_management": PestManagementTask,
    "season_optimizer": SeasonOptimizerTask,
}

__all__ = [
    "BaseTask", "BaseGrader", "TaskConfig", "EpisodeRecord",
    "IrrigationTask", "PestManagementTask", "SeasonOptimizerTask",
    "TASK_REGISTRY",
]

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from OSWorld.desktop_env.controllers.python import PythonController
from OSWorld.desktop_env.controllers.setup import SetupController
from perturbation_engine.pipeline.perturbation_controllers import VLMController


class PerturbationType(Enum):
    """Types of perturbations supported"""

    INSTRUCTION = "instruction"
    UI_VISUAL = "ui_visual"
    VISUAL_DISTRACTOR = "visual_distractor"
    ENVIRONMENT_DISTRACTOR = "environment_distractor"


class PerturbationPhase(Enum):
    """When perturbations are applied"""

    SETUP = "setup"
    RUNTIME = "runtime"


PerturbationControllers: Dict[str, Any] = {
    "python": PythonController,
    "setup": SetupController,
    "vlm": VLMController,
}


@dataclass
class PerturbationSpec:
    """Specification for a perturbation"""

    perturbation_type: PerturbationType
    phase: PerturbationPhase
    perturbation_controller: str
    parameters: Dict[str, Any]
    trigger_conditions: Dict[str, Any]
    validation_config: Optional[Dict[str, Any]] = None


@dataclass
class ScenarioSpec:
    """Complete scenario specification"""

    task_id: str
    scenario_id: str
    trajectory_folder_dir: str
    base_task_config: Dict[str, Any]
    perturbations: List[PerturbationSpec]
    metadata: Dict[str, Any]
    result_dir: str


@dataclass
class GenerationResult:
    """Result of trajectory generation"""

    task_id: str
    success: bool
    result_score: float
    perturbation_log: List[Dict[str, Any]]
    generation_time: float
    metadata: Dict[str, Any]

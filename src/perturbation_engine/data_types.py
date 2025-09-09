from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class PerturbationSpec:
    """Specification for a perturbation"""

    perturbation_type: PerturbationType
    phase: PerturbationPhase
    perturbation_controller: str
    parameters: Dict[str, Any]

    # Use string names for multiprocessing compatibility
    trigger_function_name: str
    trigger_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_function_name: Optional[str] = None
    validation_parameters: Dict[str, Any] = field(default_factory=dict)

    name: Optional[str] = None
    description: Optional[str] = None
    priority: int = 0


@dataclass
class ExecutionConfig:
    """Configuration for execution environment"""

    # VM/Provider settings
    path_to_vm: Optional[str] = None
    provider_name: str = "docker"
    region: str = "us-east-1"
    snapshot_name: Optional[str] = None

    # Environment settings
    headless: bool = True
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    screen_size: tuple = (1920, 1080)
    os_type: str = "Ubuntu"
    client_password: str = ""

    # Execution settings
    max_steps: int = 15
    sleep_after_execution: float = 0.0

    # Additional OSWorld settings
    cache_dir: str = "cache"
    require_a11y_tree: bool = True
    require_terminal: bool = False
    enable_proxy: bool = False

    # Perturbation connection
    chromium_port: int = 9222


@dataclass
class ScenarioSpec:
    """Scenario specification"""

    # Task identification
    task_id: str
    scenario_id: str

    # OSWorld task configuration (matches example.json format)
    task_config: Dict[str, Any]  # Contains: id, snapshot, instruction, source, config, evaluator, etc.

    # Trajectory information
    trajectory_file_path: str

    # Perturbation specifications
    perturbations: List[PerturbationSpec]

    # Result directory
    result_dir: str

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of trajectory generation"""

    task_id: str
    success: bool
    result_score: float
    perturbation_log: List[Dict[str, Any]]
    generation_time: float
    metadata: Dict[str, Any]

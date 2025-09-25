from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from perturbation_engine.curriculum.curriculum_config import CurriculumConfig


# Constants for clean code
class Constants:
    """Constants for the perturbation engine."""

    # Scenario types
    SCENARIO_TYPES = ["invariance", "distractor", "negative"]

    # Default scenario counts
    DEFAULT_INVARIANCE_COUNT = 15
    DEFAULT_DISTRACTOR_COUNT = 7
    DEFAULT_NEGATIVE_COUNT = 3
    DEFAULT_DIFFICULTY_LEVELS = 4

    # Environment setup
    ENVIRONMENT_READY_WAIT_TIME = 2.0  # seconds
    DEFAULT_MAX_STEPS = 15

    # Controller commands
    UI_REORDERING_CMD = "ui_reordering"
    THEME_CHANGE_CMD = "theme_change"
    ADD_DISTRACTORS_CMD = "add_distractors"
    MODIFY_RESOLUTION_CMD = "modify_resolution"
    INJECT_POPUPS_CMD = "inject_popups"


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
class SeedTrajectory:
    """Seed trajectory"""

    task_type: str
    task_instruction: str
    config: Dict[str, Any]
    gt_actions_file_path: str
    gt_actions: Optional[List[Dict[str, Any]]]


@dataclass
class GenerationConfig:
    """Generation configuration"""

    # Scenario numbers
    num_invariance_scenarios: int = Constants.DEFAULT_INVARIANCE_COUNT
    num_distractor_scenarios: int = Constants.DEFAULT_DISTRACTOR_COUNT
    num_negative_scenarios: int = Constants.DEFAULT_NEGATIVE_COUNT
    # Difficulty levels
    num_difficulty_levels: int = Constants.DEFAULT_DIFFICULTY_LEVELS


@dataclass
class DifficultyLevel:
    """Difficulty level configuration"""

    level: int
    intensity: float  # 0.0 to 1.0
    perturbation_count: int
    complexity_multiplier: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    """Scenario specification"""

    # Task identification
    task_id: str
    scenario_id: str
    task_type: str
    scenario_type: str
    difficulty_level: int
    seed_trajectory: SeedTrajectory

    # Trajectory information
    trajectory_file_path: str

    # Perturbation scenario
    perturbation_scenario_class: str
    intensity: float
    perturbation_count: int
    parameters: Dict[str, Any]  # Renamed from perturbation_parameters for clarity

    # Result directory
    result_dir: str

    # Metadata
    seed_index: int
    scenario_count: int

    # Optional curriculum configuration for runtime curriculum generation
    curriculum_config: Optional["CurriculumConfig"] = None

    def __post_init__(self):
        """Ensure all data is serializable."""
        if not isinstance(self.parameters, dict):
            self.parameters = dict(self.parameters) if self.parameters else {}

        if not isinstance(self.seed_trajectory, SeedTrajectory):
            self.seed_trajectory = SeedTrajectory(
                task_type=self.seed_trajectory.task_type,
                task_instruction=self.seed_trajectory.task_instruction,
                config=self.seed_trajectory.config,
                gt_actions_file_path=self.seed_trajectory.gt_actions_file_path,
                gt_actions=self.seed_trajectory.gt_actions,
            )

    def to_difficulty_level(self) -> "DifficultyLevel":
        """Convert to DifficultyLevel without recreation."""
        return DifficultyLevel(
            level=self.difficulty_level,
            intensity=self.intensity,
            perturbation_count=self.perturbation_count,
            parameters=self.parameters,
        )


@dataclass
class EnvironmentState:
    """Environment state extracted from first observation"""

    dom_tree: str
    a11y_tree: str
    app_type: str
    current_url: str
    window_state: Dict[str, Any]
    task_instruction: str


@dataclass
class GenerationResult:
    """Result of trajectory generation"""

    task_id: str
    success: bool
    result_score: float
    perturbation_log: List[Dict[str, Any]]
    generation_time: float
    metadata: Dict[str, Any]

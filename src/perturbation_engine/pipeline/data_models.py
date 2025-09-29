"""
Data Models: Immutable data structures for the perturbation pipeline
Following clean code principles with focused responsibilities
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PerturbationType(Enum):
    """Types of perturbations supported"""

    THEME = "theme"
    LAYOUT = "layout"
    CONTENT_VARIATION = "content_variation"
    UI_INJECTION = "ui_injection"
    NOTIFICATION = "notification"
    BACKGROUND_PROCESS = "background_process"
    WINDOW_MANAGEMENT = "window_management"
    FILE_OPERATIONS = "file_operations"


@dataclass(frozen=True)
class ExecutionConfig:
    """VM settings, timeouts, cleanup - immutable configuration"""

    # VM/Provider settings
    path_to_vm: Optional[str] = None
    provider_name: str = "vmware"
    region: str = "us-east-1"
    snapshot_name: Optional[str] = None

    # Environment settings
    headless: bool = True
    action_space: str = "pyautogui"
    screen_size: tuple = (1920, 1080)
    os_type: str = "Ubuntu"
    client_password: str = ""

    # Execution settings
    max_steps: int = 15
    sleep_after_execution: float = 0.0

    # Additional settings
    cache_dir: str = "cache"
    require_a11y_tree: bool = True
    require_terminal: bool = False
    enable_proxy: bool = False
    chromium_port: int = 9222


@dataclass(frozen=True)
class CurriculumConfig:
    """Scenario generation settings - immutable configuration"""

    scenario_count: int = 10
    num_parallel_vms: int = 1
    result_base_dir: str = "./curriculum_results"

    # Curriculum difficulty distribution
    beginner_scenarios: int = 3
    intermediate_scenarios: int = 4
    advanced_scenarios: int = 2


@dataclass(frozen=True)
class ScenarioSpec:
    """What/when/how to perturb - immutable specification"""

    scenario_id: str
    target_app: str  # e.g., "libreoffice"
    perturbation_trigger: str  # Visual condition or action-based trigger
    available_perturbation_actions: str  # Code snippets using functions
    learning_objectives: str  # Invariant themes
    target_components: List[str]  # e.g., ["buttons", "cells"]
    perturbation_types: List[PerturbationType]  # e.g., [THEME, LAYOUT]


@dataclass(frozen=True)
class SeedTrajectory:
    """Input trajectory + metadata - immutable"""

    task_id: str
    task_type: str
    task_instruction: str
    config: Dict[str, Any]
    gt_actions_file_path: str
    gt_actions: Optional[List[Dict[str, Any]]] = None


@dataclass
class GeneratedTrajectory:
    """Output trajectory + quality score - immutable result"""

    trajectory_id: str
    seed_trajectory_id: str
    scenario_spec_id: str
    success: bool
    quality_score: float
    generation_time: float
    trajectory_file_path: str
    perturbation_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime state for decisions - immutable context"""

    step_idx: int
    current_action: str
    action_history: List[str] = field(default_factory=list)
    cot_context: str = ""
    app_states: List[Dict[str, Any]] = field(default_factory=list)
    task_instruction: str = ""
    task_type: str = ""
    scenario_spec: Optional[ScenarioSpec] = None

"""Core types and interfaces for the perturbation engine."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PerturbationScope(Enum):
    """Scope of perturbation application."""

    TASK_INSTRUCTION = "task_instruction"
    UI_VISUAL = "ui_visual"
    VISUAL_DISTRACTOR = "visual_distractor"
    ENVIRONMENT_STATE = "environment_state"


@dataclass
class Command:
    """Executable command for perturbation injection."""

    command_type: str  # 'pyautogui', 'playwright', 'http', 'bash'
    target: str  # 'vm', 'container:service_name', 'browser'
    action: str  # Specific action/endpoint
    parameters: Dict[str, Any]
    retry_count: int = 3
    timeout: float = 30.0
    priority: int = 1  # Lower number = higher priority


@dataclass
class ExecutionResult:
    """Result of executing a command."""

    command: Command
    success: bool
    error_message: Optional[str] = None


@dataclass
class ScenarioParameters:
    """Complete scenario specification for perturbation."""

    task_config: Dict[str, Any]
    ui_theme_params: Dict[str, Any]
    distractor_params: Dict[str, Any]
    environment_state: Dict[str, Any]
    execution_commands: List[Command] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerturbationConfig:
    """Configuration for applying perturbations to a target element."""

    target_selector: str
    parameters: Any
    target_type: str = "web"  # "web", "desktop", "instruction", "distractor"


@dataclass
class PerturbationResult:
    """Result of applying perturbations to an observation."""

    original_data: Path | Dict[str, Any] | bytes
    perturbed_data: Optional[Path | Dict[str, Any] | bytes] = None
    applied_perturbations: Optional[List[PerturbationConfig]] = None
    success: bool = True
    target_type: str = "web"

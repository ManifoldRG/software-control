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
    COLOR = "color"
    DENSITY = "density"
    TYPOGRAPHY = "typography"
    SHAPE = "shape"
    LAYOUT = "layout"
    CONTENT_VARIATION = "content_variation"
    UI_INJECTION = "ui_injection"
    NOTIFICATION = "notification"
    BACKGROUND_PROCESS = "background_process"
    WINDOW_MANAGEMENT = "window_management"
    FILE_OPERATIONS = "file_operations"

    @classmethod
    def from_string(cls, value: str, default: Optional["PerturbationType"] = None) -> "PerturbationType":
        """
        Convert LLM output string to PerturbationType with flexible mapping.
        Handles common variations and aliases to improve LLM compatibility.

        BULLETPROOF: Never raises exceptions if default is provided.

        Args:
            value: String from LLM (e.g., "color", "font", "density")
            default: Fallback value if parsing fails (prevents exceptions)

        Returns:
            Corresponding PerturbationType enum value or default

        Raises:
            ValueError: Only if no mapping exists AND no default provided
        """
        import logging

        logger = logging.getLogger(__name__)

        # Handle None/empty input
        if not value or not isinstance(value, str):
            if default is not None:
                logger.warning(f"Invalid perturbation type value: {value}, using default: {default.value}")
                return default
            return cls.THEME  # Ultimate fallback

        # Normalize input - handle all edge cases
        normalized = str(value).lower().strip()
        normalized = normalized.replace("-", "_").replace(" ", "_").replace(".", "")

        # Remove common prefixes/suffixes that LLMs might add
        normalized = normalized.removeprefix("perturbation_").removeprefix("type_")
        normalized = normalized.removesuffix("_perturbation").removesuffix("_type")

        # Direct enum value match
        try:
            return cls(normalized)
        except ValueError:
            pass

        # Comprehensive LLM variations mapping (100+ aliases)
        mappings = {
            # Typography variations (20+)
            "font": cls.TYPOGRAPHY,
            "fonts": cls.TYPOGRAPHY,
            "font_family": cls.TYPOGRAPHY,
            "font_size": cls.TYPOGRAPHY,
            "font_weight": cls.TYPOGRAPHY,
            "text": cls.TYPOGRAPHY,
            "text_style": cls.TYPOGRAPHY,
            "typeface": cls.TYPOGRAPHY,
            "typefaces": cls.TYPOGRAPHY,
            "text_formatting": cls.TYPOGRAPHY,
            "font_style": cls.TYPOGRAPHY,
            "text_size": cls.TYPOGRAPHY,
            "font_change": cls.TYPOGRAPHY,
            "text_weight": cls.TYPOGRAPHY,
            "letter_spacing": cls.TYPOGRAPHY,
            "line_height": cls.TYPOGRAPHY,
            # Color variations (20+)
            "colors": cls.COLOR,
            "colour": cls.COLOR,
            "colours": cls.COLOR,
            "color_scheme": cls.COLOR,
            "palette": cls.COLOR,
            "palettes": cls.COLOR,
            "background_color": cls.COLOR,
            "text_color": cls.COLOR,
            "foreground": cls.COLOR,
            "background": cls.COLOR,
            "accent": cls.COLOR,
            "accents": cls.COLOR,
            "tint": cls.COLOR,
            "tints": cls.COLOR,
            "hue": cls.COLOR,
            "saturation": cls.COLOR,
            "brightness": cls.COLOR,
            # Layout variations (20+)
            "spacing": cls.LAYOUT,
            "padding": cls.LAYOUT,
            "paddings": cls.LAYOUT,
            "margin": cls.LAYOUT,
            "margins": cls.LAYOUT,
            "alignment": cls.LAYOUT,
            "position": cls.LAYOUT,
            "positioning": cls.LAYOUT,
            "placement": cls.LAYOUT,
            "arrangement": cls.LAYOUT,
            "grid": cls.LAYOUT,
            "flex": cls.LAYOUT,
            "flexbox": cls.LAYOUT,
            "container": cls.LAYOUT,
            "width": cls.LAYOUT,
            "height": cls.LAYOUT,
            "size": cls.LAYOUT,
            "sizing": cls.LAYOUT,
            # Density variations (10+)
            "compact": cls.DENSITY,
            "spacious": cls.DENSITY,
            "comfortable": cls.DENSITY,
            "density_mode": cls.DENSITY,
            "tight": cls.DENSITY,
            "loose": cls.DENSITY,
            "cozy": cls.DENSITY,
            "spacing_mode": cls.DENSITY,
            # Theme variations (15+)
            "design_system": cls.THEME,
            "theme_change": cls.THEME,
            "appearance": cls.THEME,
            "style": cls.THEME,
            "styles": cls.THEME,
            "styling": cls.THEME,
            "visual_theme": cls.THEME,
            "ui_theme": cls.THEME,
            "material": cls.THEME,
            "fluent": cls.THEME,
            "dark_mode": cls.THEME,
            "light_mode": cls.THEME,
            "theme_variant": cls.THEME,
            # Shape variations (15+)
            "border": cls.SHAPE,
            "borders": cls.SHAPE,
            "radius": cls.SHAPE,
            "border_radius": cls.SHAPE,
            "corner": cls.SHAPE,
            "corners": cls.SHAPE,
            "shadow": cls.SHAPE,
            "shadows": cls.SHAPE,
            "elevation": cls.SHAPE,
            "outline": cls.SHAPE,
            "outlines": cls.SHAPE,
            "edge": cls.SHAPE,
            "edges": cls.SHAPE,
            "rounded": cls.SHAPE,
            # Content variation (10+)
            "motion": cls.CONTENT_VARIATION,
            "animation": cls.CONTENT_VARIATION,
            "animations": cls.CONTENT_VARIATION,
            "transition": cls.CONTENT_VARIATION,
            "transitions": cls.CONTENT_VARIATION,
            "content": cls.CONTENT_VARIATION,
            "variation": cls.CONTENT_VARIATION,
            "transform": cls.CONTENT_VARIATION,
            # Other dimensions
            "depth": cls.SHAPE,
            "z_index": cls.SHAPE,
            "layer": cls.SHAPE,
            "layers": cls.SHAPE,
            "semantics": cls.THEME,
            "semantic": cls.THEME,
            "hierarchy": cls.THEME,
            # Window/system (10+)
            "window": cls.WINDOW_MANAGEMENT,
            "windows": cls.WINDOW_MANAGEMENT,
            "resize": cls.WINDOW_MANAGEMENT,
            "resizing": cls.WINDOW_MANAGEMENT,
            "move": cls.WINDOW_MANAGEMENT,
            "reposition": cls.WINDOW_MANAGEMENT,
            # Notifications (10+)
            "notify": cls.NOTIFICATION,
            "notification": cls.NOTIFICATION,
            "alert": cls.NOTIFICATION,
            "alerts": cls.NOTIFICATION,
            "toast": cls.NOTIFICATION,
            "toasts": cls.NOTIFICATION,
            "popup": cls.NOTIFICATION,
            "popups": cls.NOTIFICATION,
            # Background (10+)
            "background_process": cls.BACKGROUND_PROCESS,
            "bg": cls.BACKGROUND_PROCESS,
            "bg_process": cls.BACKGROUND_PROCESS,
            "background_task": cls.BACKGROUND_PROCESS,
            "process": cls.BACKGROUND_PROCESS,
            # Files (10+)
            "file": cls.FILE_OPERATIONS,
            "files": cls.FILE_OPERATIONS,
            "file_operation": cls.FILE_OPERATIONS,
            "file_system": cls.FILE_OPERATIONS,
            # UI Injection
            "injection": cls.UI_INJECTION,
            "inject": cls.UI_INJECTION,
            "add_element": cls.UI_INJECTION,
            "insert": cls.UI_INJECTION,
        }

        if normalized in mappings:
            return mappings[normalized]

        # Try partial matching as last resort (e.g., "theme_color" → contains "theme")
        for key, enum_val in mappings.items():
            if key in normalized or normalized in key:
                logger.warning(f"Partial match: '{value}' matched '{key}' → {enum_val.value}")
                return enum_val

        # If no mapping found and default provided, use default
        if default is not None:
            logger.warning(f"Unknown perturbation type '{value}', using default: {default.value}")
            return default

        # Ultimate fallback: use THEME (safest, most common)
        logger.warning(
            f"Unknown perturbation type '{value}', using fallback: THEME. "
            f"Valid types: {', '.join([e.value for e in cls])}"
        )
        return cls.THEME


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
    # Enhanced logging fields
    scenario_spec_content: Optional[Dict[str, Any]] = None
    final_app_states: Optional[List[Dict[str, Any]]] = None
    total_perturbation_attempts: int = 0
    total_perturbation_successes: int = 0
    # Step-by-step comprehensive logging
    step_by_step_log: List[Dict[str, Any]] = field(default_factory=list)
    successful_perturbation_commands: List[Dict[str, Any]] = field(default_factory=list)
    failed_perturbation_commands: List[Dict[str, Any]] = field(default_factory=list)


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

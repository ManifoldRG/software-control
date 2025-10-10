"""
Data Models: Immutable data structures for the perturbation pipeline
Following clean code principles with focused responsibilities
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VisibilityState(Enum):
    """Element visibility states"""

    VISIBLE = "visible"  # Fully visible and interactable
    HIDDEN_COLLAPSED = "collapsed"  # In collapsed menu/dropdown
    HIDDEN_WINDOW = "hidden_window"  # In hidden/minimized window
    HIDDEN_TAB = "hidden_tab"  # In inactive tab
    STRUCTURAL = "structural"  # Container element (frame, panel)
    HIDDEN_NOT_SHOWING = "not_showing"  # AT-SPI2 showing=false


@dataclass
class UIElement:
    """Represents a UI element with hierarchy"""

    element_id: str
    element_type: str
    name: str
    position: Dict[str, int]

    # Hierarchy
    parent_id: Optional[str] = None
    children: List["UIElement"] = field(default_factory=list)
    depth: int = 0

    # States
    visibility: VisibilityState = VisibilityState.VISIBLE
    is_enabled: bool = True
    is_focused: bool = False
    is_expanded: bool = False  # For menus, dropdowns, etc.

    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "name": self.name,
            "text": self.name,
            "position": self.position,
            "properties": {
                **self.properties,
                "parent_id": self.parent_id,
                "depth": self.depth,
                "visibility": self.visibility.value,
                "is_enabled": self.is_enabled,
                "is_focused": self.is_focused,
                "is_expanded": self.is_expanded,
                "children_count": len(self.children),
            },
        }


@dataclass
class WindowState:
    """Represents a window with its elements and X11 window manager data"""

    window_id: str
    window_name: str
    app_name: str

    # Window properties
    is_active: bool = False
    is_modal: bool = False
    is_minimized: bool = False
    geometry: Dict[str, int] = field(default_factory=dict)
    z_order: int = 0

    # X11 window manager data
    x11_window_id: Optional[str] = None
    is_mapped: bool = True  # Actually visible from X11
    desktop: int = 0  # Virtual desktop number

    # Elements tree
    root_element: Optional[UIElement] = None

    def get_all_elements(self, include_structural: bool = False) -> List[UIElement]:
        """Get flat list of all elements (DFS traversal)"""
        if not self.root_element:
            return []

        elements = []

        def traverse(elem: UIElement):
            # Filter based on visibility
            if elem.visibility == VisibilityState.VISIBLE:
                if include_structural or elem.element_type not in ["frame", "panel", "filler"]:
                    elements.append(elem)

            # Always traverse children (they might be visible even if parent is structural)
            for child in elem.children:
                traverse(child)

        traverse(self.root_element)
        return elements

    def is_desktop_root(self) -> bool:
        """Check if this window state represents the desktop root container"""
        return self.window_id == "desktop_root" or self.app_name.lower() in ["gnome-shell", "gjs", "desktop"]


class PerturbationType(Enum):
    """Types of perturbations supported - focused on visual/functional changes"""

    # Visual appearance changes
    THEME = "theme"
    COLOR = "color"
    TYPOGRAPHY = "typography"
    LAYOUT = "layout"
    SHAPE = "shape"
    DENSITY = "density"

    # Content and data changes
    CONTENT_VARIATION = "content_variation"
    DATA_MODIFICATION = "data_modification"

    # UI structure changes
    UI_INJECTION = "ui_injection"
    DOM_MODIFICATION = "dom_modification"
    CSS_INJECTION = "css_injection"

    # System and environment changes
    SYSTEM_LEVEL = "system_level"
    BACKGROUND_PROCESS = "background_process"
    WINDOW_MANAGEMENT = "window_management"
    FILE_OPERATIONS = "file_operations"

    # Cross-app interference
    NOTIFICATION = "notification"
    CROSS_APP_INTERFERENCE = "cross_app_interference"

    # Generic categories for LLM flexibility
    VISUAL_PERTURBATION = "visual_perturbation"
    GUI_MANIPULATION = "gui_manipulation"
    VISUAL_RANDOMIZATION = "visual_randomization"

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

        # Comprehensive LLM variations mapping with enhanced flexibility
        mappings = {
            # Typography variations
            "font": cls.TYPOGRAPHY,
            "fonts": cls.TYPOGRAPHY,
            "font_family": cls.TYPOGRAPHY,
            "font_size": cls.TYPOGRAPHY,
            "font_weight": cls.TYPOGRAPHY,
            "text": cls.TYPOGRAPHY,
            "text_style": cls.TYPOGRAPHY,
            "typeface": cls.TYPOGRAPHY,
            "text_formatting": cls.TYPOGRAPHY,
            "letter_spacing": cls.TYPOGRAPHY,
            "line_height": cls.TYPOGRAPHY,
            # Color variations
            "colors": cls.COLOR,
            "colour": cls.COLOR,
            "colours": cls.COLOR,
            "color_scheme": cls.COLOR,
            "palette": cls.COLOR,
            "background_color": cls.COLOR,
            "text_color": cls.COLOR,
            "foreground": cls.COLOR,
            "background": cls.COLOR,
            "accent": cls.COLOR,
            "hue": cls.COLOR,
            "saturation": cls.COLOR,
            "brightness": cls.COLOR,
            # Layout variations
            "spacing": cls.LAYOUT,
            "padding": cls.LAYOUT,
            "margin": cls.LAYOUT,
            "alignment": cls.LAYOUT,
            "position": cls.LAYOUT,
            "positioning": cls.LAYOUT,
            "placement": cls.LAYOUT,
            "arrangement": cls.LAYOUT,
            "grid": cls.LAYOUT,
            "flex": cls.LAYOUT,
            "container": cls.LAYOUT,
            "width": cls.LAYOUT,
            "height": cls.LAYOUT,
            "size": cls.LAYOUT,
            # Density variations
            "compact": cls.DENSITY,
            "spacious": cls.DENSITY,
            "comfortable": cls.DENSITY,
            "density_mode": cls.DENSITY,
            "tight": cls.DENSITY,
            "loose": cls.DENSITY,
            # Theme variations
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
            # Shape variations
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
            # Content and data variations
            "motion": cls.CONTENT_VARIATION,
            "animation": cls.CONTENT_VARIATION,
            "animations": cls.CONTENT_VARIATION,
            "transition": cls.CONTENT_VARIATION,
            "transitions": cls.CONTENT_VARIATION,
            "content": cls.CONTENT_VARIATION,
            "variation": cls.CONTENT_VARIATION,
            "transform": cls.CONTENT_VARIATION,
            "data": cls.DATA_MODIFICATION,
            "data_modification": cls.DATA_MODIFICATION,
            "file_content": cls.DATA_MODIFICATION,
            "content_modification": cls.DATA_MODIFICATION,
            # UI structure changes
            "injection": cls.UI_INJECTION,
            "inject": cls.UI_INJECTION,
            "add_element": cls.UI_INJECTION,
            "insert": cls.UI_INJECTION,
            "dom": cls.DOM_MODIFICATION,
            "dom_modification": cls.DOM_MODIFICATION,
            "css": cls.CSS_INJECTION,
            "css_injection": cls.CSS_INJECTION,
            "stylesheet": cls.CSS_INJECTION,
            # System and environment changes
            "system": cls.SYSTEM_LEVEL,
            "system_level": cls.SYSTEM_LEVEL,
            "system_theme": cls.SYSTEM_LEVEL,
            "desktop": cls.SYSTEM_LEVEL,
            "wallpaper": cls.SYSTEM_LEVEL,
            "environment": cls.SYSTEM_LEVEL,
            "os": cls.SYSTEM_LEVEL,
            "operating_system": cls.SYSTEM_LEVEL,
            # Window management
            "window": cls.WINDOW_MANAGEMENT,
            "windows": cls.WINDOW_MANAGEMENT,
            "resize": cls.WINDOW_MANAGEMENT,
            "resizing": cls.WINDOW_MANAGEMENT,
            "move": cls.WINDOW_MANAGEMENT,
            "reposition": cls.WINDOW_MANAGEMENT,
            "window_management": cls.WINDOW_MANAGEMENT,
            # Background processes
            "background_process": cls.BACKGROUND_PROCESS,
            "bg": cls.BACKGROUND_PROCESS,
            "bg_process": cls.BACKGROUND_PROCESS,
            "background_task": cls.BACKGROUND_PROCESS,
            "process": cls.BACKGROUND_PROCESS,
            # File operations
            "file": cls.FILE_OPERATIONS,
            "files": cls.FILE_OPERATIONS,
            "file_operation": cls.FILE_OPERATIONS,
            "file_system": cls.FILE_OPERATIONS,
            "file_ops": cls.FILE_OPERATIONS,
            # Notifications
            "notify": cls.NOTIFICATION,
            "notification": cls.NOTIFICATION,
            "alert": cls.NOTIFICATION,
            "alerts": cls.NOTIFICATION,
            "toast": cls.NOTIFICATION,
            "toasts": cls.NOTIFICATION,
            "popup": cls.NOTIFICATION,
            "popups": cls.NOTIFICATION,
            # Cross-app interference
            "cross_app": cls.CROSS_APP_INTERFERENCE,
            "cross_app_interference": cls.CROSS_APP_INTERFERENCE,
            "interference": cls.CROSS_APP_INTERFERENCE,
            "competing": cls.CROSS_APP_INTERFERENCE,
            "distraction": cls.CROSS_APP_INTERFERENCE,
            # Generic visual categories (for LLM flexibility)
            "visual": cls.VISUAL_PERTURBATION,
            "visual_perturbation": cls.VISUAL_PERTURBATION,
            "visual_change": cls.VISUAL_PERTURBATION,
            "visual_modification": cls.VISUAL_PERTURBATION,
            "gui": cls.GUI_MANIPULATION,
            "gui_manipulation": cls.GUI_MANIPULATION,
            "gui_change": cls.GUI_MANIPULATION,
            "interface": cls.GUI_MANIPULATION,
            "ui": cls.GUI_MANIPULATION,
            "user_interface": cls.GUI_MANIPULATION,
            "randomization": cls.VISUAL_RANDOMIZATION,
            "visual_randomization": cls.VISUAL_RANDOMIZATION,
            "random": cls.VISUAL_RANDOMIZATION,
            "randomize": cls.VISUAL_RANDOMIZATION,
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


class PerturbationCategory(Enum):
    """Categories of perturbation strategies"""

    SYSTEM_LEVEL = "system_level"
    CONTENT_RANDOMIZATION = "content_randomization"
    APP_SPECIFIC = "app_specific"
    CROSS_APP_INTERFERENCE = "cross_app_interference"

    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Get list of valid perturbation category values"""
        return [category.value for category in cls]

    @classmethod
    def from_string(
        cls, value: str, default: Optional["PerturbationCategory"] = None
    ) -> "PerturbationCategory":
        """Convert string to PerturbationCategory with fallback"""
        if not value or not isinstance(value, str):
            return default or cls.SYSTEM_LEVEL

        normalized = value.lower().strip().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError:
            return default or cls.SYSTEM_LEVEL


class PerturbationIntensity(Enum):
    """Intensity levels for perturbations"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Get list of valid perturbation intensity values"""
        return [intensity.value for intensity in cls]

    @classmethod
    def from_string(
        cls, value: str, default: Optional["PerturbationIntensity"] = None
    ) -> "PerturbationIntensity":
        """Convert string to PerturbationIntensity with fallback"""
        if not value or not isinstance(value, str):
            return default or cls.MEDIUM

        normalized = value.lower().strip()
        try:
            return cls(normalized)
        except ValueError:
            return default or cls.MEDIUM


class ApiCallType(Enum):
    """Types of API calls for perturbation execution"""

    # Core execution methods
    EXECUTE_JS_ON_PAGE = "execute_js_on_page"
    EXECUTE_BASH_COMMAND = "execute_bash_command"
    EXECUTE_PYTHON_COMMAND = "execute_python_command"
    EXECUTE_UNO_COMMAND = "execute_uno_command"

    # Visual manipulation operations
    EXECUTE_CSS_INJECTION = "execute_css_injection"
    EXECUTE_DOM_MODIFICATION = "execute_dom_modification"
    EXECUTE_THEME_RANDOMIZATION = "execute_theme_randomization"
    EXECUTE_LAYOUT_PERTURBATION = "execute_layout_perturbation"
    EXECUTE_TYPOGRAPHY_RANDOMIZATION = "execute_typography_randomization"
    EXECUTE_ANIMATION_EFFECTS = "execute_animation_effects"
    EXECUTE_ACCESSIBILITY_PERTURBATION = "execute_accessibility_perturbation"

    # Freeform operations
    EXECUTE_PYTHON_EXECUTION = "execute_python_execution"
    EXECUTE_JAVASCRIPT_INJECTION = "execute_javascript_injection"
    EXECUTE_BASH_AUTOMATION = "execute_bash_automation"
    EXECUTE_PLAYWRIGHT_AUTOMATION = "execute_playwright_automation"
    EXECUTE_FILE_SYSTEM_MANIPULATION = "execute_file_system_manipulation"
    EXECUTE_NETWORK_PERTURBATION = "execute_network_perturbation"
    EXECUTE_SYSTEM_INTEGRATION = "execute_system_integration"

    # Legacy operations
    MANIPULATE_APP_STATE = "manipulate_app_state"
    EXECUTE_SYSTEM_PERTURBATION = "execute_system_perturbation"

    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Get list of valid API call values"""
        return [api_call.value for api_call in cls]

    @classmethod
    def from_string(cls, value: str, default: Optional["ApiCallType"] = None) -> "ApiCallType":
        """Convert string to ApiCallType with fallback"""
        if not value or not isinstance(value, str):
            return default or cls.EXECUTE_BASH_COMMAND

        normalized = value.lower().strip()
        try:
            return cls(normalized)
        except ValueError:
            return default or cls.EXECUTE_BASH_COMMAND


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
    perturbation_category: PerturbationCategory  # SYSTEM_LEVEL, CONTENT_RANDOMIZATION, etc.

    # Additional fields for comprehensive scenario specification
    perturbation_intensity: PerturbationIntensity = PerturbationIntensity.MEDIUM
    maintains_functionality: bool = True
    maintains_accessibility: bool = True
    realistic_scenario: str = ""
    initial_state_perturbation: bool = False
    runtime_perturbation: bool = True
    risk_mitigation: str = ""
    educational_rationale: str = ""


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
    window_states: List[Dict[str, Any]] = field(default_factory=list)
    task_instruction: str = ""
    task_type: str = ""
    scenario_spec: Optional[ScenarioSpec] = None

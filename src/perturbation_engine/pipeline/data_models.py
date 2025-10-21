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
    """Simplified perturbation types that align with template categories"""

    VISUAL = "visual"
    SYSTEM = "system"
    CONTENT = "content"
    LAYOUT = "layout"
    THEME = "theme"
    NOTIFICATION = "notification"
    FILE_SYSTEM = "file_system"
    WINDOW_MANAGEMENT = "window_management"

    @classmethod
    def from_string(cls, value: str, default: Optional["PerturbationType"] = None) -> "PerturbationType":
        """Convert string to PerturbationType with fallback"""
        if not value or not isinstance(value, str):
            return default or cls.VISUAL

        normalized = value.lower().strip().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError:
            return default or cls.VISUAL


class TemplateCategory(Enum):
    """Categories for perturbation templates - specific to template operations"""

    VISUAL = "visual"
    SYSTEM = "system"
    CONTENT = "content"
    LAYOUT = "layout"
    THEME = "theme"
    NOTIFICATION = "notification"
    FILE_SYSTEM = "file_system"
    WINDOW_MANAGEMENT = "window_management"


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
    """Intensity levels for perturbations - simplified to just strings"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

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

    scenario_count: int = 100
    num_parallel_vms: int = 1
    result_base_dir: str = "./curriculum_results"

    # Curriculum difficulty distribution
    beginner_scenarios: int = 3
    intermediate_scenarios: int = 4
    advanced_scenarios: int = 2


@dataclass
class ScenarioSpec:
    """What/when/how to perturb - immutable specification"""

    scenario_id: str
    scenario_index: int  # Index within the curriculum
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

    # Randomized parameters for diverse concrete commands
    randomized_parameters: Optional[Dict[str, Any]] = None


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
    target_app: str
    current_action: str
    action_history: List[str] = field(default_factory=list)
    cot_context: str = ""
    window_states: List[Dict[str, Any]] = field(default_factory=list)
    task_instruction: str = ""
    task_type: str = ""
    scenario_spec: Optional[ScenarioSpec] = None
    total_steps: Optional[int] = None  # Total steps for strategic timing

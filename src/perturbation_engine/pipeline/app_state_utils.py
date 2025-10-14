"""
App State Utilities: Helper functions for app state normalization, processing, and validation
"""

import datetime
from typing import Any, Dict, List, Optional

from perturbation_engine.pipeline.data_models import UIElement, WindowState

# ============================================================================
# Data Normalization Functions
# ============================================================================


def normalize_window_states(window_states: List[Any]) -> List[WindowState]:
    """
    Convert all window states to consistent WindowState format.

    This function handles the conversion from various input formats (dictionaries,
    WindowState objects) to a consistent list of WindowState objects.

    Args:
        window_states: List of window states in various formats (dict, WindowState, etc.)

    Returns:
        List of normalized WindowState objects

    Example:
        >>> mixed_states = [{"app_name": "vlc", ...}, WindowState(...)]
        >>> normalized = normalize_window_states(mixed_states)
        >>> all(isinstance(state, WindowState) for state in normalized)
        True
    """
    normalized = []
    for window_state in window_states:
        if hasattr(window_state, "app_name") and hasattr(window_state, "window_name"):
            # Already a WindowState object
            normalized.append(window_state)
        elif isinstance(window_state, dict):
            # Convert dict to WindowState object
            normalized.append(_dict_to_window_state(window_state))
        else:
            # Skip invalid states
            continue
    return normalized


def normalize_ui_elements(elements: List[Any]) -> List[UIElement]:
    """
    Convert all elements to consistent UIElement format.

    This function handles the conversion from various input formats (dictionaries,
    UIElement objects) to a consistent list of UIElement objects.

    Args:
        elements: List of elements in various formats (dict, UIElement, etc.)

    Returns:
        List of normalized UIElement objects

    Example:
        >>> mixed_elements = [{"element_type": "button", ...}, UIElement(...)]
        >>> normalized = normalize_ui_elements(mixed_elements)
        >>> all(isinstance(elem, UIElement) for elem in normalized)
        True
    """
    normalized = []
    for element in elements:
        if hasattr(element, "element_type") and hasattr(element, "element_id"):
            # Already a UIElement object
            normalized.append(element)
        elif isinstance(element, dict):
            # Convert dict to UIElement object
            normalized.append(_dict_to_ui_element(element))
        else:
            # Skip invalid elements
            continue
    return normalized


def get_element_property(element: Any, property_name: str, default: Any = None) -> Any:
    """
    Safely get a property from an element regardless of its format.

    Args:
        element: AppElement object or dictionary
        property_name: Name of the property to get
        default: Default value if property not found

    Returns:
        Property value or default
    """
    if hasattr(element, property_name):
        return getattr(element, property_name)
    elif isinstance(element, dict):
        return element.get(property_name, default)
    else:
        return default


# ============================================================================
# Common Utility Functions
# ============================================================================


def get_timestamp() -> str:
    """Get current timestamp in standardized format"""
    return datetime.datetime.now().strftime("%Y%m%d@%H%M%S")


def map_app_name_to_type(app_name: str) -> str:
    """Map application name to standardized app type"""
    app_name_lower = app_name.lower()

    if "code" in app_name_lower or "vscode" in app_name_lower:
        return "code"
    elif "chrome" in app_name_lower or "chromium" in app_name_lower or "google-chrome" in app_name_lower:
        return "chrome"
    elif "calc" in app_name_lower or "spreadsheet" in app_name_lower:
        return "libreoffice_calc"
    elif "writer" in app_name_lower or "document" in app_name_lower:
        return "libreoffice_writer"
    elif "impress" in app_name_lower or "presentation" in app_name_lower:
        return "libreoffice_impress"
    elif "soffice" in app_name_lower or "libreoffice" in app_name_lower:
        return "libreoffice"
    elif "vlc" in app_name_lower or "media" in app_name_lower:
        return "vlc"
    elif "gnome-shell" in app_name_lower:
        return "desktop"
    elif (
        "terminal" in app_name_lower
        or "bash" in app_name_lower
        or "shell" in app_name_lower
        or "gnome-terminal" in app_name_lower
    ):
        return "terminal"
    elif "nautilus" in app_name_lower or "file manager" in app_name_lower or "files" in app_name_lower:
        return "nautilus"
    elif "gimp" in app_name_lower:
        return "gimp"
    else:
        return "unknown"


def infer_app_name_from_title(window_title: str) -> Optional[str]:
    """Infer application name from window title (excluding Electron apps like VS Code)"""
    title_lower = window_title.lower()

    # Skip VS Code - it should be handled via CDP, not X11
    if any(pattern in title_lower for pattern in ["visual studio code", "code -", "vscode"]):
        return None

    # Chrome patterns
    if any(pattern in title_lower for pattern in ["chrome", "chromium", "google chrome"]):
        return "chrome"

    # LibreOffice patterns
    if any(pattern in title_lower for pattern in ["libreoffice calc", "calc", ".xlsx", ".ods"]):
        return "libreoffice-calc"
    elif any(pattern in title_lower for pattern in ["libreoffice writer", "writer", ".docx", ".odt"]):
        return "libreoffice-writer"
    elif any(pattern in title_lower for pattern in ["libreoffice impress", "impress", ".pptx", ".odp"]):
        return "libreoffice-impress"
    elif "libreoffice" in title_lower or "soffice" in title_lower:
        return "libreoffice"

    # VLC patterns
    if "vlc" in title_lower:
        return "vlc"

    # Terminal patterns
    if any(pattern in title_lower for pattern in ["terminal", "bash", "shell", "gnome-terminal"]):
        return "terminal"

    # File manager patterns
    if any(pattern in title_lower for pattern in ["nautilus", "file manager", "files"]):
        return "nautilus"

    # GIMP patterns
    if "gimp" in title_lower:
        return "gimp"

    # If no specific pattern matches, try to extract from common patterns
    # Look for "App Name -" pattern
    if " - " in window_title:
        app_part = window_title.split(" - ")[0].strip()
        if app_part and len(app_part) < 50:  # Reasonable app name length
            return app_part.lower().replace(" ", "-")

    return None


# ============================================================================
# Window Validation Functions
# ============================================================================


def is_valid_window_geometry(geometry: Dict[str, Any], app_name: str = "") -> bool:
    """Check if window geometry is valid using adaptive criteria"""
    if not geometry:
        return False

    width = geometry.get("width", 0)
    height = geometry.get("height", 0)
    mapped = geometry.get("mapped", True)

    # Basic checks
    if width <= 0 or height <= 0:
        return False

    # Check if window is mapped (visible)
    if not mapped:
        return False

    # Adaptive size validation based on application type
    if is_libreoffice_app(app_name):
        # LibreOffice windows can be quite small (toolbars, dialogs)
        # But should still be reasonably sized
        return width >= 50 and height >= 30
    elif "chrome" in app_name.lower() or "chromium" in app_name.lower():
        # Chrome windows should be reasonably sized
        return width >= 200 and height >= 100
    elif "vlc" in app_name.lower():
        # VLC can have small control windows
        return width >= 30 and height >= 30
    elif "code" in app_name.lower() or "vscode" in app_name.lower():
        # VS Code should be reasonably sized
        return width >= 300 and height >= 200
    else:
        # Default: reasonable minimum size for most applications
        return width >= 50 and height >= 50


def get_window_quality_score(
    window_id: str, geometry: Dict[str, Any], x11_title: str, app_name: str
) -> float:
    """Calculate a quality score for window matching (higher = better)"""
    score = 0.0

    # Base score for having valid geometry
    if is_valid_window_geometry(geometry, app_name):
        score += 10.0
    else:
        return 0.0  # Invalid geometry = no score

    # Size bonus (larger windows are usually main windows)
    width = geometry.get("width", 0)
    height = geometry.get("height", 0)
    area = width * height

    if area > 1000000:  # > 1M pixels (roughly 1000x1000)
        score += 5.0
    elif area > 500000:  # > 500K pixels
        score += 3.0
    elif area > 100000:  # > 100K pixels
        score += 1.0

    # Title relevance bonus
    if app_name.lower() in x11_title.lower():
        score += 2.0

    # Avoid VCL/placeholder windows
    if "VCL" in x11_title and "ImplGetDefaultWindow" not in x11_title:
        score -= 3.0

    # Bonus for main application windows
    if is_libreoffice_app(app_name):
        if "LibreOffice" in x11_title and "VCL" not in x11_title:
            score += 3.0
    elif "chrome" in app_name.lower():
        if any(keyword in x11_title.lower() for keyword in ["chrome", "browser", "google"]):
            score += 2.0

    return score


def is_libreoffice_app(app_name: str) -> bool:
    """Check if app_name represents a LibreOffice application"""
    return any(
        pattern in app_name.lower() for pattern in ["libreoffice", "calc", "writer", "impress", "soffice"]
    )


def is_libreoffice_filename_match(atspi_name: str, x11_title: str) -> bool:
    """Check if AT-SPI name matches LibreOffice filename in X11 title"""
    # Extract filename from X11 title (e.g., "Invoices.xlsx - LibreOffice Calc")
    if " - " in x11_title:
        filename_part = x11_title.split(" - ")[0].strip()
        # Check if AT-SPI name contains this filename
        return filename_part.lower() in atspi_name.lower()
    return False


def should_skip_app(app_name: str) -> bool:
    """Skip system/background apps"""
    # Skip known system/background processes
    skip_patterns = [
        "vmware-user",
        "gsd-",
        "ibus-",
        "evolution-alarm",
        "xdg-desktop-portal",
        "org.gnome.Software",
        "gnome-shell",  # Desktop environment
        "gjs",  # GNOME JavaScript
        "gnome-session",  # Session manager
        "dbus",  # D-Bus daemon
        "systemd",  # System daemon
        "pulseaudio",  # Audio daemon
        "NetworkManager",  # Network manager
        "polkit",  # Policy kit
        "udisks",  # Disk manager
        "upower",  # Power manager
        "accounts-daemon",  # Accounts daemon
        "zeitgeist",  # Activity logger
        "tracker",  # File indexer
        "evolution",  # Email client (if not explicitly needed)
        "thunderbird",  # Email client (if not explicitly needed)
        "firefox",  # Browser (if not explicitly needed)
        "nautilus",  # File manager (if not explicitly needed)
    ]
    return any(pattern in app_name for pattern in skip_patterns)


# ============================================================================
# Helper Functions for Data Conversion
# ============================================================================


def _dict_to_window_state(data: dict) -> WindowState:
    """Convert dictionary to WindowState object"""

    # Convert elements
    elements = []
    if "elements" in data:
        for elem_data in data["elements"]:
            elements.append(_dict_to_ui_element(elem_data))

    # Build root element tree
    root_element = None
    if elements:
        root_element = _build_element_tree(elements)

    return WindowState(
        window_id=data.get("window_id", ""),
        window_name=data.get("window_name", ""),
        app_name=data.get("app_name", ""),
        is_active=data.get("is_active", False),
        is_modal=data.get("is_modal", False),
        is_minimized=data.get("is_minimized", False),
        geometry=data.get("geometry", {}),
        z_order=data.get("z_order", 0),
        x11_window_id=data.get("x11_window_id"),
        is_mapped=data.get("is_mapped", True),
        desktop=data.get("desktop", 0),
        root_element=root_element,
    )


def _dict_to_ui_element(data: dict) -> UIElement:
    """Convert dictionary to UIElement object"""
    from perturbation_engine.pipeline.data_models import VisibilityState

    return UIElement(
        element_id=data.get("element_id", ""),
        element_type=data.get("element_type", ""),
        name=data.get("name", ""),
        position=data.get("position", {}),
        parent_id=data.get("parent_id"),
        children=[],  # Will be populated by _build_element_tree
        depth=data.get("depth", 0),
        visibility=VisibilityState(data.get("visibility", "visible")),
        is_enabled=data.get("is_enabled", True),
        is_focused=data.get("is_focused", False),
        is_expanded=data.get("is_expanded", False),
        properties=data.get("properties", {}),
    )


def _build_element_tree(elements: List[UIElement]) -> UIElement:
    """Build element tree from flat list"""
    if not elements:
        return None

    # Find root element (no parent)
    root_elements = [elem for elem in elements if elem.parent_id is None]
    if not root_elements:
        return elements[0]  # Fallback to first element

    root = root_elements[0]

    # Build parent-child relationships
    element_map = {elem.element_id: elem for elem in elements}

    for elem in elements:
        if elem.parent_id and elem.parent_id in element_map:
            parent = element_map[elem.parent_id]
            parent.children.append(elem)

    return root

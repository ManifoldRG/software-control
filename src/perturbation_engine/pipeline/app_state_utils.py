"""
App State Utilities: Helper functions for app state normalization and processing
"""

from typing import Any, List

from perturbation_engine.pipeline.data_models import UIElement, WindowState


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

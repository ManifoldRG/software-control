"""
App State Utilities: Helper functions for app state normalization and processing
"""

from typing import Any, List

from perturbation_engine.pipeline.data_models import AppElement, AppState


def normalize_app_states(app_states: List[Any]) -> List[AppState]:
    """
    Convert all app states to consistent AppState format.

    This function handles the conversion from various input formats (dictionaries,
    AppState objects) to a consistent list of AppState objects.

    Args:
        app_states: List of app states in various formats (dict, AppState, etc.)

    Returns:
        List of normalized AppState objects

    Example:
        >>> mixed_states = [{"app_name": "vlc", ...}, AppState(...)]
        >>> normalized = normalize_app_states(mixed_states)
        >>> all(isinstance(state, AppState) for state in normalized)
        True
    """
    normalized = []
    for app_state in app_states:
        if hasattr(app_state, "app_name"):
            # Already an AppState object
            normalized.append(app_state)
        elif isinstance(app_state, dict):
            # Convert dict to AppState object
            normalized.append(AppState.from_dict(app_state))
        else:
            # Skip invalid states
            continue
    return normalized


def normalize_elements(elements: List[Any]) -> List[AppElement]:
    """
    Convert all elements to consistent AppElement format.

    This function handles the conversion from various input formats (dictionaries,
    AppElement objects) to a consistent list of AppElement objects.

    Args:
        elements: List of elements in various formats (dict, AppElement, etc.)

    Returns:
        List of normalized AppElement objects

    Example:
        >>> mixed_elements = [{"element_type": "button", ...}, AppElement(...)]
        >>> normalized = normalize_elements(mixed_elements)
        >>> all(isinstance(elem, AppElement) for elem in normalized)
        True
    """
    normalized = []
    for element in elements:
        if hasattr(element, "element_type"):
            # Already an AppElement object
            normalized.append(element)
        elif isinstance(element, dict):
            # Convert dict to AppElement object
            normalized.append(AppElement.from_dict(element))
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

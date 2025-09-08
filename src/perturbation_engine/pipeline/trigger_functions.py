"""Trigger and validation functions for perturbations"""

from typing import Any, Dict


def step_range_trigger(step_idx: int, obs: Dict, env: Any, params: Dict) -> bool:
    """Trigger if current step is within range"""
    return params["start"] <= step_idx <= params["end"]


def url_contains_trigger(step_idx: int, obs: Dict, env: Any, params: Dict) -> bool:
    """Trigger if current URL contains fragment"""
    url_fragment = params.get("fragment", "")
    try:
        if hasattr(env, "controller") and hasattr(env.controller, "get_current_url"):
            current_url = env.controller.get_current_url()
            return url_fragment.lower() in current_url.lower()
    except Exception:
        pass
    return False


def element_exists_trigger(step_idx: int, obs: Dict, env: Any, params: Dict) -> bool:
    """Trigger if element exists in DOM"""
    selector = params.get("selector", "")
    try:
        if hasattr(env, "controller") and hasattr(env.controller, "page"):
            page = env.controller.page
            element = page.query_selector(selector)
            return element is not None
    except Exception:
        pass
    return False


def element_created_validation(step_idx: int, obs: Dict, env: Any, params: Dict) -> bool:
    """Validate that element was created"""
    selector = params.get("selector", "")
    try:
        if hasattr(env, "controller") and hasattr(env.controller, "page"):
            page = env.controller.page
            element = page.query_selector(selector)
            return element is not None
    except Exception:
        pass
    return False


# Registry for multiprocessing-safe function lookup
TRIGGER_FUNCTIONS = {
    "step_range": step_range_trigger,
    "url_contains": url_contains_trigger,
    "element_exists": element_exists_trigger,
}

VALIDATION_FUNCTIONS = {
    "element_created": element_created_validation,
}

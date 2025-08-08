"""Scene analysis data structures for the perturbation pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Selector = str  # CSS selector


class ComponentType(Enum):
    """Types of functional UI components."""

    BUTTON = "button"
    INPUT_FIELD = "input_field"
    DROPDOWN = "dropdown"
    NAVIGATION = "navigation"
    HEADER = "header"
    FOOTER = "footer"
    CARD = "card"
    MODAL = "modal"
    TAB = "tab"
    BREADCRUMB = "breadcrumb"
    SEARCH_BAR = "search_bar"
    FORM = "form"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    TEXT_BLOCK = "text_block"
    UNKNOWN = "unknown"


@dataclass
class ComponentAttribute:
    """Attribute of a functional component."""

    name: str
    value: Any
    description: str = ""


@dataclass
class Element:
    """Represents a DOM element identified by scene analysis."""

    element_id: str
    element_type: str  # button, input, div, etc.
    selector: Selector  # CSS selector to target this element
    is_interactive: bool
    text_content: str = ""
    attributes: dict[str, str] = field(default_factory=dict)  # HTML attributes
    bounding_box: dict[str, int] | None = None  # x, y, width, height
    parent_id: str | None = None  # ID of parent element
    children_ids: list[str] = field(default_factory=list)  # IDs of child elements


@dataclass
class FunctionalComponent:
    """Represents a functional UI component that groups related elements."""

    component_id: str
    component_type: ComponentType
    elements: list[Element]  # The DOM elements that make up this component
    attributes: list[ComponentAttribute] = field(default_factory=list)
    bounding_box: dict[str, int] | None = None  # x, y, width, height
    is_interactive: bool = False
    text_content: str = ""
    selector: str = ""  # Primary selector for the component


@dataclass
class Layout:
    """Layout of the scene (placeholder for MVP)."""

    pass


@dataclass
class SceneAnalysis:
    """Results from analyzing a scene/observation."""

    scene_id: str
    plausibility_score: float
    solvability_score: float

    elements: list[Element] = field(default_factory=list)
    goal_relevant_elements: list[Element] = field(default_factory=list)
    background_elements: list[Element] = field(default_factory=list)
    functional_components: list[FunctionalComponent] = field(default_factory=list)
    layout: Layout | None = None

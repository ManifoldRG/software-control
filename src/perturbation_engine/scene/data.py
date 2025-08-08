"""Scene analysis data structures for the perturbation pipeline."""

from dataclasses import dataclass

# Type aliases
Selector = str  # CSS selector


@dataclass
class Element:
    """Represents a DOM element identified by scene analysis."""

    element_id: str
    element_type: str  # button, input, div, etc.
    selector: Selector  # CSS selector to target this element
    is_interactive: bool


@dataclass
class Layout:
    """Layout of the scene (placeholder for MVP)."""

    pass


@dataclass
class SceneAnalysis:
    """Results from analyzing a scene/observation."""

    scene_id: str
    elements: list[Element]
    layout: Layout
    goal_relevant_elements: list[Element]
    background_elements: list[Element]
    plausibility_score: float
    solvability_score: float

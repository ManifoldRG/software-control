"""Scene analysis data structures for the perturbation pipeline."""

from dataclasses import dataclass, field

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
    plausibility_score: float
    solvability_score: float

    elements: list[Element] = field(default_factory=list)
    goal_relevant_elements: list[Element] = field(default_factory=list)
    background_elements: list[Element] = field(default_factory=list)
    layout: Layout | None = None

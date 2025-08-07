from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar


class PerturbationType(Enum):
    """Types of perturbations that can be applied."""

    COLOR_CHANGE = "color_change"
    POSITION_SHIFT = "position_shift"
    SIZE_SCALE = "size_scale"
    TEXT_SUBSTITUTION = "text_substitution"
    ELEMENT_ADDITION = "element_addition"
    CSS_MODIFICATION = "css_modification"
    ELEMENT_REMOVAL = "element_removal"
    LAYOUT_REORGANIZATION = "layout_reorganization"


class ElementType(Enum):
    """Types of DOM elements."""

    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    IMAGE = "image"
    DIV = "div"
    SPAN = "span"
    FORM = "form"
    SELECT = "select"
    TEXTAREA = "textarea"


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation."""

    type: PerturbationType
    parameters: dict[str, Any]
    bounds: tuple[float, float] | None = None
    probability: float = 1.0  # Probability of applying this perturbation
    target_elements: list[str] | None = None  # CSS selectors or element types


@dataclass
class SceneData:
    """Represents a scene with its metadata."""

    scene_id: str
    elements: list[dict[str, Any]]
    metadata: dict[str, Any]
    screenshot_path: Path | None = None
    dom_tree: dict[str, Any] | None = None


@dataclass
class PerturbationResult:
    """Result of applying perturbations."""

    original_scene: SceneData
    perturbed_scene: SceneData
    applied_perturbations: list[PerturbationConfig]
    quality_score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PipelineConfig:
    """Configuration for the perturbation pipeline."""

    perturbation_configs: list[PerturbationConfig]
    max_perturbations: int = 5
    quality_threshold: float = 0.7
    output_dir: Path | None = None
    preserve_original: bool = True


@dataclass
class QualityMetrics:
    """Quality metrics for perturbation evaluation."""

    visual_similarity: float
    functional_preservation: float
    task_compatibility: float
    overall_score: float


# Type aliases for cleaner code
Element = dict[str, Any]
PerturbationPipeline = TypeVar("PerturbationPipeline")
SceneAnalyzer = TypeVar("SceneAnalyzer")
SceneVerifier = TypeVar("SceneVerifier")

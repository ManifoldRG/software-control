"""
Curriculum configuration for controlled trajectory generation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RandomizationType(Enum):
    """Types of randomizations available"""

    UI_VISUAL = "ui_visual"  # Theme changes, layout modifications
    VISUAL_DISTRACTOR = "visual_distractor"  # Adding distracting elements
    INSTRUCTION_VARIATION = "instruction_variation"  # Text rephrasing
    LAYOUT_PERTURBATION = "layout_perturbation"  # UI element reordering
    COMBINED = "combined"  # Multiple randomization types combined


class IntensityLevel(Enum):
    """Intensity levels for curriculum generation"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class RandomizationRatio:
    """Ratio configuration for randomization types"""

    ui_visual: float = 0.25
    visual_distractor: float = 0.25
    instruction_variation: float = 0.15
    layout_perturbation: float = 0.15
    combined: float = 0.2  # Combined randomizations

    def validate(self) -> bool:
        """Validate that ratios sum to 1.0"""
        total = (
            self.ui_visual
            + self.visual_distractor
            + self.instruction_variation
            + self.layout_perturbation
            + self.combined
        )
        return abs(total - 1.0) < 0.01


@dataclass
class IntensityConfig:
    """Configuration for intensity levels"""

    easy: float = 0.2  # 20% intensity
    medium: float = 0.5  # 50% intensity
    hard: float = 0.8  # 80% intensity


@dataclass
class CurriculumConfig:
    """Main curriculum configuration"""

    # Generation parameters
    num_trajectories: int = 100
    num_seeds: int = 1

    # Randomization control
    randomization_ratios: RandomizationRatio = field(default_factory=RandomizationRatio)
    intensity_levels: IntensityConfig = field(default_factory=IntensityConfig)

    # Level distribution (how many trajectories per level)
    level_distribution: Dict[IntensityLevel, float] = None

    # Feature distribution targets (for real-world alignment)
    target_action_distribution: Optional[Dict[str, float]] = None
    target_ui_component_distribution: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Initialize default level distribution if not provided"""
        if self.level_distribution is None:
            self.level_distribution = {
                IntensityLevel.EASY: 0.4,  # 40% easy
                IntensityLevel.MEDIUM: 0.4,  # 40% medium
                IntensityLevel.HARD: 0.2,  # 20% hard
            }

        # Validate ratios
        if not self.randomization_ratios.validate():
            raise ValueError("Randomization ratios must sum to 1.0")

        # Validate level distribution
        total_level = sum(self.level_distribution.values())
        if abs(total_level - 1.0) > 0.01:
            raise ValueError("Level distribution must sum to 1.0")


@dataclass
class CombinedRandomization:
    """Configuration for combined randomization types"""

    primary_type: RandomizationType
    secondary_types: List[RandomizationType]
    primary_weight: float = 0.6  # Weight of primary randomization
    secondary_weight: float = 0.4  # Weight of secondary randomizations

    def get_combined_types(self) -> List[RandomizationType]:
        """Get all types in this combination"""
        return [self.primary_type] + self.secondary_types


@dataclass
class CurriculumLevel:
    """Represents a single curriculum level"""

    intensity: IntensityLevel
    randomization_type: RandomizationType
    intensity_value: float
    count: int  # Number of trajectories for this level
    combined_config: Optional[CombinedRandomization] = None  # For combined randomizations


class CombinedRandomizationPresets:
    """Predefined combined randomization configurations"""

    @staticmethod
    def visual_chaos() -> CombinedRandomization:
        """UI visual + visual distractors for maximum visual confusion"""
        return CombinedRandomization(
            primary_type=RandomizationType.UI_VISUAL,
            secondary_types=[RandomizationType.VISUAL_DISTRACTOR],
            primary_weight=0.6,
            secondary_weight=0.4,
        )

    @staticmethod
    def layout_instruction_chaos() -> CombinedRandomization:
        """Layout + instruction variations for cognitive load"""
        return CombinedRandomization(
            primary_type=RandomizationType.LAYOUT_PERTURBATION,
            secondary_types=[RandomizationType.INSTRUCTION_VARIATION],
            primary_weight=0.5,
            secondary_weight=0.5,
        )

    @staticmethod
    def full_chaos() -> CombinedRandomization:
        """All randomization types combined for maximum difficulty"""
        return CombinedRandomization(
            primary_type=RandomizationType.UI_VISUAL,
            secondary_types=[
                RandomizationType.VISUAL_DISTRACTOR,
                RandomizationType.INSTRUCTION_VARIATION,
                RandomizationType.LAYOUT_PERTURBATION,
            ],
            primary_weight=0.4,
            secondary_weight=0.6,
        )

    @staticmethod
    def subtle_combination() -> CombinedRandomization:
        """Subtle combination of UI and instruction changes"""
        return CombinedRandomization(
            primary_type=RandomizationType.INSTRUCTION_VARIATION,
            secondary_types=[RandomizationType.UI_VISUAL],
            primary_weight=0.7,
            secondary_weight=0.3,
        )


class CurriculumPresets:
    """Predefined curriculum presets for common use cases"""

    @staticmethod
    def balanced_curriculum(num_trajectories: int = 100) -> CurriculumConfig:
        """Balanced curriculum with equal distribution"""
        return CurriculumConfig(
            num_trajectories=num_trajectories,
            randomization_ratios=RandomizationRatio(
                ui_visual=0.25, visual_distractor=0.25, instruction_variation=0.25, layout_perturbation=0.25
            ),
            level_distribution={
                IntensityLevel.EASY: 0.33,
                IntensityLevel.MEDIUM: 0.33,
                IntensityLevel.HARD: 0.34,
            },
        )

    @staticmethod
    def progressive_curriculum(num_trajectories: int = 100) -> CurriculumConfig:
        """Progressive curriculum starting easy, getting harder"""
        return CurriculumConfig(
            num_trajectories=num_trajectories,
            level_distribution={
                IntensityLevel.EASY: 0.5,  # 50% easy
                IntensityLevel.MEDIUM: 0.3,  # 30% medium
                IntensityLevel.HARD: 0.2,  # 20% hard
            },
        )

    @staticmethod
    def challenging_curriculum(num_trajectories: int = 100) -> CurriculumConfig:
        """Challenging curriculum with more hard examples"""
        return CurriculumConfig(
            num_trajectories=num_trajectories,
            level_distribution={
                IntensityLevel.EASY: 0.2,  # 20% easy
                IntensityLevel.MEDIUM: 0.3,  # 30% medium
                IntensityLevel.HARD: 0.5,  # 50% hard
            },
        )

    @staticmethod
    def ui_focused_curriculum(num_trajectories: int = 100) -> CurriculumConfig:
        """UI-focused curriculum emphasizing visual perturbations"""
        return CurriculumConfig(
            num_trajectories=num_trajectories,
            randomization_ratios=RandomizationRatio(
                ui_visual=0.5,  # 50% UI visual
                visual_distractor=0.3,  # 30% distractors
                instruction_variation=0.1,  # 10% instruction
                layout_perturbation=0.1,  # 10% layout
            ),
        )

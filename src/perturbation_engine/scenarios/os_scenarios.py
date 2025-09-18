"""
Perturbation scenarios for OSWorld OS task type.
"""

import random
from typing import Any, Dict, List

from perturbation_engine.data_types import Constants, DifficultyLevel
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.scenarios.base_scenarios import BasePerturbationScenario


class OSInvarianceScenario(BasePerturbationScenario):
    """OS invariance perturbation scenario."""

    def __init__(self):
        super().__init__("os", "invariance")

    def get_difficulty_levels(self) -> List[DifficultyLevel]:
        """Get difficulty levels for OS invariance scenarios."""
        return [
            DifficultyLevel(
                level=1, intensity=0.2, perturbation_count=1, parameters={"theme_options": ["light", "dark"]}
            ),
            DifficultyLevel(
                level=2,
                intensity=0.4,
                perturbation_count=2,
                parameters={"theme_options": ["light", "dark"], "window_management": True},
            ),
            DifficultyLevel(
                level=3,
                intensity=0.6,
                perturbation_count=3,
                parameters={"theme_options": ["light", "dark", "high_contrast"], "window_management": True},
            ),
            DifficultyLevel(
                level=4,
                intensity=0.8,
                perturbation_count=4,
                parameters={
                    "theme_options": ["light", "dark", "high_contrast"],
                    "window_management": True,
                    "resolution_change": True,
                },
            ),
        ]

    def apply_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Apply OS invariance setup perturbations based on difficulty level."""
        method_name = f"_apply_setup_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(task_config, difficulty_level)
        else:
            return self._apply_generic_setup_perturbations(task_config, difficulty_level)

    def apply_runtime_perturbations(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply OS invariance runtime perturbations based on difficulty level."""
        method_name = f"_apply_runtime_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(env, difficulty_level, step_idx, obs)
        else:
            return self._apply_generic_runtime_perturbations(env, difficulty_level, step_idx, obs)

    def _apply_setup_level_1(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 1: Basic OS theme changes."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Light instruction rephrasing
        if random.random() < 0.3:
            perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.2)

        # Simple OS theme
        if "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            perturbed_config["os_theme"] = theme

        return perturbed_config

    def _apply_setup_level_2(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 2: OS theme + window management context."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Moderate instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.4)

        # OS theme with window management
        if "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            perturbed_config["os_theme"] = theme
            perturbed_config["window_management"] = True

        return perturbed_config

    def _apply_setup_level_3(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 3: Complex OS setup with multiple themes and window management."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Heavy instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.6)

        # Multiple OS themes
        if "theme_options" in difficulty_level.parameters:
            themes = random.sample(difficulty_level.parameters["theme_options"], 2)
            perturbed_config["os_theme"] = themes[0]
            perturbed_config["os_theme_fallback"] = themes[1]
            perturbed_config["window_management"] = True
            perturbed_config["theme_cycling"] = True

        return perturbed_config

    def _apply_setup_level_4(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 4: Maximum OS complexity with all features."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Maximum instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.8)

        # Complex OS setup
        if "theme_options" in difficulty_level.parameters:
            themes = random.sample(difficulty_level.parameters["theme_options"], 3)
            perturbed_config["os_theme_cycle"] = themes
            perturbed_config["window_management"] = True
            perturbed_config["theme_cycling"] = True
            perturbed_config["resolution_change"] = True

        if "resolution_options" in difficulty_level.parameters:
            resolutions = random.sample(difficulty_level.parameters["resolution_options"], 2)
            perturbed_config["resolution_cycle"] = resolutions

        # Add accessibility features
        perturbed_config["accessibility_mode"] = True
        perturbed_config["high_contrast_os"] = True

        return perturbed_config

    def _apply_generic_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Generic setup perturbations for unknown difficulty levels."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        if difficulty_level.intensity > 0.3:
            perturbed_config["instruction"] = self._rephrase_instruction(
                original_instruction, difficulty_level.intensity
            )

        if "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            perturbed_config["os_theme"] = theme

        return perturbed_config

    def _apply_runtime_level_1(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 1: Occasional simple window management."""
        if step_idx % 12 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.2, "num_elements": 1}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_2(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 2: Regular window management."""
        if step_idx % 8 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.4, "num_elements": 2}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_3(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 3: Frequent window management + resolution changes."""
        if step_idx % 6 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.6, "num_elements": 3}
            )
        elif step_idx % 10 == 0 and "resolution_options" in difficulty_level.parameters:
            resolution = random.choice(difficulty_level.parameters["resolution_options"])
            return self._execute_controller_command(
                env, Constants.MODIFY_RESOLUTION_CMD, {"resolution": resolution, "intensity": 0.4}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_4(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 4: Maximum OS perturbation frequency."""
        if step_idx % 4 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.8, "num_elements": 4}
            )
        elif step_idx % 7 == 0 and "resolution_options" in difficulty_level.parameters:
            resolution = random.choice(difficulty_level.parameters["resolution_options"])
            return self._execute_controller_command(
                env, Constants.MODIFY_RESOLUTION_CMD, {"resolution": resolution, "intensity": 0.6}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_generic_runtime_perturbations(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generic runtime perturbations for unknown difficulty levels."""
        if "window_management" in difficulty_level.parameters and step_idx % 7 == 0:
            return self._execute_controller_command(
                env,
                Constants.UI_REORDERING_CMD,
                {
                    "intensity": difficulty_level.intensity,
                    "num_elements": difficulty_level.perturbation_count,
                },
            )

        return {"applied": False, "reason": "no_perturbation_needed"}

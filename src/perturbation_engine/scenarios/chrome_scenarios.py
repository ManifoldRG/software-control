"""
Perturbation scenarios for OSWorld Chrome task type.
"""

import random
from typing import Any, Dict, List

from perturbation_engine.data_types import Constants, DifficultyLevel
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.scenarios.base_scenarios import BasePerturbationScenario


class ChromeInvarianceScenario(BasePerturbationScenario):
    """Chrome invariance perturbation scenario."""

    def __init__(self):
        super().__init__("chrome", "invariance")

    def get_difficulty_levels(self) -> List[DifficultyLevel]:
        """Get difficulty levels for Chrome invariance scenarios."""
        return [
            DifficultyLevel(
                level=1, intensity=0.2, perturbation_count=1, parameters={"theme_options": ["light", "dark"]}
            ),
            DifficultyLevel(
                level=2,
                intensity=0.4,
                perturbation_count=2,
                parameters={"theme_options": ["light", "dark", "high_contrast"]},
            ),
            DifficultyLevel(
                level=3,
                intensity=0.6,
                perturbation_count=3,
                parameters={
                    "theme_options": ["light", "dark", "high_contrast"],
                    "resolution_options": [(1920, 1080), (1366, 768)],
                },
            ),
            DifficultyLevel(
                level=4,
                intensity=0.8,
                perturbation_count=4,
                parameters={
                    "theme_options": ["light", "dark", "high_contrast"],
                    "resolution_options": [(1920, 1080), (1366, 768), (1440, 900)],
                },
            ),
        ]

    def apply_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Apply Chrome invariance setup perturbations based on difficulty level."""
        # Delegate to difficulty-specific method
        method_name = f"_apply_setup_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(task_config, difficulty_level)
        else:
            # Fallback to generic method
            return self._apply_generic_setup_perturbations(task_config, difficulty_level)

    def apply_runtime_perturbations(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply Chrome invariance runtime perturbations based on difficulty level."""
        # Delegate to difficulty-specific method
        method_name = f"_apply_runtime_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(env, difficulty_level, step_idx, obs)
        else:
            # Fallback to generic method
            return self._apply_generic_runtime_perturbations(env, difficulty_level, step_idx, obs)

    def _apply_setup_level_1(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 1: Basic theme changes only."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Light instruction rephrasing
        if random.random() < 0.3:
            perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.2)

        # Simple theme requirement
        if "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            perturbed_config["instruction"] = f"{perturbed_config['instruction']} (Use {theme} theme)"
            perturbed_config["theme_requirement"] = theme

        return perturbed_config

    def _apply_setup_level_2(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 2: Theme changes + instruction complexity."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Moderate instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.4)

        # Theme with additional context
        if "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            perturbed_config["instruction"] = (
                f"{perturbed_config['instruction']} (Note: Ensure {theme} theme is applied)"
            )
            perturbed_config["theme_requirement"] = theme
            perturbed_config["theme_context"] = "high_contrast" if theme == "high_contrast" else "standard"

        return perturbed_config

    def _apply_setup_level_3(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 3: Complex instruction + multiple theme options."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Heavy instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.6)

        # Multiple theme requirements
        if "theme_options" in difficulty_level.parameters:
            themes = random.sample(difficulty_level.parameters["theme_options"], 2)
            perturbed_config["instruction"] = (
                f"{perturbed_config['instruction']} (Apply {themes[0]} theme, fallback to {themes[1]})"
            )
            perturbed_config["theme_requirement"] = themes[0]
            perturbed_config["theme_fallback"] = themes[1]

        # Add resolution context
        if "resolution_options" in difficulty_level.parameters:
            resolution = random.choice(difficulty_level.parameters["resolution_options"])
            perturbed_config["resolution_requirement"] = resolution

        return perturbed_config

    def _apply_setup_level_4(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 4: Maximum complexity with all perturbation types."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        # Maximum instruction rephrasing
        perturbed_config["instruction"] = self._rephrase_instruction(original_instruction, 0.8)

        # Complex theme and resolution requirements
        if "theme_options" in difficulty_level.parameters:
            themes = random.sample(difficulty_level.parameters["theme_options"], 3)
            perturbed_config["instruction"] = (
                f"{perturbed_config['instruction']} (Cycle through themes: {', '.join(themes)})"
            )
            perturbed_config["theme_cycle"] = themes

        if "resolution_options" in difficulty_level.parameters:
            resolutions = random.sample(difficulty_level.parameters["resolution_options"], 2)
            perturbed_config["resolution_cycle"] = resolutions

        # Add accessibility requirements
        perturbed_config["accessibility_mode"] = True
        perturbed_config["high_contrast_required"] = True

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
            perturbed_config["instruction"] = f"{perturbed_config['instruction']} (Note: Use {theme} theme)"
            perturbed_config["theme_requirement"] = theme

        return perturbed_config

    def _apply_runtime_level_1(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 1: Simple theme changes only."""
        if step_idx % 10 == 0 and "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            return self._execute_controller_command(
                env, Constants.THEME_CHANGE_CMD, {"theme": theme, "intensity": 0.2}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_2(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 2: Theme changes + occasional UI reordering."""
        if step_idx % 8 == 0 and "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            return self._execute_controller_command(
                env, Constants.THEME_CHANGE_CMD, {"theme": theme, "intensity": 0.4}
            )
        elif step_idx % 15 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.3, "num_elements": 1}
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_3(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 3: Frequent theme changes + UI reordering + resolution changes."""
        if step_idx % 5 == 0 and "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            return self._execute_controller_command(
                env, Constants.THEME_CHANGE_CMD, {"theme": theme, "intensity": 0.6}
            )
        elif step_idx % 7 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.5, "num_elements": 2}
            )
        elif step_idx % 12 == 0 and "resolution_options" in difficulty_level.parameters:
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
        """Level 4: Maximum perturbation frequency and complexity."""
        if step_idx % 3 == 0 and "theme_options" in difficulty_level.parameters:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            return self._execute_controller_command(
                env, Constants.THEME_CHANGE_CMD, {"theme": theme, "intensity": 0.8}
            )
        elif step_idx % 4 == 0:
            return self._execute_controller_command(
                env, Constants.UI_REORDERING_CMD, {"intensity": 0.7, "num_elements": 3}
            )
        elif step_idx % 6 == 0 and "resolution_options" in difficulty_level.parameters:
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
        if "theme_options" in difficulty_level.parameters and step_idx % 5 == 0:
            theme = random.choice(difficulty_level.parameters["theme_options"])
            return self._execute_controller_command(
                env, Constants.THEME_CHANGE_CMD, {"theme": theme, "intensity": difficulty_level.intensity}
            )

        if step_idx % 3 == 0:
            return self._execute_controller_command(
                env,
                Constants.UI_REORDERING_CMD,
                {
                    "intensity": difficulty_level.intensity,
                    "num_elements": difficulty_level.perturbation_count,
                },
            )

        return {"applied": False, "reason": "no_perturbation_needed"}


class ChromeDistractorScenario(BasePerturbationScenario):
    """Chrome distractor perturbation scenario."""

    def __init__(self):
        super().__init__("chrome", "distractor")

    def get_difficulty_levels(self) -> List[DifficultyLevel]:
        """Get difficulty levels for Chrome distractor scenarios."""
        return [
            DifficultyLevel(
                level=1, intensity=0.2, perturbation_count=1, parameters={"component_types": ["button"]}
            ),
            DifficultyLevel(
                level=2,
                intensity=0.4,
                perturbation_count=2,
                parameters={"component_types": ["button", "div"]},
            ),
            DifficultyLevel(
                level=3,
                intensity=0.6,
                perturbation_count=3,
                parameters={"component_types": ["button", "div", "input"], "popup_types": ["modal"]},
            ),
            DifficultyLevel(
                level=4,
                intensity=0.8,
                perturbation_count=4,
                parameters={
                    "component_types": ["button", "div", "input", "select"],
                    "popup_types": ["modal", "notification"],
                },
            ),
        ]

    def apply_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Apply Chrome distractor setup perturbations based on difficulty level."""
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
        """Apply Chrome distractor runtime perturbations based on difficulty level."""
        method_name = f"_apply_runtime_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(env, difficulty_level, step_idx, obs)
        else:
            return self._apply_generic_runtime_perturbations(env, difficulty_level, step_idx, obs)

    def _apply_setup_level_1(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 1: Basic distractor configuration."""
        perturbed_config = task_config.copy()
        perturbed_config["distractor_config"] = {
            "enabled": True,
            "difficulty_level": 1,
            "simple_distractors": True,
            "component_types": ["button"],
        }
        return perturbed_config

    def _apply_setup_level_2(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 2: Multiple distractor types."""
        perturbed_config = task_config.copy()
        perturbed_config["distractor_config"] = {
            "enabled": True,
            "difficulty_level": 2,
            "component_types": ["button", "div"],
            "distractor_placement": "random",
        }
        return perturbed_config

    def _apply_setup_level_3(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 3: Complex distractors with popups."""
        perturbed_config = task_config.copy()
        perturbed_config["distractor_config"] = {
            "enabled": True,
            "difficulty_level": 3,
            "component_types": ["button", "div", "input"],
            "popup_types": ["modal"],
            "distractor_placement": "strategic",
            "interactive_distractors": True,
        }
        return perturbed_config

    def _apply_setup_level_4(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 4: Maximum distractor complexity."""
        perturbed_config = task_config.copy()
        perturbed_config["distractor_config"] = {
            "enabled": True,
            "difficulty_level": 4,
            "component_types": ["button", "div", "input", "select"],
            "popup_types": ["modal", "notification"],
            "distractor_placement": "aggressive",
            "interactive_distractors": True,
            "animated_distractors": True,
            "misleading_labels": True,
        }
        return perturbed_config

    def _apply_generic_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Generic setup perturbations for unknown difficulty levels."""
        perturbed_config = task_config.copy()
        perturbed_config["distractor_config"] = {
            "enabled": True,
            "difficulty_level": difficulty_level.level,
            "parameters": difficulty_level.parameters,
        }
        return perturbed_config

    def _apply_runtime_level_1(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 1: Occasional simple distractors."""
        if step_idx % 8 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {"intensity": 0.2, "num_components": 1, "component_types": ["button"]},
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_2(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 2: Regular distractors with multiple types."""
        if step_idx % 6 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {"intensity": 0.4, "num_components": 2, "component_types": ["button", "div"]},
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_3(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 3: Frequent distractors + popups."""
        if step_idx % 4 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {"intensity": 0.6, "num_components": 3, "component_types": ["button", "div", "input"]},
            )
        elif step_idx % 8 == 0:
            return self._execute_controller_command(
                env,
                Constants.INJECT_POPUPS_CMD,
                {"intensity": 0.5, "num_popups": 1, "popup_types": ["modal"]},
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_4(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 4: Maximum distractor frequency and complexity."""
        if step_idx % 3 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": 0.8,
                    "num_components": 4,
                    "component_types": ["button", "div", "input", "select"],
                },
            )
        elif step_idx % 5 == 0:
            return self._execute_controller_command(
                env,
                Constants.INJECT_POPUPS_CMD,
                {"intensity": 0.7, "num_popups": 2, "popup_types": ["modal", "notification"]},
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
        if step_idx % 4 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": difficulty_level.intensity,
                    "num_components": difficulty_level.perturbation_count,
                    "component_types": difficulty_level.parameters.get("component_types", ["button"]),
                },
            )

        if "popup_types" in difficulty_level.parameters and step_idx % 6 == 0:
            return self._execute_controller_command(
                env,
                Constants.INJECT_POPUPS_CMD,
                {
                    "intensity": difficulty_level.intensity,
                    "num_popups": min(difficulty_level.perturbation_count, 2),
                    "popup_types": difficulty_level.parameters["popup_types"],
                },
            )

        return {"applied": False, "reason": "no_perturbation_needed"}


class ChromeNegativeScenario(BasePerturbationScenario):
    """Chrome negative perturbation scenario."""

    def __init__(self):
        super().__init__("chrome", "negative")

    def get_difficulty_levels(self) -> List[DifficultyLevel]:
        """Get difficulty levels for Chrome negative scenarios."""
        return [
            DifficultyLevel(
                level=1, intensity=0.2, perturbation_count=1, parameters={"misleading_elements": 1}
            ),
            DifficultyLevel(
                level=2, intensity=0.4, perturbation_count=2, parameters={"misleading_elements": 2}
            ),
            DifficultyLevel(
                level=3,
                intensity=0.6,
                perturbation_count=3,
                parameters={"misleading_elements": 3, "false_buttons": True},
            ),
            DifficultyLevel(
                level=4,
                intensity=0.8,
                perturbation_count=4,
                parameters={"misleading_elements": 4, "false_buttons": True, "fake_links": True},
            ),
        ]

    def apply_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Apply Chrome negative setup perturbations based on difficulty level."""
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
        """Apply Chrome negative runtime perturbations based on difficulty level."""
        method_name = f"_apply_runtime_level_{difficulty_level.level}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(env, difficulty_level, step_idx, obs)
        else:
            return self._apply_generic_runtime_perturbations(env, difficulty_level, step_idx, obs)

    def _apply_setup_level_1(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 1: Subtle misleading hints."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        misleading_hints = [" (Look for the highlighted element)", " (Check the top section first)"]

        if random.random() < 0.3:
            hint = random.choice(misleading_hints)
            perturbed_config["instruction"] = f"{original_instruction}{hint}"

        return perturbed_config

    def _apply_setup_level_2(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 2: More specific misleading instructions."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        misleading_additions = [
            " (Note: Look for the red button)",
            " (Note: Click the first option you see)",
            " (Note: Use the menu on the right)",
        ]

        if random.random() < 0.5:
            misleading_addition = random.choice(misleading_additions)
            perturbed_config["instruction"] = f"{original_instruction}{misleading_addition}"

        return perturbed_config

    def _apply_setup_level_3(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 3: Complex misleading instructions with false buttons."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        misleading_additions = [
            " (Note: Look for the red button)",
            " (Note: Click the first option you see)",
            " (Note: Use the menu on the right)",
            " (Note: Find the hidden button)",
            " (Note: The correct button is usually green)",
        ]

        misleading_addition = random.choice(misleading_additions)
        perturbed_config["instruction"] = f"{original_instruction}{misleading_addition}"
        perturbed_config["false_buttons_enabled"] = True

        return perturbed_config

    def _apply_setup_level_4(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Level 4: Maximum misleading complexity."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        misleading_additions = [
            " (Note: Look for the red button)",
            " (Note: Click the first option you see)",
            " (Note: Use the menu on the right)",
            " (Note: Find the hidden button)",
            " (Note: The correct button is usually green)",
            " (Note: Ignore any blue elements)",
            " (Note: The answer is always in the top-left)",
        ]

        # Add multiple misleading hints
        selected_hints = random.sample(misleading_additions, 2)
        misleading_text = "".join(selected_hints)
        perturbed_config["instruction"] = f"{original_instruction}{misleading_text}"
        perturbed_config["false_buttons_enabled"] = True
        perturbed_config["fake_links_enabled"] = True
        perturbed_config["misleading_labels"] = True

        return perturbed_config

    def _apply_generic_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Generic setup perturbations for unknown difficulty levels."""
        perturbed_config = task_config.copy()
        original_instruction = task_config.get("instruction", "")

        misleading_additions = [
            " (Note: Look for the red button)",
            " (Note: Click the first option you see)",
            " (Note: Use the menu on the right)",
            " (Note: Find the hidden button)",
        ]

        if difficulty_level.intensity > 0.5:
            misleading_addition = random.choice(misleading_additions)
            perturbed_config["instruction"] = f"{original_instruction}{misleading_addition}"

        return perturbed_config

    def _apply_runtime_level_1(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 1: Occasional subtle misleading elements."""
        if step_idx % 10 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": 0.2,
                    "num_components": 1,
                    "component_types": ["button"],
                    "misleading": True,
                    "subtle": True,
                },
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_2(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 2: Regular misleading elements."""
        if step_idx % 6 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": 0.4,
                    "num_components": 2,
                    "component_types": ["button", "link"],
                    "misleading": True,
                },
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_3(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 3: Frequent misleading elements with false buttons."""
        if step_idx % 4 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": 0.6,
                    "num_components": 3,
                    "component_types": ["button", "link"],
                    "misleading": True,
                    "false_buttons": True,
                },
            )
        return {"applied": False, "reason": "no_perturbation_needed"}

    def _apply_runtime_level_4(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Level 4: Maximum misleading complexity."""
        if step_idx % 3 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": 0.8,
                    "num_components": 4,
                    "component_types": ["button", "link"],
                    "misleading": True,
                    "false_buttons": True,
                    "fake_links": True,
                    "misleading_labels": True,
                },
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
        if step_idx % 3 == 0:
            return self._execute_controller_command(
                env,
                Constants.ADD_DISTRACTORS_CMD,
                {
                    "intensity": difficulty_level.intensity,
                    "num_components": difficulty_level.parameters.get("misleading_elements", 1),
                    "component_types": ["button", "link"],
                    "misleading": True,
                },
            )

        return {"applied": False, "reason": "no_perturbation_needed"}

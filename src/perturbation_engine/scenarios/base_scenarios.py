import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from perturbation_engine.data_types import Constants, DifficultyLevel
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv


class BasePerturbationScenario(ABC):
    """Base class for all perturbation scenarios."""

    def __init__(self, task_type: str, scenario_type: str):
        self.task_type = task_type
        self.scenario_type = scenario_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_difficulty_levels(self) -> List[DifficultyLevel]:
        """Get available difficulty levels for this scenario."""
        pass

    @abstractmethod
    def apply_setup_perturbations(
        self, task_config: Dict[str, Any], difficulty_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """Apply setup perturbations to task config."""
        pass

    @abstractmethod
    def apply_runtime_perturbations(
        self,
        env: PerturbationDesktopEnv,
        difficulty_level: DifficultyLevel,
        step_idx: int,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply runtime perturbations."""
        pass

    def _execute_controller_command(
        self, env: PerturbationDesktopEnv, command: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a controller command for perturbation using the controller directly."""
        try:
            controller = env.controller

            # Use the controller's built-in perturbation system
            from perturbation_engine.data_types import PerturbationPhase, PerturbationSpec, PerturbationType

            # Map command to perturbation type
            command_to_type = {
                Constants.UI_REORDERING_CMD: PerturbationType.UI_VISUAL,
                Constants.THEME_CHANGE_CMD: PerturbationType.UI_VISUAL,
                Constants.ADD_DISTRACTORS_CMD: PerturbationType.VISUAL_DISTRACTOR,
                Constants.MODIFY_RESOLUTION_CMD: PerturbationType.UI_VISUAL,
                Constants.INJECT_POPUPS_CMD: PerturbationType.VISUAL_DISTRACTOR,
            }

            perturbation_type = command_to_type.get(command)
            if not perturbation_type:
                return {"applied": False, "error": f"Unknown command: {command}"}

            # Create perturbation spec
            spec = PerturbationSpec(
                perturbation_type=perturbation_type,
                phase=PerturbationPhase.RUNTIME,
                perturbation_controller="gemini",
                parameters=parameters,
                trigger_function_name="immediate",
                trigger_parameters={},
            )

            # Use controller's apply_perturbation method directly
            return controller.apply_perturbation(spec, {})

        except Exception as e:
            self.logger.error(f"Error executing controller command {command}: {e}")
            return {"applied": False, "error": str(e)}

    def _rephrase_instruction(self, instruction: str, intensity: float = 0.5) -> str:
        """Rephrase instruction based on intensity."""
        rephrasing_map = {
            "click": "select",
            "find": "locate",
            "search": "look for",
            "open": "launch",
            "close": "shut",
            "navigate": "go to",
        }

        rephrased = instruction
        for original, replacement in rephrasing_map.items():
            if random.random() < intensity:
                rephrased = rephrased.replace(original, replacement)
        return rephrased

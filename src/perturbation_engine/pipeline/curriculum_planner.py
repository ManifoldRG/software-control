"""
CurriculumPlanner: Plans scenarios using CurriculumLLM
Clean interface for curriculum generation
"""

import logging
from typing import Any, Dict, List

from perturbation_engine.pipeline.clean_llm_services import CleanCurriculumLLM
from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.tools.autoglm_integration import AutoglmCurriculumGenerator


class CurriculumPlanner:
    """Plans scenarios using CurriculumLLM"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.curriculum_llm = CleanCurriculumLLM()
        self.autoglm_curriculum_generator = AutoglmCurriculumGenerator()

    def plan_curriculum(
        self,
        seed_trajectory: SeedTrajectory,
        app_states: List[Dict[str, Any]],
        curriculum_config: CurriculumConfig,
    ) -> List[ScenarioSpec]:
        """Generate curriculum of scenario specifications with retry and validation"""

        self.logger.info(f"Planning curriculum for task: {seed_trajectory.task_instruction}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use autoglm_v curriculum generator first
                autoglm_scenarios = self.autoglm_curriculum_generator.generate_scenario_specs(
                    seed_trajectory, app_states, curriculum_config
                )

                if autoglm_scenarios:
                    # Convert autoglm scenarios to ScenarioSpec objects
                    scenario_specs = self._convert_autoglm_scenarios(autoglm_scenarios)

                    # Validate and filter scenario specs
                    valid_specs = self.validate_scenario_specs(scenario_specs)

                    # Enhanced validation for diversity and quality
                    diverse_specs = self._ensure_scenario_diversity(valid_specs)

                    if (
                        len(diverse_specs) >= curriculum_config.scenario_count * 0.5
                    ):  # At least 50% from autoglm_v
                        self.logger.info(f"Generated {len(diverse_specs)} valid scenarios using autoglm_v")
                        return diverse_specs

                # Fallback to LLM if autoglm_v doesn't provide enough scenarios
                self.logger.info("Falling back to LLM for additional scenario generation")
                llm_scenarios = self.curriculum_llm.generate_scenario_specs(
                    seed_trajectory, app_states, curriculum_config
                )

                if llm_scenarios:
                    # Combine autoglm_v and LLM scenarios
                    all_scenarios = autoglm_scenarios + llm_scenarios
                    scenario_specs = self._convert_autoglm_scenarios(all_scenarios)

                    # Validate and filter scenario specs
                    valid_specs = self.validate_scenario_specs(scenario_specs)

                    # Enhanced validation for diversity and quality
                    diverse_specs = self._ensure_scenario_diversity(valid_specs)

                    self.logger.info(f"Generated {len(diverse_specs)} valid scenarios using autoglm_v + LLM")
                    return diverse_specs[: curriculum_config.scenario_count]

                self.logger.warning(f"Attempt {attempt + 1}: No scenario specs generated")
                if attempt < max_retries - 1:
                    continue
                return []

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} curriculum generation attempts failed")
                    return []

    def validate_scenario_specs(self, scenario_specs: List[ScenarioSpec]) -> List[ScenarioSpec]:
        """Validate scenario specifications"""
        valid_specs = []

        for spec in scenario_specs:
            try:
                # Basic validation
                if not spec.scenario_id:
                    self.logger.warning("Invalid scenario spec: missing scenario_id")
                    continue

                if not spec.target_app:
                    self.logger.warning("Invalid scenario spec: missing target_app")
                    continue

                if not spec.perturbation_trigger:
                    self.logger.warning("Invalid scenario spec: missing perturbation_trigger")
                    continue

                if not spec.available_perturbation_actions:
                    self.logger.warning("Invalid scenario spec: missing available_perturbation_actions")
                    continue

                valid_specs.append(spec)

            except Exception as e:
                self.logger.warning(f"Error validating scenario spec {spec.scenario_id}: {e}")
                continue

        self.logger.info(f"Validated {len(valid_specs)}/{len(scenario_specs)} scenario specifications")
        return valid_specs

    def _ensure_scenario_diversity(self, scenario_specs: List[ScenarioSpec]) -> List[ScenarioSpec]:
        """Ensure scenario diversity by filtering out similar scenarios based on actual commands"""
        diverse_specs = []
        seen_commands = set()
        seen_target_apps = set()

        for spec in scenario_specs:
            # Check for diversity in actual perturbation commands (not just types)
            command_signature = self._extract_command_signature(spec.available_perturbation_actions)
            if command_signature in seen_commands:
                self.logger.debug(f"Skipping duplicate command signature: {command_signature[:50]}...")
                continue

            # Check for diversity in target apps (allow some overlap but not too much)
            if spec.target_app in seen_target_apps and len(seen_target_apps) > 2:
                self.logger.debug(f"Skipping duplicate target app: {spec.target_app}")
                continue

            # Check for meaningful learning objectives
            if not spec.learning_objectives or len(spec.learning_objectives.strip()) < 20:
                self.logger.debug(
                    f"Skipping scenario with weak learning objectives: {spec.learning_objectives}"
                )
                continue

            # Check for realistic perturbation actions
            if (
                not spec.available_perturbation_actions
                or len(spec.available_perturbation_actions.strip()) < 30
            ):
                self.logger.debug(
                    f"Skipping scenario with weak perturbation actions: {spec.available_perturbation_actions}"
                )
                continue

            # Check for creativity - avoid scenarios that are too similar to examples
            if self._is_too_similar_to_examples(spec.available_perturbation_actions):
                self.logger.debug(
                    f"Skipping scenario too similar to examples: {spec.available_perturbation_actions[:50]}..."
                )
                continue

            diverse_specs.append(spec)
            seen_commands.add(command_signature)
            seen_target_apps.add(spec.target_app)

        self.logger.info(
            f"Ensured diversity: {len(diverse_specs)}/{len(scenario_specs)} scenarios are diverse"
        )
        return diverse_specs

    def _extract_command_signature(self, perturbation_actions: str) -> str:
        """Extract a signature from perturbation actions to detect duplicates"""
        if not perturbation_actions:
            return ""

        # Extract key command patterns
        import re

        # Remove variable values and focus on command structure
        signature = perturbation_actions.lower()

        # Replace random values with placeholders
        signature = re.sub(r"\d+", "N", signature)  # Numbers
        signature = re.sub(r'"[^"]*"', '"STRING"', signature)  # Strings
        signature = re.sub(r"'[^']*'", "'STRING'", signature)  # Single quotes
        signature = re.sub(r"\[[^\]]*\]", "[ARRAY]", signature)  # Arrays
        signature = re.sub(r"\([^)]*\)", "(PARAMS)", signature)  # Function calls

        # Remove whitespace and normalize
        signature = re.sub(r"\s+", " ", signature).strip()

        return signature

    def _is_too_similar_to_examples(self, perturbation_actions: str) -> bool:
        """Check if perturbation actions are too similar to common examples"""
        if not perturbation_actions:
            return True

        # Common example patterns to avoid
        example_patterns = [
            "gsettings set org.gnome.desktop.interface gtk-theme",
            "document.body.style.backgroundColor",
            "document.querySelectorAll('button')",
            "gsettings set org.gnome.desktop.interface icon-theme",
            "notify-send 'Background Process'",
            "mkdir -p /tmp/background_work",
        ]

        action_lower = perturbation_actions.lower()
        for pattern in example_patterns:
            if pattern in action_lower:
                # Check if it's just a simple copy
                if len(perturbation_actions.strip()) < 100:  # Too short to be creative
                    return True

        return False

    def _convert_autoglm_scenarios(self, autoglm_scenarios: List[Dict[str, Any]]) -> List[ScenarioSpec]:
        """Convert autoglm_v scenarios to ScenarioSpec objects"""
        scenario_specs = []

        for i, scenario_data in enumerate(autoglm_scenarios):
            try:
                # Parse perturbation types
                perturbation_types = []
                for pt_str in scenario_data.get("perturbation_types", []):
                    from perturbation_engine.pipeline.data_models import PerturbationType

                    mapped_type = PerturbationType.from_string(pt_str, default=PerturbationType.THEME)
                    perturbation_types.append(mapped_type)

                if not perturbation_types:
                    from perturbation_engine.pipeline.data_models import PerturbationType

                    perturbation_types.append(PerturbationType.THEME)

                scenario_spec = ScenarioSpec(
                    scenario_id=scenario_data.get("scenario_id", f"scenario_{i + 1}"),
                    target_app=scenario_data.get("target_app", "unknown"),
                    perturbation_trigger=scenario_data.get("perturbation_trigger", ""),
                    available_perturbation_actions=scenario_data.get("available_perturbation_actions", ""),
                    learning_objectives=scenario_data.get("learning_objectives", ""),
                    target_components=scenario_data.get("target_components", []),
                    perturbation_types=perturbation_types,
                )
                scenario_specs.append(scenario_spec)
            except Exception as e:
                self.logger.error(f"Error converting autoglm scenario: {e}")
                continue

        return scenario_specs

"""
CurriculumPlanner: Plans scenarios using CurriculumLLM
Clean interface for curriculum generation
"""

import logging
from typing import Any, Dict, List

from perturbation_engine.pipeline_refactored.data_models import (
    CurriculumConfig,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline_refactored.llm_services import CurriculumLLM


class CurriculumPlanner:
    """Plans scenarios using CurriculumLLM"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.curriculum_llm = CurriculumLLM()

    def plan_curriculum(
        self,
        seed_trajectory: SeedTrajectory,
        app_states: List[Dict[str, Any]],
        curriculum_config: CurriculumConfig,
    ) -> List[ScenarioSpec]:
        """Generate curriculum of scenario specifications"""

        self.logger.info(f"Planning curriculum for task: {seed_trajectory.task_instruction}")

        try:
            # Use LLM to generate scenario specs
            scenario_specs = self.curriculum_llm.generate_scenario_specs(
                seed_trajectory, app_states, curriculum_config
            )

            self.logger.info(f"Generated {len(scenario_specs)} scenario specifications")
            return scenario_specs

        except Exception as e:
            self.logger.error(f"Error planning curriculum: {e}")
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

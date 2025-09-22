import logging
from typing import Dict, List

from perturbation_engine.data_types import Constants, GenerationConfig, ScenarioSpec, SeedTrajectory
from perturbation_engine.scenarios.scenario_factory import ScenarioFactory, create_default_factory


class ScenarioGenerator:
    """Generates perturbation scenario specifications from seed trajectories."""

    def __init__(self, scenario_factory: ScenarioFactory = None):
        self._logger = logging.getLogger(__name__)
        self._scenario_factory = scenario_factory or create_default_factory()

    def generate_scenarios(
        self,
        seed_trajectories: List[SeedTrajectory],
        generation_config: GenerationConfig,
        result_base_dir: str = "./perturbation_results",
    ) -> List[ScenarioSpec]:
        """
        Generate a curriculum of perturbation scenarios from given seed trajectories.

        Design: Loop through task types, then scenario types, then difficulty levels.
        For each task type + scenario type combination, use the factory to create scenarios.

        Args:
            seed_trajectories: List of seed trajectories with different task types
            generation_config: Generation configuration with per-task-type settings
            result_base_dir: Base directory to save perturbation results

        Returns:
            List of scenario specifications
        """
        scenario_specs = []

        # Group seed trajectories by task type
        task_type_groups = self._group_trajectories_by_task_type(seed_trajectories)

        for task_type, trajectories in task_type_groups.items():
            self._logger.info(f"Generating scenarios for task type: {task_type}")

            # Get scenario types
            scenario_types = Constants.SCENARIO_TYPES

            for scenario_type in scenario_types:
                # Get count for this scenario type
                num_scenarios = getattr(generation_config, f"num_{scenario_type}_scenarios")
                if num_scenarios == 0:
                    self._logger.info(f"  Skipping {scenario_type} scenarios for {task_type} (count=0)")
                    continue

                self._logger.info(f"  Generating {num_scenarios} {scenario_type} scenarios for {task_type}")

                # Get difficulty levels for this scenario
                all_difficulty_levels = self._scenario_factory.get_difficulty_levels(task_type, scenario_type)
                if generation_config.num_difficulty_levels < len(all_difficulty_levels):
                    difficulty_levels = all_difficulty_levels[: generation_config.num_difficulty_levels]
                else:
                    self._logger.warning(
                        f"Only {len(all_difficulty_levels)} difficulty levels available for {task_type} {scenario_type}, using all levels"
                    )
                    difficulty_levels = all_difficulty_levels

                # Generate scenarios for each trajectory and difficulty level
                for seed_idx, seed_trajectory in enumerate(trajectories):
                    for difficulty_level in difficulty_levels:
                        for scenario_count in range(num_scenarios):
                            scenario_spec = self._create_scenario_spec(
                                seed_trajectory,
                                seed_idx,
                                task_type,
                                scenario_type,
                                difficulty_level,
                                result_base_dir,
                                scenario_count,
                            )
                            scenario_specs.append(scenario_spec)

        self._logger.info(f"Generated {len(scenario_specs)} total scenario specifications")
        return scenario_specs

    def _group_trajectories_by_task_type(
        self, seed_trajectories: List[SeedTrajectory]
    ) -> Dict[str, List[SeedTrajectory]]:
        """Group seed trajectories by task type."""
        groups = {}
        for trajectory in seed_trajectories:
            task_type = trajectory.task_type
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(trajectory)
        return groups

    def _create_scenario_spec(
        self,
        seed_trajectory: SeedTrajectory,
        seed_idx: int,
        task_type: str,
        scenario_type: str,
        difficulty_level,
        result_base_dir: str,
        scenario_count: int,
    ) -> ScenarioSpec:
        """Create a scenario specification."""
        scenario_id = (
            f"seed_{seed_idx}_{task_type}_{scenario_type}_level_{difficulty_level.level}_gen_{scenario_count}"
        )

        # Get scenario class name without creating instance
        scenario_class_name = self._get_scenario_class_name(task_type, scenario_type)

        return ScenarioSpec(
            scenario_id=scenario_id,
            task_id=seed_trajectory.config["id"],
            task_type=task_type,
            scenario_type=scenario_type,
            difficulty_level=difficulty_level.level,
            task_config=seed_trajectory.config,
            trajectory_file_path=seed_trajectory.gt_actions_file_path,
            perturbation_scenario_class=scenario_class_name,
            intensity=difficulty_level.intensity,
            perturbation_count=difficulty_level.perturbation_count,
            parameters=difficulty_level.parameters,
            result_dir=f"{result_base_dir}/{scenario_id}",
            seed_index=seed_idx,
            scenario_count=scenario_count,
        )

    def _get_scenario_class_name(self, task_type: str, scenario_type: str) -> str:
        """Get scenario class name without creating instance."""
        # Get the class from registry without instantiating
        scenario_class = self._scenario_factory._registry.get((task_type, scenario_type))
        if not scenario_class:
            # Fallback to Chrome scenario
            scenario_class = self._scenario_factory._registry.get(("chrome", scenario_type))
            if not scenario_class:
                return "ChromeInvarianceScenario"  # Default fallback
        return scenario_class.__name__

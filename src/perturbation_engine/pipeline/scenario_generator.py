import logging
from typing import Any, Dict, List

from perturbation_engine.data_types import ScenarioSpec, SeedTrajectory
from perturbation_engine.scenarios.scenario_factory import PerturbationScenarioFactory
from perturbation_engine.scenarios.scenarios import ChromePerturbationScenario


class ScenarioGenerator:
    """Manages perturbation scenarios and parameter generation

    based on what apps the seed scenario uses, select the corresponding perturbation scenario
    and generate scenario specifications for each seed scenario
    - generate a curriculum of perturbation scenario specifications for given seed task trajectories


    - takes seed scenarios and generates scenario specifications, each scenario specification contains a task config, perturbations, and a result directory
    - for each seed scenario, a set of scenarios specs are chosen from pre-defined scenario specifications to guide the perturbed trajectory generation process
    """

    def __init__(self):
        """Initialize scenario manager."""
        self._logger = logging.getLogger(__name__)
        self.scenario_factory = PerturbationScenarioFactory()

        # Register available scenarios
        self.scenario_factory.register_scenario("chrome", ChromePerturbationScenario)
        # TODO: Register other scenarios as they are implemented

    def generate_scenarios(
        # default to generating 100 trajectories per seed trajectory
        self,
        seed_trajectories: List[SeedTrajectory],
        num_generations_per_seed: int = 100,
        result_base_dir: str = "./perturbation_results",
    ) -> List[ScenarioSpec]:
        """
        Generate a curriculum of perturbation scenarios from given seed scenarios

        Loop through the seed trajectories
        Generate a perturbation scenario for each trajectory based on the task_type
        and task instruction type (e.g., information retrieval, data entry)

        A perturbation scenario contains:
        - perturbation scenario function name
        - parameter configs for the params of the perturbations in the perturbation scenario function

        - Q: Is it necessary to break down the task types further into instruction types?
        - Q: If so, how to categorize the task instruction types?

        Args:
            - seed_trajectories: list of seed trajectories with the 10 OSWorld task types
            - num_generations_per_seed: number of trajectories to generate for each seed scenario

        Return:
            - list of curriculum of perturbation scenarios for each task type
            - each perturbation scenario:
                - perturbation scenario function name
                    - the function takes the perturbation parameters
                    - the function returns a list of concrete perturbation specs
                - perturbation scenario parameters
        """
        scenario_specs = []
        for seed_idx, seed_trajectory in enumerate(seed_trajectories):
            for traj_idx in range(num_generations_per_seed):
                scenario_spec = self._create_scenario_spec_from_seed_trajectory(
                    seed_trajectory, seed_idx, traj_idx, result_base_dir
                )
                scenario_specs.append(scenario_spec)

        return scenario_specs

    def _create_scenario_spec_from_seed_trajectory(
        self, seed_trajectory: SeedTrajectory, seed_idx: int, traj_idx: int, result_base_dir: str
    ) -> ScenarioSpec:
        """Create a scenario specification from a seed trajectory."""
        scenario_id = f"seed_{seed_idx}_gen_{traj_idx}"
        task_id = getattr(seed_trajectory, "task_id", f"seed_{seed_idx}_gen_{traj_idx}")

        # Determine perturbation scenario class based on task type
        perturbation_scenario_class = self._get_scenario_class_for_task_type(seed_trajectory)

        # Generate scenario-specific parameters
        perturbation_parameters = self._generate_scenario_parameters(seed_trajectory)

        # Create result directory
        result_dir = f"{result_base_dir}/{scenario_id}"

        return ScenarioSpec(
            scenario_id=scenario_id,
            task_id=task_id,
            task_config=seed_trajectory.task_config,
            trajectory_file_path=seed_trajectory.gt_actions_file_path,
            perturbation_scenario_class=perturbation_scenario_class,
            perturbation_parameters=perturbation_parameters,
            result_dir=result_dir,
            metadata={
                "seed_index": seed_idx,
                "trajectory_index": traj_idx,
                "source": "scenario_generator",
                "task_type": getattr(seed_trajectory, "task_type", "chrome"),
            },
        )

    def _get_scenario_class_for_task_type(self, task_type: str) -> str:
        """Get scenario class name for task type."""
        # Map task types to scenario classes
        scenario_mapping = {
            "chrome": "chrome",
            "gimp": "chrome",  # Default to chrome for now
            "libreoffice_calc": "chrome",
            "libreoffice_impress": "chrome",
            "libreoffice_writer": "chrome",
            "multi_apps": "chrome",
            "os": "chrome",
            "thunderbird": "chrome",
            "vlc": "chrome",
            "vs_code": "chrome",
        }
        return scenario_mapping.get(task_type, "chrome")

    def _generate_scenario_parameters(self, seed_trajectory: SeedTrajectory) -> Dict[str, Any]:
        """Generate parameters for the perturbation scenario."""
        import random

        # Generate Chrome-specific parameters
        return {
            "num_components": random.randint(3, 8),
            "injection_delay": random.uniform(0.5, 2.0),
            "theme_change": random.choice([True, False]),
        }

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of scenario generation capabilities"""
        return {
            "available_scenarios": self.scenario_factory.get_available_scenarios(),
            "available_controllers": ["gemini"],
        }

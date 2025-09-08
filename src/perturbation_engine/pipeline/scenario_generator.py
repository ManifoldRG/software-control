import logging
from typing import Any, Dict, List

from perturbation_engine.data_types import PerturbationPhase, PerturbationSpec, PerturbationType, ScenarioSpec


class ScenarioGenerator:
    """Manages perturbation scenarios and parameter generation"""

    def __init__(self, seed_task_config: Dict[str, Any] = None, perturbation_config: Dict[str, Any] = None):
        """Initialize scenario manager.

        Args:
            seed_task_config: Base task configuration
            perturbation_config: Perturbation system configuration
        """
        self._logger = logging.getLogger(__name__)
        self.seed_task_config = seed_task_config or {}
        self.perturbation_config = perturbation_config or {}

    def generate_scenarios(
        self, seed_scenarios: List[Dict[str, Any]], num_trajectories_per_seed: int
    ) -> List[ScenarioSpec]:
        """Generate multiple perturbation scenarios from seed scenarios"""
        scenario_specs = []

        for seed_idx, seed_scenario in enumerate(seed_scenarios):
            for traj_idx in range(num_trajectories_per_seed):
                scenario_spec = self._create_scenario_from_seed(seed_scenario, seed_idx, traj_idx)
                scenario_specs.append(scenario_spec)

        return scenario_specs

    def _create_scenario_from_seed(
        self, seed_scenario: Dict[str, Any], seed_idx: int, traj_idx: int
    ) -> ScenarioSpec:
        """Create a scenario specification from a seed scenario"""
        scenario_id = f"scenario_{seed_idx}_{traj_idx}"
        task_id = f"task_{seed_idx}_{traj_idx}"

        # Create perturbations for this scenario
        perturbations = self._generate_perturbations_for_scenario(seed_scenario)

        # Create result directory
        result_dir = f"./results/{scenario_id}"

        return ScenarioSpec(
            scenario_id=scenario_id,
            task_id=task_id,
            task_config=seed_scenario,
            trajectory_file_path=seed_scenario.get("trajectory", ""),
            perturbations=perturbations,
            result_dir=result_dir,
            metadata={"seed_index": seed_idx, "trajectory_index": traj_idx, "source": "scenario_generator"},
        )

    def _generate_perturbations_for_scenario(self, seed_scenario: Dict[str, Any]) -> List[PerturbationSpec]:
        """Generate perturbations for a specific scenario"""
        perturbations = []

        # Add UI visual perturbation
        ui_perturbation = PerturbationSpec(
            perturbation_type=PerturbationType.UI_VISUAL,
            phase=PerturbationPhase.RUNTIME,
            perturbation_controller="gemini",
            parameters={"action": "ui_injection", "num_components": 5},
            trigger_function_name="step_range",
            trigger_parameters={"start": 2, "end": 8},
            validation_function_name="element_created",
            validation_parameters={"selector": ".injected-element"},
            name="ui_injection_early",
            description="Inject UI elements early in the task",
        )
        perturbations.append(ui_perturbation)

        # Add theme change perturbation
        theme_perturbation = PerturbationSpec(
            perturbation_type=PerturbationType.UI_VISUAL,
            phase=PerturbationPhase.RUNTIME,
            perturbation_controller="gemini",
            parameters={"action": "theme_change", "theme": "dark"},
            trigger_function_name="step_range",
            trigger_parameters={"start": 5, "end": 10},
            name="theme_change",
            description="Change to dark theme mid-task",
        )
        perturbations.append(theme_perturbation)

        return perturbations

    def load_seed_scenarios(self, config_base_dir: str) -> List[Dict[str, Any]]:
        """Load seed scenarios from task configs and existing trajectories"""
        seed_scenarios = []

        # Example: Load from OSWorld evaluation examples
        # This is a placeholder implementation
        example_scenario = {
            "id": "0d8b7de3-e8de-4d86-b9fd-dd2dce58a217",
            "snapshot": "chrome",
            "instruction": "Browse the natural products database.",
            "source": "Mind2Web",
            "config": [
                {
                    "type": "launch",
                    "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]},
                },
                {
                    "type": "launch",
                    "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]},
                },
                {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://drugs.com"]}},
                {"type": "activate_window", "parameters": {"window_name": "Google Chrome"}},
            ],
            "trajectory": "external_data/osworld-verified/jedi-7b-4o-15steps/chrome/0d8b7de3-e8de-4d86-b9fd-dd2dce58a217",
            "related_apps": ["chrome"],
            "evaluator": {
                "func": ["is_expected_active_tab"],
                "result": [{"type": "active_url_from_accessTree"}],
                "expected": [{"type": "rule", "rules": {"type": "url", "url": "https://www.drugs.com/npc/"}}],
            },
        }

        seed_scenarios.append(example_scenario)
        self._logger.info(f"Loaded {len(seed_scenarios)} seed scenarios")

        return seed_scenarios

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of scenario generation capabilities"""
        return {
            "perturbation_types": [pt.value for pt in PerturbationType],
            "perturbation_phases": [pp.value for pp in PerturbationPhase],
            "available_controllers": ["gemini"],
            "seed_scenarios_loaded": len(self.seed_task_config) if self.seed_task_config else 0,
        }

"""
Unified orchestrator for trajectory generation with both static and curriculum-based approaches
"""

import logging
from typing import List, Optional

from perturbation_engine.curriculum.curriculum_config import CurriculumConfig
from perturbation_engine.curriculum.curriculum_generator import CurriculumGenerator
from perturbation_engine.curriculum.curriculum_orchestrator import CurriculumOrchestrator
from perturbation_engine.data_types import ExecutionConfig, GenerationConfig, GenerationResult, SeedTrajectory
from perturbation_engine.llm_orchestra import LLMOrchestra
from perturbation_engine.pipeline.scenario_generator import ScenarioGenerator
from perturbation_engine.pipeline.shared_execution_engine import SharedExecutionEngine


class UnifiedOrchestrator:
    """Unified orchestrator supporting both static and curriculum-based trajectory generation"""

    def __init__(self, execution_config: ExecutionConfig = None):
        self.execution_config = execution_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize shared components
        self.llm_orchestra = LLMOrchestra()
        self.scenario_generator = ScenarioGenerator(llm_orchestra=self.llm_orchestra)
        self.curriculum_generator = CurriculumGenerator(llm_orchestra=self.llm_orchestra)
        self.curriculum_orchestrator = CurriculumOrchestrator(execution_config)

    def generate_static_trajectories(
        self,
        seed_trajectories: List[SeedTrajectory],
        generation_config: GenerationConfig,
        num_parallel_vms: int = 1,
        result_base_dir: str = "./perturbation_results",
    ) -> List[GenerationResult]:
        """Generate trajectories using static scenario generation"""

        self.logger.info(f"Generating {len(seed_trajectories)} static trajectories")

        # Generate scenarios using static approach
        scenario_specs = self.scenario_generator.generate_scenarios(
            seed_trajectories, generation_config, result_base_dir, env=None
        )

        # Execute scenarios
        return self._execute_scenarios(scenario_specs, num_parallel_vms)

    def generate_curriculum_trajectories(
        self,
        seed_trajectory: SeedTrajectory,
        curriculum_config: CurriculumConfig,
        num_parallel_vms: int = 1,
        result_base_dir: str = "./curriculum_results",
        env=None,
    ) -> List[GenerationResult]:
        """Generate trajectories using curriculum-based approach"""

        # Use curriculum orchestrator with shared LLM orchestra
        return self.curriculum_orchestrator.generate_curriculum_trajectories(
            seed_trajectory, curriculum_config, num_parallel_vms, result_base_dir, self.llm_orchestra, env
        )

    def generate_hybrid_trajectories(
        self,
        seed_trajectories: List[SeedTrajectory],
        generation_config: GenerationConfig,
        curriculum_config: Optional[CurriculumConfig] = None,
        num_parallel_vms: int = 1,
        result_base_dir: str = "./hybrid_results",
    ) -> List[GenerationResult]:
        """Generate trajectories using both static and curriculum approaches"""

        all_results = []

        # Generate static trajectories
        static_results = self.generate_static_trajectories(
            seed_trajectories, generation_config, num_parallel_vms, f"{result_base_dir}/static"
        )
        all_results.extend(static_results)

        # Generate curriculum trajectories for each seed
        if curriculum_config:
            for seed_trajectory in seed_trajectories:
                curriculum_results = self.generate_curriculum_trajectories(
                    seed_trajectory, curriculum_config, num_parallel_vms, f"{result_base_dir}/curriculum"
                )
                all_results.extend(curriculum_results)

        self.logger.info(f"Generated {len(all_results)} total hybrid trajectories")
        return all_results

    def _execute_scenarios(self, scenario_specs: List, num_parallel_vms: int) -> List[GenerationResult]:
        """Execute scenarios using shared execution engine"""

        execution_engine = SharedExecutionEngine(self.execution_config)
        return execution_engine.execute_scenarios_parallel(scenario_specs, num_parallel_vms, "UnifiedProcess")

    def load_seed_trajectories(self, config_base_dir: str, trajectory_base_dir: str) -> List[SeedTrajectory]:
        """Load seed trajectories from task configs and existing trajectories"""
        import json
        import os
        from pathlib import Path

        seed_trajectories = []
        config_path = Path(config_base_dir)

        # Find all task config JSON files in the evaluation examples
        if config_path.name == "evaluation_examples":
            # Look in examples subdirectories
            examples_dir = config_path / "examples"
        else:
            examples_dir = config_path

        if not examples_dir.exists():
            raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

        # Get all app directories (chrome, gimp, etc.)
        app_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]

        for app_dir in app_dirs:
            app_name = app_dir.name
            self.logger.info(f"Loading trajectories for app: {app_name}")

            # Find all JSON config files in this app directory
            config_files = list(app_dir.glob("*.json"))

            for config_file in config_files:
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        task_config = json.load(f)

                    # Verify required fields
                    if not all(
                        field in task_config for field in ["id", "instruction", "config", "evaluator"]
                    ):
                        self.logger.warning(f"Skipping {config_file.name} - missing required fields")
                        continue

                    # Construct trajectory path based on the task ID
                    task_id = task_config["id"]
                    task_trajectory_dir = os.path.join(trajectory_base_dir, app_name, task_id)

                    # Verify trajectory directory exists
                    if not os.path.exists(task_trajectory_dir):
                        self.logger.warning(f"Trajectory directory not found: {task_trajectory_dir}")
                        continue

                    # Verify traj.jsonl exists
                    traj_file = os.path.join(task_trajectory_dir, "traj.jsonl")
                    if not os.path.exists(traj_file):
                        self.logger.warning(f"Trajectory file not found: {traj_file}")
                        continue

                    # Create seed trajectory with trajectory path
                    seed_trajectory = SeedTrajectory(
                        task_type=task_config.get("snapshot", "chrome"),
                        task_instruction=task_config["instruction"],
                        config=task_config,
                        gt_actions_file_path=traj_file,
                        gt_actions=None,
                    )

                    seed_trajectories.append(seed_trajectory)
                    self.logger.debug(f"Loaded trajectory: {task_id}")

                except (json.JSONDecodeError, KeyError, OSError) as e:
                    self.logger.error(f"Error loading {config_file.name}: {e}")
                    continue

        self.logger.info(
            f"Loaded {len(seed_trajectories)} seed trajectories from {len(app_dirs)} app directories"
        )
        return seed_trajectories


def create_unified_orchestrator(execution_config: ExecutionConfig = None) -> UnifiedOrchestrator:
    """Create a unified orchestrator with default configuration"""
    return UnifiedOrchestrator(execution_config)

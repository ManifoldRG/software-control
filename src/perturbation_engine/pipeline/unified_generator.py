"""
UnifiedGenerator: Main orchestrator
Clean interface for the entire pipeline
"""

import logging
import time
from typing import List

from perturbation_engine.pipeline.curriculum_planner import CurriculumPlanner
from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    ExecutionConfig,
    GeneratedTrajectory,
    SeedTrajectory,
)
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.quality_evaluator import QualityEvaluator
from perturbation_engine.pipeline.shared_execution_engine import SharedExecutionEngine
from perturbation_engine.pipeline.trajectory_generator import TrajectoryGenerator


class UnifiedGenerator:
    """Main orchestrator for the perturbation pipeline"""

    def __init__(self, execution_config: ExecutionConfig):
        self.execution_config = execution_config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.curriculum_planner = CurriculumPlanner()
        self.trajectory_generator = TrajectoryGenerator()
        self.shared_execution_engine = SharedExecutionEngine(execution_config)
        self.quality_evaluator = QualityEvaluator()

    def generate_trajectories(
        self, seed_trajectory: SeedTrajectory, curriculum_config: CurriculumConfig
    ) -> List[GeneratedTrajectory]:
        """Generate trajectories using the complete pipeline"""

        self.logger.info(f"Starting trajectory generation for {seed_trajectory.task_id}")

        try:
            # Step 1: Extract environment state
            # Initialize environment and extract first observation
            env = PerturbationDesktopEnv(
                path_to_vm=self.execution_config.path_to_vm,
                action_space=self.execution_config.action_space,
                provider_name=self.execution_config.provider_name,
                region=self.execution_config.region,
                snapshot_name=self.execution_config.snapshot_name,
                screen_size=self.execution_config.screen_size,
                headless=self.execution_config.headless,
                os_type=self.execution_config.os_type,
                require_a11y_tree=self.execution_config.require_a11y_tree,
                require_terminal=self.execution_config.require_terminal,
                enable_proxy=self.execution_config.enable_proxy,
                client_password=self.execution_config.client_password,
                cache_dir=self.execution_config.cache_dir,
                chromium_port=self.execution_config.chromium_port,
            )

            env.reset(task_config=seed_trajectory.config)
            app_states = env.get_app_states_from_accessibility_tree()
            env.close()
            time.sleep(5)

            if app_states == []:
                self.logger.error("No app states found")
                return []

            # Step 2: Generate curriculum of scenario specs
            scenario_specs = self.curriculum_planner.plan_curriculum(
                seed_trajectory, app_states, curriculum_config
            )

            if not scenario_specs:
                self.logger.error("No scenario specs generated")
                return []

            # Wait for main process VM to fully clean up before starting parallel processes
            self.logger.info("Waiting for main process VM cleanup to complete")
            time.sleep(10)  # Give extra time for VM cleanup

            # Step 3: Execute scenarios in parallel
            generated_trajectories = self.shared_execution_engine.execute_scenarios_parallel(
                seed_trajectory, scenario_specs, curriculum_config.num_parallel_vms
            )

            # Step 4: Evaluate quality
            for i, trajectory in enumerate(generated_trajectories):
                if i < len(scenario_specs):
                    quality_score = self.quality_evaluator.evaluate_trajectory_quality(
                        trajectory, scenario_specs[i]
                    )
                    # Update trajectory with quality score
                    trajectory = trajectory._replace(quality_score=quality_score)
                    generated_trajectories[i] = trajectory

            self.logger.info(f"Generated {len(generated_trajectories)} trajectories")
            return generated_trajectories

        except Exception as e:
            self.logger.error(f"Error generating trajectories: {e}")
            return []

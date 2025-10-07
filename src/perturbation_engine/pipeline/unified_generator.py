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
from perturbation_engine.utils.memory_utils import force_garbage_collection, log_memory_usage


class UnifiedGenerator:
    """Main orchestrator for the perturbation pipeline"""

    def __init__(self, execution_config: ExecutionConfig, result_base_dir: str = "/opt/manifold/results"):
        self.execution_config = execution_config
        self.result_base_dir = result_base_dir
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.curriculum_planner = CurriculumPlanner()
        self.trajectory_generator = TrajectoryGenerator(result_base_dir)
        self.shared_execution_engine = SharedExecutionEngine(execution_config, result_base_dir)
        self.quality_evaluator = QualityEvaluator()

    def generate_trajectories(
        self, seed_trajectory: SeedTrajectory, curriculum_config: CurriculumConfig
    ) -> List[GeneratedTrajectory]:
        """Generate trajectories using the complete pipeline"""

        self.logger.info(f"Starting trajectory generation for {seed_trajectory.task_id}")
        log_memory_usage("Start of trajectory generation", self.logger)

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

            # Register environment for signal handling
            try:
                from perturbation_engine.pipeline.generate_trajectories import active_environments

                active_environments.append(env)
            except ImportError:
                pass  # Not running from main script

            env.reset(task_config=seed_trajectory.config)

            # Extract enhanced app states using autoglm_v processing
            app_states = env.controller.get_app_states(use_autoglm_enhancement=True)

            env.close()

            # Remove from active environments after closing
            try:
                from perturbation_engine.pipeline.generate_trajectories import active_environments

                if env in active_environments:
                    active_environments.remove(env)
            except ImportError:
                pass  # Not running from main script

            time.sleep(5)

            # Force garbage collection after environment cleanup
            force_garbage_collection(self.logger)
            log_memory_usage("After environment cleanup", self.logger)

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
            time.sleep(5)  # Give extra time for VM cleanup

            # Step 3: Execute scenarios in parallel
            generated_trajectories = self.shared_execution_engine.execute_scenarios_parallel(
                seed_trajectory, scenario_specs, curriculum_config.num_parallel_vms
            )

            # Step 4: Evaluate quality and filter trajectories
            valid_trajectories = []
            quality_stats = {
                "total_trajectories": len(generated_trajectories),
                "high_quality_trajectories": 0,
                "low_perturbation_success": 0,
                "failed_trajectories": 0,
            }

            for i, trajectory in enumerate(generated_trajectories):
                if i < len(scenario_specs):
                    # Evaluate quality
                    quality_score = self.quality_evaluator.evaluate_trajectory_quality(
                        trajectory, scenario_specs[i]
                    )
                    trajectory.quality_score = quality_score

                    # Check perturbation success rate from log
                    perturbation_success_rate = 0.0
                    if trajectory.perturbation_log:
                        summary = trajectory.perturbation_log[-1].get("summary", {})
                        perturbation_success_rate = summary.get("perturbation_success_rate", 0.0)

                    # Filter trajectories based on quality and perturbation success
                    if (
                        quality_score >= 0.6 and perturbation_success_rate >= 0.3
                    ):  # 60% quality, 30% perturbation success
                        valid_trajectories.append(trajectory)
                        quality_stats["high_quality_trajectories"] += 1
                    elif perturbation_success_rate < 0.3:
                        quality_stats["low_perturbation_success"] += 1
                        self.logger.warning(
                            f"Trajectory {trajectory.trajectory_id} dropped: low perturbation success rate {perturbation_success_rate:.2%}"
                        )
                    else:
                        quality_stats["failed_trajectories"] += 1
                        self.logger.warning(
                            f"Trajectory {trajectory.trajectory_id} dropped: low quality score {quality_score:.2f}"
                        )

            # Log comprehensive statistics
            self.logger.info("Trajectory Quality Analysis:")
            self.logger.info(f"  Total generated: {quality_stats['total_trajectories']}")
            self.logger.info(f"  High quality (kept): {quality_stats['high_quality_trajectories']}")
            self.logger.info(
                f"  Low perturbation success (dropped): {quality_stats['low_perturbation_success']}"
            )
            self.logger.info(f"  Failed quality check (dropped): {quality_stats['failed_trajectories']}")

            generated_trajectories = valid_trajectories
            self.logger.info(f"Final valid trajectories: {len(generated_trajectories)}")

            # Final cleanup and memory check
            force_garbage_collection(self.logger)
            log_memory_usage("End of trajectory generation", self.logger)
            return generated_trajectories

        except Exception as e:
            self.logger.error(f"Error generating trajectories: {e}")
            return []

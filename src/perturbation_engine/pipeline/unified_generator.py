"""
UnifiedGenerator: Main orchestrator
Clean interface for the entire pipeline
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, List

from perturbation_engine.configure_logging import set_run_context
from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    ExecutionConfig,
    GeneratedTrajectory,
    SeedTrajectory,
)
from perturbation_engine.pipeline.llm_services import CurriculumGenerator
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.phase_data_manager import PhaseDataManager
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

        # Generate a unique run ID for this execution
        self.run_id = self._generate_run_id()

        # Initialize components
        self.curriculum_generator = CurriculumGenerator()
        self.trajectory_generator = TrajectoryGenerator(result_base_dir, self.run_id)
        self.shared_execution_engine = SharedExecutionEngine(execution_config, result_base_dir)
        self.quality_evaluator = QualityEvaluator()

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this execution"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def generate_trajectories(
        self, seed_trajectory: SeedTrajectory, curriculum_config: CurriculumConfig
    ) -> List[GeneratedTrajectory]:
        """Generate trajectories using the complete pipeline"""

        self.logger.info(f"Starting trajectory generation for {seed_trajectory.task_id}")
        log_memory_usage("Start of trajectory generation", self.logger)

        # Set run context for logging
        set_run_context(seed_trajectory.task_id, self.run_id)

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
                pass

            env.reset(task_config=seed_trajectory.config)

            window_states = env.controller.get_window_states()

            # Save window states using phase data manager
            phase_data_manager = PhaseDataManager(trajectory_id=seed_trajectory.task_id, run_id=self.run_id)
            phase_data_manager.save_window_states(step_idx=0, phase="initial", window_states=window_states)

            env.close()

            # Remove from active environments after closing
            try:
                from perturbation_engine.pipeline.generate_trajectories import active_environments

                if env in active_environments:
                    active_environments.remove(env)
            except ImportError:
                pass  # Not running from main script

            force_garbage_collection(self.logger)
            log_memory_usage("After environment cleanup", self.logger)

            if window_states == []:
                self.logger.error("No window states found")
                return []

            # Step 2: Generate curriculum of scenario specs
            scenario_specs = self.curriculum_generator.generate_scenario_specs(
                seed_trajectory, window_states, curriculum_config
            )

            if not scenario_specs:
                self.logger.error("No scenario specs generated")
                return []

            self.logger.info(f"Scenario specs: {scenario_specs}")

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

            # Save run summary
            self._save_run_summary(seed_trajectory, generated_trajectories, quality_stats)

            return generated_trajectories

        except Exception as e:
            self.logger.error(f"Error generating trajectories: {e}")
            return []

    def _save_run_summary(
        self,
        seed_trajectory: SeedTrajectory,
        generated_trajectories: List[GeneratedTrajectory],
        quality_stats: Dict[str, Any],
    ):
        """Save a comprehensive summary of this run"""
        try:
            # Create seed directory if it doesn't exist
            seed_dir = os.path.join(self.result_base_dir, seed_trajectory.task_id)
            os.makedirs(seed_dir, exist_ok=True)

            # Create run directory
            run_dir = os.path.join(seed_dir, self.run_id)
            os.makedirs(run_dir, exist_ok=True)

            # Prepare summary data
            summary_data = {
                "run_info": {
                    "run_id": self.run_id,
                    "seed_trajectory_id": seed_trajectory.task_id,
                    "task_instruction": seed_trajectory.task_instruction,
                    "task_type": seed_trajectory.task_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "execution_config": {
                    "provider_name": self.execution_config.provider_name,
                    "region": self.execution_config.region,
                    "snapshot_name": self.execution_config.snapshot_name,
                    "screen_size": self.execution_config.screen_size,
                    "headless": self.execution_config.headless,
                    "os_type": self.execution_config.os_type,
                },
                "results": {
                    "total_trajectories_generated": quality_stats["total_trajectories"],
                    "high_quality_trajectories": quality_stats["high_quality_trajectories"],
                    "low_perturbation_success": quality_stats["low_perturbation_success"],
                    "failed_trajectories": quality_stats["failed_trajectories"],
                    "final_valid_trajectories": len(generated_trajectories),
                },
                "trajectory_details": [
                    {
                        "trajectory_id": traj.trajectory_id,
                        "scenario_spec_id": traj.scenario_spec_id,
                        "success": traj.success,
                        "quality_score": traj.quality_score,
                        "generation_time": traj.generation_time,
                        "total_perturbation_attempts": traj.total_perturbation_attempts,
                        "total_perturbation_successes": traj.total_perturbation_successes,
                        "perturbation_success_rate": (
                            traj.total_perturbation_successes / traj.total_perturbation_attempts
                            if traj.total_perturbation_attempts > 0
                            else 0.0
                        ),
                        "trajectory_file_path": traj.trajectory_file_path,
                    }
                    for traj in generated_trajectories
                ],
                "folder_structure": {
                    "debug_folder": f"./debug/{seed_trajectory.task_id}/{self.run_id}",
                    "results_folder": f"{self.result_base_dir}/{seed_trajectory.task_id}/{self.run_id}",
                    "phases_subfolder": "phases",
                    "visualizations_subfolder": "visualizations",
                    "window_states_subfolder": "window_states",
                    "summaries_subfolder": "summaries",
                },
            }

            # Save run summary
            summary_path = os.path.join(run_dir, "run_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)

            self.logger.info(f"Run summary saved: {summary_path}")

            # Also save a seed-level summary for easy comparison across runs
            seed_summary_path = os.path.join(seed_dir, "seed_summary.json")
            seed_summary = {
                "seed_trajectory_id": seed_trajectory.task_id,
                "task_instruction": seed_trajectory.task_instruction,
                "task_type": seed_trajectory.task_type,
                "latest_run_id": self.run_id,
                "latest_run_timestamp": datetime.datetime.now().isoformat(),
                "total_runs": self._count_runs_for_seed(seed_dir),
                "latest_run_results": summary_data["results"],
            }

            with open(seed_summary_path, "w") as f:
                json.dump(seed_summary, f, indent=2)

            self.logger.info(f"Seed summary updated: {seed_summary_path}")

        except Exception as e:
            self.logger.error(f"Error saving run summary: {e}")

    def _count_runs_for_seed(self, seed_dir: str) -> int:
        """Count the number of runs for a seed trajectory"""
        try:
            if not os.path.exists(seed_dir):
                return 0

            # Count directories that start with "run_"
            run_dirs = [
                d
                for d in os.listdir(seed_dir)
                if os.path.isdir(os.path.join(seed_dir, d)) and d.startswith("run_")
            ]
            return len(run_dirs)
        except Exception:
            return 0

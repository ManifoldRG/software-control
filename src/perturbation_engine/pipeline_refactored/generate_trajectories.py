"""
generate_trajectories.py: Entry point with dependency injection
Clean main flow following the data flow specification
"""

import json
import logging
import os
import signal
import sys
from typing import List

from dotenv import load_dotenv

from perturbation_engine.configure_logging import configure_logging
from perturbation_engine.pipeline_refactored.data_models import (
    CurriculumConfig,
    ExecutionConfig,
    SeedTrajectory,
)
from perturbation_engine.pipeline_refactored.unified_generator import UnifiedGenerator

# Global state for graceful shutdown
active_environments = []
processes = []
is_terminating = False


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown"""
    global is_terminating, active_environments, processes

    if is_terminating:
        return

    is_terminating = True
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Gracefully shutting down...")

    # Close environments and terminate processes
    for env in active_environments:
        try:
            env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")

    sys.exit(0)


def load_seed_trajectories(config_base_dir: str, trajectory_base_dir: str) -> List[SeedTrajectory]:
    """Load seed trajectories from task configs and existing trajectories"""
    from pathlib import Path

    seed_trajectories = []
    config_path = Path(config_base_dir)
    logger = logging.getLogger(__name__)

    # Find all task config JSON files in the evaluation examples
    if config_path.name == "evaluation_examples":
        examples_dir = config_path / "examples"
    else:
        examples_dir = config_path

    if not examples_dir.exists():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

    # Get all app directories (chrome, gimp, etc.)
    app_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]

    for app_dir in app_dirs:
        app_name = app_dir.name
        logger.info(f"Loading trajectories for app: {app_name}")

        # Find all JSON config files in this app directory
        config_files = list(app_dir.glob("*.json"))

        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    task_config = json.load(f)

                # Verify required fields
                if not all(field in task_config for field in ["id", "instruction", "config", "evaluator"]):
                    logger.warning(f"Skipping {config_file.name} - missing required fields")
                    continue

                # Construct trajectory path based on the task ID
                task_id = task_config["id"]
                task_trajectory_dir = os.path.join(trajectory_base_dir, app_name, task_id)

                # Verify trajectory directory exists
                if not os.path.exists(task_trajectory_dir):
                    logger.warning(f"Trajectory directory not found: {task_trajectory_dir}")
                    continue

                # Verify traj.jsonl exists
                traj_file = os.path.join(task_trajectory_dir, "traj.jsonl")
                if not os.path.exists(traj_file):
                    logger.warning(f"Trajectory file not found: {traj_file}")
                    continue

                # Create seed trajectory
                seed_trajectory = SeedTrajectory(
                    task_id=task_id,
                    task_type=task_config.get("snapshot", "chrome"),
                    task_instruction=task_config["instruction"],
                    config=task_config,
                    gt_actions_file_path=traj_file,
                    gt_actions=None,
                )

                seed_trajectories.append(seed_trajectory)
                logger.debug(f"Loaded trajectory: {task_id}")

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.error(f"Error loading {config_file.name}: {e}")
                continue

    logger.info(f"Loaded {len(seed_trajectories)} seed trajectories from {len(app_dirs)} app directories")
    return seed_trajectories


def main():
    """Main entry point for the perturbation pipeline"""
    # Load environment variables
    load_dotenv()
    configure_logging()
    logger = logging.getLogger(__name__)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Configuration
    execution_config = ExecutionConfig(
        # VM/Provider settings
        path_to_vm="/Users/lockewang/FIG/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
        provider_name="vmware",
        region=os.environ.get("AWS_REGION", "us-east-1"),
        snapshot_name="chrome",
        # Environment settings
        headless=True,
        action_space="pyautogui",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        client_password="",
        # Execution settings
        max_steps=15,
        sleep_after_execution=0.0,
        # Additional settings
        cache_dir="cache",
        require_a11y_tree=True,
        require_terminal=False,
        enable_proxy=False,
        chromium_port=9222,
    )

    curriculum_config = CurriculumConfig(
        scenario_count=10,
        num_parallel_vms=1,
        result_base_dir="./perturbation_results",
        beginner_scenarios=3,
        intermediate_scenarios=4,
        advanced_scenarios=2,
    )

    # Initialize unified generator
    generator = UnifiedGenerator(execution_config)

    # Load seed trajectories
    task_config_base_dir = "src/OSWorld/evaluation_examples"
    trajectory_base_dir = "external_data/osworld-verified/jedi-7b-4o-15steps/jedi-7b-4o-15steps"

    seed_trajectories = load_seed_trajectories(task_config_base_dir, trajectory_base_dir)
    seed_trajectories = seed_trajectories[:1]  # Limit for testing

    logger.info("Starting perturbation pipeline")
    logger.info(f"Using {len(seed_trajectories)} seed trajectories")

    # Generate trajectories using the complete pipeline
    all_results = []
    for i, seed_trajectory in enumerate(seed_trajectories):
        logger.info(f"Processing seed trajectory {i + 1}/{len(seed_trajectories)}: {seed_trajectory.task_id}")

        try:
            # Generate curriculum-based trajectories for this seed
            trajectory_results = generator.generate_trajectories(
                seed_trajectory=seed_trajectory, curriculum_config=curriculum_config
            )
            all_results.extend(trajectory_results)

            logger.info(f"Generated {len(trajectory_results)} trajectories for seed {i + 1}")

        except Exception as e:
            logger.error(f"Error processing seed trajectory {i + 1}: {e}")
            continue

    # Summary
    logger.info(f"Total results: {len(all_results)}")
    if all_results:
        avg_score = sum(r.quality_score for r in all_results) / len(all_results)
        success_count = sum(1 for r in all_results if r.success)
        logger.info(f"Average quality score: {avg_score:.2f}")
        logger.info(
            f"Success rate: {success_count}/{len(all_results)} ({success_count / len(all_results) * 100:.1f}%)"
        )

    return all_results


if __name__ == "__main__":
    main()

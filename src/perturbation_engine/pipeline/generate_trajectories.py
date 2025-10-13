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
from perturbation_engine.pipeline.data_models import (
    CurriculumConfig,
    ExecutionConfig,
    SeedTrajectory,
)
from perturbation_engine.pipeline.unified_generator import UnifiedGenerator
from perturbation_engine.utils.memory_utils import force_garbage_collection, log_memory_usage

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

    # Close all registered environments in the main process
    for env in active_environments:
        try:
            logger.info("Closing environment...")
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    # Send termination signal to all child processes first
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")

    # Allow a short time for processes to handle their own cleanup
    import time

    time.sleep(1)

    # Forcefully terminate any processes that didn't exit
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import os
                import signal as sig

                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")

    logger.info("Shutdown complete. Exiting.")
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

                # Construct trajectory path based on the task ID
                task_id = task_config["id"]
                # task_trajectory_dir = os.path.join(trajectory_base_dir, app_name, task_id + ".json")
                traj_file = os.path.join(trajectory_base_dir, app_name, task_id + ".json")

                # # Verify trajectory directory exists
                # if not os.path.exists(task_trajectory_dir):
                #     logger.warning(f"Trajectory directory not found: {task_trajectory_dir}")
                #     continue

                # Verify traj.jsonl exists
                # traj_file = os.path.join(task_trajectory_dir, "traj.jsonl")

                if not os.path.exists(traj_file):
                    logger.warning(f"Trajectory file not found: {traj_file}")
                    continue

                # Create seed trajectory
                seed_trajectory = SeedTrajectory(
                    task_id=task_id,
                    task_type=app_name,
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
        path_to_vm=os.environ.get("VMWARE_PATH_TO_VM", None),
        provider_name=os.environ.get("PROVIDER_NAME", "aws"),
        region=os.environ.get("AWS_REGION", "us-east-1"),
        snapshot_name=os.environ.get("AWS_SNAPSHOT_NAME", "chrome"),
        # snapshot_name="chrome",
        # Environment settings
        headless=True,
        action_space="pyautogui",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        client_password=os.environ.get("AWS_CLIENT_PASSWORD", "osworld-public-evaluation"),
        # Execution settings
        max_steps=15,
        sleep_after_execution=0.0,
        # Additional settings
        cache_dir="cache",
        require_a11y_tree=True,
        require_terminal=True,
        enable_proxy=False,
        chromium_port=9222,
    )

    curriculum_config = CurriculumConfig(
        scenario_count=1,
        num_parallel_vms=1,
        result_base_dir=os.environ.get("RESULT_BASE_DIR", "/opt/manifold/results"),
        beginner_scenarios=0,
        intermediate_scenarios=1,
        advanced_scenarios=0,
    )

    # Initialize unified generator
    generator = UnifiedGenerator(execution_config, curriculum_config.result_base_dir)

    # Load seed trajectories
    # task_config_base_dir = "src/OSWorld/evaluation_examples"
    task_config_base_dir = "osworld-human-main"

    # trajectory_base_dir = "external_data/osworld-verified/autoglm_50steps"
    # trajectory_base_dir = "external_data/osworld-verified/o3_gta1_100steps/o3_gta1_100steps"
    # trajectory_base_dir = "external_data/osworld-verified/jedi-7b-4o-15steps/jedi-7b-4o-15steps"
    trajectory_base_dir = "osworld-human-main"

    seed_trajectories = load_seed_trajectories(task_config_base_dir, trajectory_base_dir)

    # TODO: get 1 seed traj for each task type for testing
    # traj_task_type_set = set()
    # test_seed_trajectories = []
    # for traj in seed_trajectories:
    #     if traj.task_type not in traj_task_type_set:
    #         with open(f"{traj.gt_actions_file_path}") as f:
    #             temp_gt_actions = [json.loads(line) for line in f]
    #             if temp_gt_actions[-1]["action"] != "FAIL":
    #                 test_seed_trajectories.append(traj)
    #                 traj_task_type_set.add(traj.task_type)
    #             else:
    #                 logger.info(f"Task type already exists: {traj.task_type}, skipping")
    #                 continue
    # logger.info(f"Task types: {traj_task_type_set}")

    test_seed_trajectories = [
        traj
        for traj in seed_trajectories
        if traj.task_type == "chrome" and traj.task_id == "7f52cab9-535c-4835-ac8c-391ee64dc930"
    ][:2]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "vlc"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "chrome"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "vs_code"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "os"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "multi_apps" or traj.task_type == "multiapps"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "libreoffice_calc"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "libreoffice_impress"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "libreoffice_writer"][:1]

    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "gimp"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "vlc"][:1]
    # test_seed_trajectories = [traj for traj in seed_trajectories if traj.task_type == "thunderbird"][:1]

    logger.info(f"Test seed trajectories: {len(test_seed_trajectories)}")

    seed_trajectories = test_seed_trajectories

    logger.info("Starting perturbation pipeline")
    logger.info(f"Using {len(seed_trajectories)} seed trajectories")

    # Generate trajectories using the complete pipeline
    all_results = []
    for i, seed_trajectory in enumerate(seed_trajectories):
        logger.info(f"Processing seed trajectory {i + 1}/{len(seed_trajectories)}: {seed_trajectory.task_id}")

        # Log memory usage before each trajectory
        log_memory_usage(f"Before trajectory {i + 1}", logger, threshold_mb=3000)

        try:
            # Generate curriculum-based trajectories for this seed
            trajectory_results = generator.generate_trajectories(
                seed_trajectory=seed_trajectory, curriculum_config=curriculum_config
            )
            all_results.extend(trajectory_results)

            logger.info(f"Generated {len(trajectory_results)} trajectories for seed {i + 1}")

            # Force garbage collection after each trajectory
            force_garbage_collection(logger)

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

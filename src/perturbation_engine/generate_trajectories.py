# Portions of this code are adapted from the OSWorld repository
# https://github.com/xlang-ai/OSWorld
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process
from typing import Any, Dict, List

from perturbation_engine.configure_logging import configure_logging
from perturbation_engine.data_types import ExecutionConfig, GenerationResult
from perturbation_engine.pipeline.parallel_execution_engine import ParallelExecutionEngine
from perturbation_engine.pipeline.scenario_generator import ScenarioGenerator

configure_logging()
logger = logging.getLogger(__name__)

active_environments = []
processes = []
is_terminating = False


class TrajectoryGenerationOrchestrator:
    """Main orchestrator for trajectory generation with perturbations"""

    def __init__(self, scenario_generator: ScenarioGenerator) -> None:
        self.scenario_generator = scenario_generator

    def generate_trajectories(
        self,
        num_seed_scenarios: int,
        num_trajectories_per_seed: int,
        num_parallel_vms: int = 1,
        config: ExecutionConfig = ExecutionConfig(),
        task_config_base_dir: str = "evaluation_examples",
        trajectory_base_dir: str = "external_data/osworld-verified/jedi-7b-4o-15steps",
        result_base_dir: str = "./perturbation_results",
    ) -> List[GenerationResult]:
        """Generate trajectories with perturbation injection"""
        logger.info(
            f"Generating trajectories for {num_seed_scenarios} seed scenarios with {num_trajectories_per_seed} trajectories per seed..."
        )
        seed_scenarios = self.load_seed_scenarios(task_config_base_dir, trajectory_base_dir)

        scenario_specs = self.scenario_generator.generate_scenarios(
            seed_scenarios, num_trajectories_per_seed, result_base_dir
        )

        with Manager() as manager:
            shared_results = manager.list()
            scenario_queue = manager.Queue()

            for scenario_spec in scenario_specs:
                scenario_queue.put(scenario_spec)

            processes = []
            for i in range(num_parallel_vms):
                execution_engine = ParallelExecutionEngine(config)
                p = Process(
                    target=execution_engine.run_vm_tasks,
                    args=(scenario_queue, shared_results),
                    name=f"PerturbationProcess-{i + 1}",
                )
                p.daemon = True
                p.start()
                processes.append(p)
                logger.info(f"Started process {p.name} with PID {p.pid}")

            try:
                while True:
                    alive_count = sum(1 for p in processes if p.is_alive())
                    if scenario_queue.empty():
                        logger.info("All tasks finished.")
                        break
                    if alive_count == 0:
                        logger.error("All processes died, exiting.")
                        break
                    time.sleep(5)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                logger.info("Main process received KeyboardInterrupt.")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise

        results = list(shared_results)
        logger.info(
            f"Average result: {sum(r.result_score for r in results) / len(results) if results else 0}"
        )
        return results

    def load_seed_scenarios(self, config_base_dir: str, trajectory_base_dir: str) -> List[Dict[str, Any]]:
        """Load seed scenarios from task configs and existing trajectories"""
        from pathlib import Path

        seed_scenarios = []

        # Convert to Path for easier manipulation
        config_path = Path(config_base_dir)
        trajectory_path = Path(trajectory_base_dir)

        # Find all task config JSON files in the evaluation examples
        if config_path.name == "evaluation_examples":
            # Look in examples subdirectories
            examples_dir = config_path / "examples"
        else:
            examples_dir = config_path

        if not examples_dir.exists():
            logger.warning(f"Examples directory not found: {examples_dir}")
            return seed_scenarios

        # Get all app directories (chrome, gimp, etc.)
        app_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]

        for app_dir in app_dirs:
            app_name = app_dir.name
            logger.info(f"Loading scenarios for app: {app_name}")

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
                        logger.warning(f"Skipping {config_file.name} - missing required fields")
                        continue

                    # Construct trajectory path based on the task ID
                    task_id = task_config["id"]
                    trajectory_path = os.path.join(trajectory_base_dir, app_name, task_id)

                    # Verify trajectory directory exists
                    if not os.path.exists(trajectory_path):
                        logger.warning(f"Trajectory directory not found: {trajectory_path}")
                        continue

                    # Verify traj.jsonl exists
                    traj_file = os.path.join(trajectory_path, "traj.jsonl")
                    if not os.path.exists(traj_file):
                        logger.warning(f"Trajectory file not found: {traj_file}")
                        continue

                    # Create seed scenario with trajectory path
                    seed_scenario = {
                        "id": task_config["id"],
                        "snapshot": task_config.get("snapshot", "chrome"),
                        "instruction": task_config["instruction"],
                        "source": task_config.get("source", "unknown"),
                        "config": task_config["config"],
                        "trajectory": trajectory_path,
                        "related_apps": task_config.get("related_apps", [app_name]),
                        "evaluator": task_config["evaluator"],
                        "proxy": task_config.get("proxy", False),
                        "fixed_ip": task_config.get("fixed_ip", False),
                        "possibility_of_env_change": task_config.get("possibility_of_env_change", "low"),
                    }

                    # Add any additional fields from the original config
                    for key, value in task_config.items():
                        if key not in seed_scenario:
                            seed_scenario[key] = value

                    seed_scenarios.append(seed_scenario)
                    logger.debug(f"Loaded scenario: {task_id}")

                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.error(f"Error loading {config_file.name}: {e}")
                    continue

        logger.info(f"Loaded {len(seed_scenarios)} seed scenarios from {len(app_dirs)} app directories")
        return seed_scenarios


# ============================================================================
# Signal Handler
# ============================================================================


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown"""
    global is_terminating, active_environments, processes

    if is_terminating:
        return

    is_terminating = True
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


def main():
    """Main entry point for trajectory generation"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components
    scenario_generator = ScenarioGenerator()
    orchestrator = TrajectoryGenerationOrchestrator(scenario_generator)

    # Example usage
    config = ExecutionConfig(
        # VM/Provider settings
        path_to_vm=None,
        provider_name="docker",
        region="us-east-1",
        snapshot_name=None,
        # Environment settings
        headless=True,
        action_space="pyautogui",
        observation_type="screenshot",
        screen_size=(1920, 1080),
        os_type="Ubuntu",
        client_password="password",
        # Execution settings
        max_steps=15,
        sleep_after_execution=0.0,
        # Additional OSWorld settings
        cache_dir="cache",
        require_a11y_tree=True,
        require_terminal=False,
        enable_proxy=True,
        # Perturbation connection
        chromium_port=9222,
    )

    # Test configuration
    task_config_base_dir = "evaluation_examples"
    trajectory_base_dir = "external_data/osworld-verified/jedi-7b-4o-15steps"
    result_base_dir = "./perturbation_results"

    results = orchestrator.generate_trajectories(
        num_seed_scenarios=10,
        num_trajectories_per_seed=3,
        num_parallel_vms=2,
        config=config,
        task_config_base_dir=task_config_base_dir,
        trajectory_base_dir=trajectory_base_dir,
        result_base_dir=result_base_dir,
    )

    return results


if __name__ == "__main__":
    main()

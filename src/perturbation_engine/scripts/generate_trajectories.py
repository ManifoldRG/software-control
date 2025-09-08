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

import logging
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
        env_args: Dict[str, Any] = None,
        result_base_dir: str = "./perturbation_results",
    ) -> List[GenerationResult]:
        """Generate trajectories with perturbation injection"""
        logger.info(
            f"Generating trajectories for {num_seed_scenarios} seed scenarios with {num_trajectories_per_seed} trajectories per seed..."
        )

        if env_args is None:
            env_args = {}

        # Create execution config
        config = ExecutionConfig(**env_args)

        # Load seed scenarios
        seed_scenarios = self.scenario_generator.load_seed_scenarios(
            env_args.get("test_config_base_dir", "evaluation_examples")
        )

        # Generate scenario specifications
        scenario_specs = self.scenario_generator.generate_scenarios(seed_scenarios, num_trajectories_per_seed)

        # Execute in parallel
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
                # Wait for completion
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
    env_args = {
        # VM/Provider settings
        "path_to_vm": None,
        "provider_name": "docker",
        "region": "us-east-1",
        "snapshot_name": None,
        # Environment settings
        "headless": True,
        "action_space": "pyautogui",
        "observation_type": "screenshot",
        "screen_size": (1920, 1080),
        "os_type": "Ubuntu",
        "client_password": "",
        # Execution settings
        "max_steps": 15,
        "sleep_after_execution": 0.0,
        # Additional OSWorld settings
        "cache_dir": "cache",
        "require_a11y_tree": True,
        "require_terminal": False,
        "enable_proxy": True,
        # Test configuration
        "test_config_base_dir": "evaluation_examples",
    }

    results = orchestrator.generate_trajectories(
        num_seed_scenarios=10, num_trajectories_per_seed=3, num_parallel_vms=2, env_args=env_args
    )

    return results


if __name__ == "__main__":
    main()

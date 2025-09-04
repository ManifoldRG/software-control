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

import argparse
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process
from typing import Any, Dict, List

from perturbation_engine.configure_logging import configure_logging
from perturbation_engine.data_types import GenerationResult, ScenarioSpec
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
        self.execution_engine = ParallelExecutionEngine()

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

        # Load seed scenarios
        seed_scenarios = self._load_seed_scenarios(
            env_args.get("test_config_base_dir", "evaluation_examples")
        )

        # Generate scenario specifications
        scenario_specs = self.scenario_generator.generate_scenarios(seed_scenarios, num_trajectories_per_seed)

        # Create tasks
        tasks = self._create_tasks(scenario_specs, result_base_dir)

        # Create argument namespace for compatibility
        args = argparse.Namespace(**env_args)

        # Execute in parallel
        with Manager() as manager:
            shared_results = manager.list()
            task_queue = manager.Queue()

            for task in tasks:
                task_queue.put(task)

            processes = []
            for i in range(num_parallel_vms):
                p = Process(
                    target=self.execution_engine.run_vm_tasks,
                    args=(task_queue, args, shared_results),
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
                    if task_queue.empty():
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

    def _load_seed_scenarios(self, config_base_dir: str) -> List[Dict[str, Any]]:
        """Load seed scenarios from task configs and existing trajectories"""
        # TODO: Implement scenario loading logic
        # - Load from evaluation_examples directory
        # - Filter based on domain/type requirements
        # - Return list of task configurations
        return []

    def _create_tasks(self, scenario_specs: List[ScenarioSpec], result_base_dir: str) -> List[ScenarioSpec]:
        """Create trajectory tasks from scenario specifications"""
        tasks = []
        for i, scenario_spec in enumerate(scenario_specs):
            result_dir = os.path.join(result_base_dir, f"scenario_{i}")
            task_id = f"task_{i}_{scenario_spec.scenario_id}"
            tasks.append(
                ScenarioSpec(
                    scenario_id=scenario_spec.scenario_id,
                    base_task_config=scenario_spec.base_task_config,
                    perturbations=scenario_spec.perturbations,
                    metadata=scenario_spec.metadata,
                    result_dir=result_dir,
                    task_id=task_id,
                    perturbation_type=scenario_spec.perturbation_type,
                    perturbation_phase=scenario_spec.perturbation_phase,
                    perturbation_params=scenario_spec.perturbation_params,
                )
            )
        return tasks


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
        "path_to_vm": None,
        "headless": True,
        "action_space": "pyautogui",
        "observation_type": "screenshot",
        "max_steps": 15,
        "model": "gpt-4o",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 1500,
        "provider_name": "docker",
        "region": "us-east-1",
        "screen_width": 1920,
        "screen_height": 1080,
        "client_password": "",
        "os_type": "Ubuntu",
        "test_config_base_dir": "evaluation_examples",
    }

    results = orchestrator.generate_trajectories(
        num_seed_scenarios=10, num_trajectories_per_seed=3, num_parallel_vms=2, env_args=env_args
    )

    return results


if __name__ == "__main__":
    main()

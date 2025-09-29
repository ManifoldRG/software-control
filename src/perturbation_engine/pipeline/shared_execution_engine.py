"""
SharedExecutionEngine: Parallel VM execution manager
Clean interface for parallel trajectory generation
"""

import logging
import time
from multiprocessing import Manager, Process, Queue, current_process
from typing import List

from perturbation_engine.pipeline.data_models import (
    ExecutionConfig,
    GeneratedTrajectory,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.trajectory_generator import TrajectoryGenerator


class SharedExecutionEngine:
    """Parallel VM execution manager"""

    def __init__(self, execution_config: ExecutionConfig):
        self.execution_config = execution_config
        self.logger = logging.getLogger(__name__)

    def execute_scenarios_parallel(
        self, seed_trajectory: SeedTrajectory, scenario_specs: List[ScenarioSpec], num_parallel_vms: int = 1
    ) -> List[GeneratedTrajectory]:
        """Execute scenarios in parallel using shared queue management"""

        self.logger.info(f"Starting parallel execution with {num_parallel_vms} VMs")

        with Manager() as manager:
            shared_results = manager.list()
            scenario_queue = manager.Queue()

            # Add scenarios to queue
            for scenario_spec in scenario_specs:
                scenario_queue.put((seed_trajectory, scenario_spec))

            # Start parallel processes
            processes = []
            for i in range(num_parallel_vms):
                # TODO: initializie env here per process and reset it in the parallel execution engine

                execution_engine = ParallelExecutionEngine(self.execution_config)
                p = Process(
                    target=execution_engine.run_vm_tasks,
                    args=(scenario_queue, shared_results),
                    name=f"ExecutionProcess-{i + 1}",
                )
                p.daemon = True
                p.start()
                processes.append(p)
                self.logger.info(f"Started process {p.name} with PID {p.pid}")

            # Wait for completion
            try:
                while True:
                    alive_count = sum(1 for p in processes if p.is_alive())
                    if scenario_queue.empty():
                        self.logger.info("All tasks finished")
                        break
                    if alive_count == 0:
                        self.logger.error("All processes died, exiting")
                        break
                    time.sleep(5)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                self.logger.info("Execution interrupted")
                raise

            results = list(shared_results)
            self.logger.info(f"Completed {len(results)} trajectories")
            return results


class ParallelExecutionEngine:
    """Manages parallel execution of trajectory generation tasks"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trajectory_generator = TrajectoryGenerator()

    def run_vm_tasks(self, scenario_queue: Queue, shared_results: list):
        """Run trajectory generation scenarios in a single VM process"""
        env = None
        try:
            # Initialize environment
            env = PerturbationDesktopEnv(
                path_to_vm=self.config.path_to_vm,
                action_space=self.config.action_space,
                provider_name=self.config.provider_name,
                region=self.config.region,
                snapshot_name=self.config.snapshot_name,
                screen_size=self.config.screen_size,
                headless=self.config.headless,
                os_type=self.config.os_type,
                require_a11y_tree=self.config.require_a11y_tree,
                require_terminal=self.config.require_terminal,
                enable_proxy=self.config.enable_proxy,
                client_password=self.config.client_password,
                cache_dir=self.config.cache_dir,
                chromium_port=self.config.chromium_port,
            )
            self.logger.info(f"Process {current_process().name} started with environment initialized")

            while True:
                try:
                    seed_trajectory, scenario_spec = scenario_queue.get(timeout=5)
                except Exception:
                    break

                try:
                    # Execute trajectory with scenario
                    result = self.trajectory_generator.execute_trajectory(
                        env, seed_trajectory, scenario_spec, self.config.max_steps
                    )
                    shared_results.append(result)

                except Exception as e:
                    self.logger.error(f"Task-level error in {current_process().name}: {e}")
                    import traceback

                    self.logger.error(traceback.format_exc())

        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg and "docker" in error_msg.lower():
                self.logger.error(
                    f"Environment initialization error in {current_process().name}: "
                    f"Docker connection failed. Please ensure Docker is running or change provider to 'vmware'. "
                    f"Original error: {e}"
                )
            else:
                self.logger.error(f"Environment initialization error in {current_process().name}: {e}")

            import traceback

            self.logger.error(traceback.format_exc())

        finally:
            if env:
                try:
                    env.close()
                    self.logger.info(f"Environment closed for {current_process().name}")
                except Exception as e:
                    self.logger.error(f"Error closing environment in {current_process().name}: {e}")

        self.logger.info(f"{current_process().name} finished")

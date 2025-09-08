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
from multiprocessing import Queue, current_process

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.data_types import ExecutionConfig, GenerationResult, ScenarioSpec
from perturbation_engine.pipeline.trajectory_generator import TrajectoryGenerator
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer


class TaskExecutor:
    """Handles single task execution with environment setup"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.trajectory_generator = TrajectoryGenerator()
        self.logger = logging.getLogger(__name__)

    def execute_scenario(self, scenario: ScenarioSpec) -> GenerationResult:
        """Execute a single scenario"""
        env = None
        try:
            # Initialize DesktopEnv
            env = DesktopEnv(
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
            )

            # Initialize trajectory replayer
            trajectory_replayer = TrajectoryReplayer(trajectory_file_path=scenario.trajectory_file_path)

            # Execute trajectory
            return self.trajectory_generator.execute_trajectory(
                trajectory_replayer, env, scenario, self.config.max_steps, self.config.sleep_after_execution
            )

        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise
        finally:
            if env:
                try:
                    env.close()
                except Exception as e:
                    self.logger.error(f"Error closing environment: {e}")


class ParallelExecutionEngine:
    """Manages parallel execution of trajectory generation tasks"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_vm_tasks(self, scenario_queue: Queue, shared_results: list):
        """Run trajectory generation scenarios in a single VM process"""
        executor = TaskExecutor(self.config)
        self.logger.info(f"Process {current_process().name} started.")

        while True:
            try:
                scenario = scenario_queue.get(timeout=5)
            except Exception:
                break

            try:
                result = executor.execute_scenario(scenario)
                shared_results.append(result)
            except Exception as e:
                self.logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback

                self.logger.error(traceback.format_exc())

        self.logger.info(f"{current_process().name} finished.")


if __name__ == "__main__":
    from perturbation_engine.data_types import ExecutionConfig

    config = ExecutionConfig()
    engine = ParallelExecutionEngine(config)
    engine.run_vm_tasks(Queue(), [])

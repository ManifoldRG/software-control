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

from perturbation_engine.data_types import ExecutionConfig
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.trajectory_generator import TrajectoryGenerator


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
            # Initialize DesktopEnv and TrajectoryReplayer once per process
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
            self.logger.info(f"Process {current_process().name} started with environment initialized.")

            while True:
                try:
                    scenario = scenario_queue.get(timeout=5)
                except Exception:
                    break

                try:
                    # TODO: Execute the scenario with the perturbation scenario
                    result = self.trajectory_generator.execute_trajectory(
                        env,
                        scenario,
                        self.config.max_steps,
                        self.config.sleep_after_execution,
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

        self.logger.info(f"{current_process().name} finished.")


if __name__ == "__main__":
    from perturbation_engine.data_types import ExecutionConfig

    config = ExecutionConfig()
    engine = ParallelExecutionEngine(config)
    engine.run_vm_tasks(Queue(), [])

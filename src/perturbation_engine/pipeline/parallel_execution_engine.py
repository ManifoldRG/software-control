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
from multiprocessing import Queue, current_process

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.pipeline.replay_agent import ReplayAgent
from perturbation_engine.pipeline.trajectory_generator import TrajectoryGenerator


class ParallelExecutionEngine:
    """Manages parallel execution of trajectory generation tasks"""

    def __init__(self):
        self.trajectory_generator = TrajectoryGenerator()
        self.logger = logging.getLogger(__name__)

    def run_vm_tasks(self, scenario_queue: Queue, args: argparse.Namespace, shared_results: list):
        """Run trajectory generation tasks in a single VM process"""
        active_environments = []
        env = None
        try:
            # Initialize DesktopEnv
            env = DesktopEnv(
                path_to_vm=args.path_to_vm,
                action_space=args.action_space,
                provider_name=args.provider_name,
                region=args.region,
                snapshot_name=args.snapshot_name,
                screen_size=(args.screen_width, args.screen_height),
                headless=args.headless,
                os_type=args.os_type,
                require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
                enable_proxy=True,
                client_password=args.client_password,
            )
            active_environments.append(env)

            # Initialize agent
            agent = ReplayAgent(trajectory_folder_dir=args.trajectory_folder_dir)

            self.logger.info(f"Process {current_process().name} started.")

            while True:
                try:
                    scenario = scenario_queue.get(timeout=5)
                except Exception:
                    break

                try:
                    result = self.trajectory_generator.execute_trajectory(agent, env, scenario, args)
                    shared_results.append(result)
                except Exception as e:
                    self.logger.error(f"Task-level error in {current_process().name}: {e}")
                    import traceback

                    self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Process-level error in {current_process().name}: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info(f"{current_process().name} cleaning up environment...")
            try:
                if env:
                    env.close()
                    self.logger.info(f"{current_process().name} environment closed successfully")
            except Exception as e:
                self.logger.error(f"{current_process().name} error during environment cleanup: {e}")

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


import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List

from perturbation_engine.data_types import GenerationResult, ScenarioSpec
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer
from perturbation_engine.scenarios.scenario_factory import PerturbationScenarioFactory


class TrajectoryGenerator:
    """Executes trajectory generation from existing task trajectories with perturbation injection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trajectory_replayer = TrajectoryReplayer()
        self.scenario_factory = PerturbationScenarioFactory()

    def execute_trajectory(
        self,
        env: PerturbationDesktopEnv,
        scenario: ScenarioSpec,
        max_steps: int,
        sleep_after_execution: float = 0.0,
    ) -> GenerationResult:
        """Execute a single trajectory with perturbation injection"""
        start_time = time.time()
        os.makedirs(scenario.result_dir, exist_ok=True)

        self.trajectory_replayer.load_trajectory(scenario.trajectory_file_path)

        # Apply setup perturbations before environment reset
        perturbation_scenario = self.scenario_factory.create_scenario(scenario.perturbation_scenario_class)
        perturbed_config = perturbation_scenario.apply_setup_perturbations(
            scenario.task_config, scenario.perturbation_scenario_class, scenario.perturbation_parameters
        )

        env.reset(task_config=perturbed_config)
        time.sleep(60)  # Wait for environment to be ready

        # Start recording video
        env.controller.start_recording()

        # Get initial observation
        obs = env._get_obs()
        done = False
        step_idx = 0
        perturbation_log = []

        # Main execution loop
        while not done and step_idx < max_steps and self.trajectory_replayer.has_more_steps():
            # Get next action from trajectory replayer
            response, actions = self.trajectory_replayer.step()

            # Execute actions
            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                self.logger.info("Step %d: %s", step_idx + 1, action)

                # Apply runtime perturbations
                runtime_perturbation_result = perturbation_scenario.check_and_apply_runtime_perturbations(
                    env,
                    perturbation_scenario,
                    scenario.perturbation_parameters,
                    step_idx,
                    obs,
                    perturbation_log,
                )

                # Execute action
                obs, reward, done, info = env.step(action, sleep_after_execution)

                # Save trajectory data
                self._save_trajectory_step(
                    scenario.result_dir,
                    step_idx + 1,
                    action_timestamp,
                    action,
                    response,
                    reward,
                    done,
                    info,
                    obs,
                    runtime_perturbation_result,
                )

                if done:
                    self.logger.info("The episode is done.")
                    break

            step_idx += 1

        # Complete trajectory
        result = env.evaluate()
        generation_time = time.time() - start_time

        # Save final results
        self._save_trajectory_results(scenario.result_dir, result, perturbation_log)
        env.controller.end_recording(os.path.join(scenario.result_dir, "recording.mp4"))

        return GenerationResult(
            task_id=scenario.task_id,
            success=result > 0,
            result_score=result,
            perturbation_log=perturbation_log,
            generation_time=generation_time,
            metadata={"scenario_id": scenario.scenario_id},
        )

    def _save_trajectory_step(
        self,
        result_dir: str,
        step_num: int,
        timestamp: str,
        action,
        response,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        obs: Dict[str, Any],
        perturbation_applied: bool,
    ):
        """Save individual trajectory step data"""
        # Save screenshot
        with open(os.path.join(result_dir, f"step_{step_num}_{timestamp}.png"), "wb") as f:
            f.write(obs["screenshot"])

        # Save trajectory data
        with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "step_num": step_num,
                        "action_timestamp": timestamp,
                        "action": action,
                        "response": response,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": f"step_{step_num}_{timestamp}.png",
                        "perturbation_applied": perturbation_applied,
                    }
                )
            )
            f.write("\n")

    def _save_trajectory_results(
        self, result_dir: str, result: float, perturbation_log: List[Dict[str, Any]]
    ):
        """Save final trajectory results and perturbation log"""
        # Save result
        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"{result}\n")

        # Save perturbation log
        with open(os.path.join(result_dir, "perturbations.json"), "w", encoding="utf-8") as f:
            json.dump(perturbation_log, f, indent=2)

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
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.data_types import (
    GenerationResult,
    PerturbationControllers,
    PerturbationPhase,
    PerturbationSpec,
    PerturbationType,
    ScenarioSpec,
)
from perturbation_engine.pipeline.replay_agent import ReplayAgent


class TrajectoryGenerator:
    """Executes trajectory generation from existing task trajectories with perturbation injection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def execute_trajectory(
        self, replay_agent: ReplayAgent, env: DesktopEnv, scenario: ScenarioSpec, args: argparse.Namespace
    ) -> GenerationResult:
        """Execute a single trajectory with perturbation injection"""
        start_time = time.time()
        os.makedirs(scenario.result_dir, exist_ok=True)

        # Apply setup perturbations
        perturbed_config = self._apply_setup_perturbations(
            scenario.base_task_config, scenario.perturbations, env
        )

        # Reset environment with perturbed task
        env.reset(task_config=perturbed_config)
        time.sleep(60)  # Wait for environment to be ready

        # Initialize replay agent by loading the trajectory
        replay_agent.reset(scenario.trajectory_folder_dir)

        # Start recording video
        env.controller.start_recording()

        # Main execution loop
        obs = env._get_obs()
        done = False
        step_idx = 0
        perturbation_log = []

        while not done and step_idx < args.max_steps:
            # Get next action
            response, actions = replay_agent.step()

            # Execute actions
            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                self.logger.info("Step %d: %s", step_idx + 1, action)

                # Apply runtime perturbations
                runtime_perturbation = self._check_and_apply_runtime_perturbations(
                    env, scenario.perturbations, step_idx, obs, perturbation_log
                )

                # Execute action
                obs, reward, done, info = env.step(action, args.sleep_after_execution)

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
                    runtime_perturbation,
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

    def _apply_setup_perturbations(
        self, base_config: Dict[str, Any], perturbations: List[PerturbationSpec], env: DesktopEnv
    ) -> Dict[str, Any]:
        """Apply setup-phase perturbations"""
        perturbed_config = base_config.copy()
        context = {"phase": "setup", "base_config": base_config}

        for perturbation in perturbations:
            if perturbation.phase == PerturbationPhase.SETUP:
                result = self.perturbation_manager.apply_perturbation(env, perturbation, context)
                if result.get("applied"):
                    # Update config based on perturbation result
                    if perturbation.perturbation_type == PerturbationType.INSTRUCTION:
                        # TODO: Update instruction based on perturbation
                        pass

        return perturbed_config

    def _check_and_apply_runtime_perturbations(
        self,
        env: DesktopEnv,
        perturbations: List[PerturbationSpec],
        step_idx: int,
        obs: Dict[str, Any],
        perturbation_log: List[Dict[str, Any]],
    ) -> bool:
        """Check and apply runtime perturbations"""
        context = {"phase": "runtime", "step_idx": step_idx, "obs": obs}

        for perturbation in perturbations:
            if perturbation.phase == PerturbationPhase.RUNTIME and self._should_trigger_perturbation(
                perturbation, step_idx, obs
            ):
                result = PerturbationControllers[perturbation.perturbation_controller].apply_perturbation(
                    env, perturbation, context
                )
                if result.get("applied"):
                    perturbation_log.append(
                        {
                            "step": step_idx,
                            "type": perturbation.perturbation_type.value,
                            "parameters": perturbation.parameters,
                            "result": result,
                        }
                    )
                    return True

        return False

    def _should_trigger_perturbation(
        self, perturbation: PerturbationSpec, step_idx: int, obs: Dict[str, Any]
    ) -> bool:
        """Check if perturbation should be triggered"""
        conditions = perturbation.trigger_conditions

        # Time-based triggers
        if "step_range" in conditions:
            start, end = conditions["step_range"]
            if not (start <= step_idx <= end):
                return False

        # State-based triggers
        if "ui_elements" in conditions:
            # TODO: Check if specific UI elements are present in obs
            pass

        return True

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

"""
TrajectoryGenerator: Single trajectory execution
Seed + spec → perturbed trajectory
"""

import datetime
import json
import logging
import os
import time
from typing import Any, Dict

from perturbation_engine.control_refactored.perturbation_controller import PerturbationController
from perturbation_engine.pipeline_refactored.data_models import (
    ExecutionContext,
    GeneratedTrajectory,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline_refactored.llm_services import PerturbationLLM


class TrajectoryGenerator:
    """Execute single trajectory with perturbation injection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.perturbation_llm = PerturbationLLM()

    def execute_trajectory(
        self, env, seed_trajectory: SeedTrajectory, scenario_spec: ScenarioSpec, max_steps: int = 15
    ) -> GeneratedTrajectory:
        """Execute trajectory with runtime perturbation"""

        start_time = time.time()
        trajectory_id = f"{seed_trajectory.task_id}_{scenario_spec.scenario_id}"

        self.logger.info(f"Executing trajectory {trajectory_id}")

        try:
            # Initialize trajectory replayer
            from perturbation_engine.pipeline_refactored.trajectory_replayer import TrajectoryReplayer

            trajectory_replayer = TrajectoryReplayer()
            trajectory_replayer.load_trajectory(seed_trajectory.gt_actions_file_path)

            # Reset environment
            env.reset(task_config=seed_trajectory.config)
            env.controller.start_recording()

            # Initialize execution state
            app_states = env.get_app_states_from_accessibility_tree()
            done = False
            step_idx = 0
            perturbation_log = []
            action_history = []

            # Main execution loop
            while not done and step_idx < max_steps and trajectory_replayer.has_more_steps():
                # Get next action from trajectory replayer
                response, actions = trajectory_replayer.step()

                # Execute actions
                for action in actions:
                    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                    self.logger.info(f"Step {step_idx + 1}: {action}")

                    # Create execution context
                    execution_context = ExecutionContext(
                        step_idx=step_idx,
                        current_action=str(action),
                        action_history=action_history.copy(),
                        cot_context=response.get("thought", ""),
                        app_states=app_states,
                        task_instruction=seed_trajectory.task_instruction,
                        task_type=seed_trajectory.task_type,
                        scenario_spec=scenario_spec,
                    )

                    # Let LLM decide whether to apply perturbation
                    perturbation_decision = self.perturbation_llm.decide_perturbation(
                        execution_context, scenario_spec
                    )

                    # Apply perturbation if LLM decides to
                    if perturbation_decision.get("should_apply", False):
                        perturbation_result = self._apply_perturbation(env.controller, perturbation_decision)
                        perturbation_log.append(
                            {
                                "step": step_idx + 1,
                                "timestamp": action_timestamp,
                                "decision": perturbation_decision,
                                "result": perturbation_result,
                            }
                        )
                        self.logger.info(
                            f"Applied perturbation: {perturbation_decision.get('reasoning', '')}"
                        )

                    # Execute original action
                    obs, reward, done, info = env.step(action)
                    action_history.append(str(action))

                    # Save trajectory step
                    self._save_trajectory_step(
                        trajectory_id,
                        step_idx + 1,
                        action_timestamp,
                        action,
                        response,
                        reward,
                        done,
                        info,
                        obs,
                        perturbation_decision.get("should_apply", False),
                    )

                    if done:
                        self.logger.info("Episode completed")
                        break

                step_idx += 1

            # Complete trajectory
            result = env.evaluate()
            generation_time = time.time() - start_time

            # Stop recording
            env.controller.stop_recording()

            # Create generated trajectory
            generated_trajectory = GeneratedTrajectory(
                trajectory_id=trajectory_id,
                seed_trajectory_id=seed_trajectory.task_id,
                scenario_spec_id=scenario_spec.scenario_id,
                success=result > 0,
                quality_score=result,
                generation_time=generation_time,
                trajectory_file_path=f"trajectories/{trajectory_id}.jsonl",
                perturbation_log=perturbation_log,
            )

            self.logger.info(f"Trajectory {trajectory_id} completed: success={result > 0}, score={result}")
            return generated_trajectory

        except Exception as e:
            self.logger.error(f"Error executing trajectory {trajectory_id}: {e}")
            return GeneratedTrajectory(
                trajectory_id=trajectory_id,
                seed_trajectory_id=seed_trajectory.task_id,
                scenario_spec_id=scenario_spec.scenario_id,
                success=False,
                quality_score=0.0,
                generation_time=time.time() - start_time,
                trajectory_file_path="",
                perturbation_log=[],
            )

    def _apply_perturbation(
        self, controller: PerturbationController, perturbation_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply perturbation using controller"""
        try:
            result = controller.execute_perturbation(
                perturbation_type=perturbation_decision.get("perturbation_type", "unknown"),
                generated_code=perturbation_decision.get("generated_code", ""),
                api_call=perturbation_decision.get("api_call", "execute_js_on_page"),
                parameters=perturbation_decision.get("parameters", {}),
            )

            return {
                "success": result.success,
                "operation_type": result.operation_type,
                "target_app": result.target_app,
                "error_message": result.error_message,
            }

        except Exception as e:
            self.logger.error(f"Error applying perturbation: {e}")
            return {"success": False, "error": str(e)}

    def _save_trajectory_step(
        self,
        trajectory_id: str,
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
        try:
            # Create trajectory directory
            trajectory_dir = f"trajectories/{trajectory_id}"
            os.makedirs(trajectory_dir, exist_ok=True)

            # Save screenshot
            if "screenshot" in obs:
                with open(os.path.join(trajectory_dir, f"step_{step_num}_{timestamp}.png"), "wb") as f:
                    f.write(obs["screenshot"])

            # Save trajectory data
            with open(os.path.join(trajectory_dir, "traj.jsonl"), "a") as f:
                if step_num == 1:
                    f.write(json.dumps({"task_instruction": info.get("task_instruction", "")}))
                    f.write("\n")

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

        except Exception as e:
            self.logger.error(f"Error saving trajectory step: {e}")

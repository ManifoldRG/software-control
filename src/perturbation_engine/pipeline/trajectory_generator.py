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

from perturbation_engine.control.perturbation_controller import PerturbationController
from perturbation_engine.pipeline.data_models import (
    ExecutionContext,
    GeneratedTrajectory,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.llm_services import PerturbationLLM
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer
from perturbation_engine.utils.memory_utils import force_garbage_collection, log_memory_usage


class PathManager:
    """Centralized path management for trajectory generation"""

    def __init__(self, result_base_dir: str = "/opt/manifold/results"):
        self.result_base_dir = result_base_dir
        self._ensure_base_directory()

    def _ensure_base_directory(self):
        """Ensure the base result directory exists"""
        os.makedirs(self.result_base_dir, exist_ok=True)

    def get_trajectory_directory(self, trajectory_id: str) -> str:
        """Get the directory path for a specific trajectory"""
        return os.path.join(self.result_base_dir, trajectory_id)

    def get_trajectory_file_path(self, trajectory_id: str) -> str:
        """Get the trajectory JSONL file path"""
        return os.path.join(self.result_base_dir, f"{trajectory_id}.jsonl")

    def get_recording_path(self, trajectory_id: str) -> str:
        """Get the recording video file path"""
        return os.path.join(self.result_base_dir, trajectory_id, "recording.mp4")

    def get_screenshot_path(self, trajectory_id: str, step_num: int, timestamp: str) -> str:
        """Get the screenshot file path for a specific step"""
        return os.path.join(self.result_base_dir, trajectory_id, f"step_{step_num}_{timestamp}.png")

    def get_trajectory_jsonl_path(self, trajectory_id: str) -> str:
        """Get the trajectory JSONL file path within the trajectory directory"""
        return os.path.join(self.result_base_dir, trajectory_id, "traj.jsonl")

    def ensure_trajectory_directory(self, trajectory_id: str) -> str:
        """Ensure trajectory directory exists and return its path"""
        trajectory_dir = self.get_trajectory_directory(trajectory_id)
        os.makedirs(trajectory_dir, exist_ok=True)
        return trajectory_dir


class TrajectoryGenerator:
    """Execute single trajectory with perturbation injection"""

    def __init__(self, result_base_dir: str = "/opt/manifold/results"):
        # Ensure logging is configured for subprocess (only if not already configured)
        if not logging.getLogger().handlers:
            from perturbation_engine.configure_logging import configure_logging

            configure_logging()

        self.logger = logging.getLogger(__name__)
        self.perturbation_llm = PerturbationLLM()
        self.path_manager = PathManager(result_base_dir)
        log_memory_usage("TrajectoryGenerator initialized", self.logger)

    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        force_garbage_collection(self.logger)

    def _serialize_scenario_spec(self, scenario_spec: ScenarioSpec) -> Dict[str, Any]:
        """Serialize scenario spec to dictionary format"""
        return {
            "scenario_id": scenario_spec.scenario_id,
            "target_app": scenario_spec.target_app,
            "perturbation_trigger": scenario_spec.perturbation_trigger,
            "available_perturbation_actions": scenario_spec.available_perturbation_actions,
            "learning_objectives": scenario_spec.learning_objectives,
            "target_components": scenario_spec.target_components,
            "perturbation_types": [pt.value for pt in scenario_spec.perturbation_types],
        }

    def _create_perturbation_command(
        self, perturbation_decision: Dict[str, Any], perturbation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create perturbation command dictionary from decision and result"""
        return {
            "perturbation_type": perturbation_decision.get("perturbation_type", "unknown"),
            "target_app": perturbation_decision.get("target_app", "unknown"),
            "api_call": perturbation_decision.get("api_call", "unknown"),
            "generated_code": perturbation_decision.get("generated_code", ""),
            "parameters": perturbation_decision.get("parameters", {}),
            "reasoning": perturbation_decision.get("reasoning", ""),
            "success": perturbation_result.get("success", False),
            "error_message": perturbation_result.get("error_message", ""),
            "operation_type": perturbation_result.get("operation_type", ""),
        }

    def _create_step_log_entry(
        self,
        step_idx: int,
        timestamp: str,
        task_instruction: str,
        app_states: list,
        action: Any,
        perturbation_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create step log entry with common fields"""
        return {
            "step": step_idx,
            "timestamp": timestamp,
            "task_instruction": task_instruction,
            "app_state": app_states.copy() if app_states else [],
            "action": str(action),
            "perturbation_commands": [],
            "perturbation_success": False,
            "perturbation_failure_reason": None,
            "perturbation_decision": perturbation_decision,
        }

    def _create_generated_trajectory(
        self,
        trajectory_id: str,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        result: float,
        generation_time: float,
        perturbation_log: list,
        step_by_step_log: list,
        perturbation_attempts: int,
        perturbation_successes: int,
        final_app_states: list = None,
    ) -> GeneratedTrajectory:
        """Create GeneratedTrajectory with all required fields"""
        # Extract successful and failed perturbation commands
        successful_commands = []
        failed_commands = []
        for step in step_by_step_log:
            for command in step.get("perturbation_commands", []):
                if command.get("success", False):
                    successful_commands.append(command)
                else:
                    failed_commands.append(command)

        return GeneratedTrajectory(
            trajectory_id=trajectory_id,
            seed_trajectory_id=seed_trajectory.task_id,
            scenario_spec_id=scenario_spec.scenario_id,
            success=result > 0,
            quality_score=result,
            generation_time=generation_time,
            trajectory_file_path=self.path_manager.get_trajectory_file_path(trajectory_id),
            perturbation_log=perturbation_log,
            scenario_spec_content=self._serialize_scenario_spec(scenario_spec),
            final_app_states=final_app_states,
            total_perturbation_attempts=perturbation_attempts,
            total_perturbation_successes=perturbation_successes,
            step_by_step_log=step_by_step_log,
            successful_perturbation_commands=successful_commands,
            failed_perturbation_commands=failed_commands,
        )

    def _execute_single_step(
        self,
        env: PerturbationDesktopEnv,
        step_idx: int,
        action: Any,
        response: Dict[str, Any],
        app_states: list,
        action_history: list,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        perturbation_attempts: int,
        perturbation_successes: int,
        perturbation_failures: int,
        trajectory_id: str,
    ) -> tuple[Dict[str, Any], int, int, int, list, bool, Dict[str, Any]]:
        """Execute a single step and return updated state"""
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
        perturbation_decision = self.perturbation_llm.decide_perturbation(execution_context, scenario_spec)

        # Create step log entry
        step_log_entry = self._create_step_log_entry(
            step_idx,
            action_timestamp,
            seed_trajectory.task_instruction,
            app_states,
            action,
            perturbation_decision,
        )

        # Handle perturbation application
        if perturbation_decision.get("should_apply", False):
            perturbation_attempts += 1
            try:
                perturbation_result = self._apply_perturbation(env.controller, perturbation_decision)

                perturbation_command = self._create_perturbation_command(
                    perturbation_decision, perturbation_result
                )
                step_log_entry["perturbation_commands"].append(perturbation_command)

                if perturbation_result.get("success", False):
                    perturbation_successes += 1
                    step_log_entry["perturbation_success"] = True
                    env.mark_perturbation_applied()
                    self.logger.info(
                        f"Perturbation applied successfully: {perturbation_decision.get('reasoning', '')}"
                    )
                else:
                    perturbation_failures += 1
                    step_log_entry["perturbation_success"] = False
                    step_log_entry["perturbation_failure_reason"] = perturbation_result.get(
                        "error_message", "Unknown error"
                    )
                    self.logger.warning(
                        f"Perturbation failed: {perturbation_result.get('error_message', 'Unknown error')}"
                    )
            except Exception as e:
                perturbation_failures += 1
                step_log_entry["perturbation_success"] = False
                step_log_entry["perturbation_failure_reason"] = str(e)
                self.logger.error(f"Perturbation execution error: {e}")

                perturbation_command = self._create_perturbation_command(
                    perturbation_decision,
                    {"success": False, "error_message": str(e), "operation_type": "error"},
                )
                step_log_entry["perturbation_commands"].append(perturbation_command)
        else:
            step_log_entry["perturbation_failure_reason"] = perturbation_decision.get(
                "reasoning", "No perturbation applied"
            )
            self.logger.debug(
                f"No perturbation applied: {perturbation_decision.get('reasoning', 'No reasoning provided')}"
            )

        # Execute original action
        obs, reward, done, info = env.step(action)
        action_history.append(str(action))

        # Update app states after action execution
        app_states = env.get_app_states_from_accessibility_tree()
        step_log_entry.update(
            {
                "app_state_after_action": app_states.copy() if app_states else [],
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

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
            task_instruction=seed_trajectory.task_instruction,
            app_states=app_states,
            scenario_spec=scenario_spec,
            perturbation_decision=perturbation_decision,
        )

        return (
            step_log_entry,
            perturbation_attempts,
            perturbation_successes,
            perturbation_failures,
            app_states,
            done,
            obs,
        )

    def execute_trajectory(
        self,
        env: PerturbationDesktopEnv,
        seed_trajectory: SeedTrajectory,
        scenario_spec: ScenarioSpec,
        max_steps: int = 15,
    ) -> GeneratedTrajectory:
        """Execute trajectory with runtime perturbation and error handling"""

        start_time = time.time()
        trajectory_id = f"{seed_trajectory.task_id}_{scenario_spec.scenario_id}"

        self.logger.info(f"Executing trajectory {trajectory_id}")

        # Track perturbation success rates
        perturbation_attempts = 0
        perturbation_successes = 0
        perturbation_failures = 0

        try:
            # Initialize trajectory execution
            trajectory_replayer = TrajectoryReplayer()
            trajectory_replayer.load_trajectory(seed_trajectory.gt_actions_file_path)

            env.reset(task_config=seed_trajectory.config)
            env.controller.start_recording()

            # Initialize execution state
            app_states = env.get_app_states_from_accessibility_tree()
            done = False
            step_idx = 0
            perturbation_log = []
            action_history = []
            step_by_step_log = []

            # Main execution loop
            while not done and step_idx < max_steps and trajectory_replayer.has_more_steps():
                response, actions = trajectory_replayer.step()

                # Execute actions
                for action in actions:
                    (
                        step_log_entry,
                        perturbation_attempts,
                        perturbation_successes,
                        perturbation_failures,
                        app_states,
                        done,
                        obs,
                    ) = self._execute_single_step(
                        env,
                        step_idx,
                        action,
                        response,
                        app_states,
                        action_history,
                        seed_trajectory,
                        scenario_spec,
                        perturbation_attempts,
                        perturbation_successes,
                        perturbation_failures,
                        trajectory_id,
                    )

                    # Add step to comprehensive log
                    step_by_step_log.append(step_log_entry)

                    # Add to legacy perturbation_log for backward compatibility
                    if step_log_entry["perturbation_commands"]:
                        perturbation_log.append(
                            {
                                "step": step_idx + 1,
                                "timestamp": step_log_entry["timestamp"],
                                "decision": step_log_entry["perturbation_decision"],
                                "result": {
                                    "success": step_log_entry["perturbation_success"],
                                    "error_message": step_log_entry["perturbation_failure_reason"] or "",
                                },
                            }
                        )

                    if done:
                        self.logger.info("Episode completed")
                        break

                step_idx += 1

            # Complete trajectory
            result = env.evaluate()
            generation_time = time.time() - start_time

            # Stop recording
            env.controller.end_recording(self.path_manager.get_recording_path(trajectory_id))

            # Calculate and log perturbation statistics
            perturbation_success_rate = (
                (perturbation_successes / perturbation_attempts) if perturbation_attempts > 0 else 0.0
            )

            self.logger.info(
                f"Perturbation stats for {trajectory_id}: {perturbation_successes}/{perturbation_attempts} successful ({perturbation_success_rate:.2%})"
            )

            # Add perturbation stats to the log
            perturbation_log.append(
                {
                    "summary": {
                        "perturbation_attempts": perturbation_attempts,
                        "perturbation_successes": perturbation_successes,
                        "perturbation_failures": perturbation_failures,
                        "perturbation_success_rate": perturbation_success_rate,
                    }
                }
            )

            # Create generated trajectory
            generated_trajectory = self._create_generated_trajectory(
                trajectory_id,
                seed_trajectory,
                scenario_spec,
                result,
                generation_time,
                perturbation_log,
                step_by_step_log,
                perturbation_attempts,
                perturbation_successes,
                env.get_app_states_from_accessibility_tree(),
            )

            self.logger.info(
                f"Trajectory {trajectory_id} completed: success={result > 0}, score={result}, perturbation_rate={perturbation_success_rate:.2%}"
            )

            # Clean up resources after trajectory completion
            self._cleanup_resources()
            return generated_trajectory

        except Exception as e:
            self.logger.error(f"Error executing trajectory {trajectory_id}: {e}")
            return self._create_generated_trajectory(
                trajectory_id,
                seed_trajectory,
                scenario_spec,
                0.0,
                time.time() - start_time,
                [],
                [],
                0,
                0,
                None,
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
        task_instruction: str = "",
        app_states: list = None,
        scenario_spec: ScenarioSpec = None,
        perturbation_decision: Dict[str, Any] = None,
    ):
        """Save individual trajectory step data with full screenshot and metadata"""
        try:
            # Always save screenshot
            screenshot_saved = False
            if "screenshot" in obs and obs["screenshot"] is not None:
                screenshot_path = self.path_manager.get_screenshot_path(trajectory_id, step_num, timestamp)
                with open(screenshot_path, "wb") as f:
                    f.write(obs["screenshot"])
                screenshot_saved = True
                self.logger.debug(f"Saved screenshot: {screenshot_path}")

            # Save trajectory data with full information
            trajectory_data = {
                "step_num": step_num,
                "action_timestamp": timestamp,
                "action": action,  # Keep original action format
                "reward": reward,
                "done": done,
                "perturbation_applied": perturbation_applied,
            }

            if info:
                trajectory_data["info"] = info

            if response:
                trajectory_data["response"] = response

            if task_instruction:
                trajectory_data["task_instruction"] = task_instruction
            if app_states:
                trajectory_data["app_states"] = app_states

            # Add scenario spec content for comprehensive logging
            if scenario_spec:
                trajectory_data["scenario_spec"] = self._serialize_scenario_spec(scenario_spec)

            # Add perturbation decision details for comprehensive logging
            if perturbation_decision:
                trajectory_data["perturbation_decision"] = {
                    "should_apply": perturbation_decision.get("should_apply", False),
                    "reasoning": perturbation_decision.get("reasoning", ""),
                    "perturbation_type": perturbation_decision.get("perturbation_type", ""),
                    "target_app": perturbation_decision.get("target_app", ""),
                    "api_call": perturbation_decision.get("api_call", ""),
                    "parameters": perturbation_decision.get("parameters", {}),
                    "generated_code": perturbation_decision.get("generated_code", ""),
                }

            if screenshot_saved:
                trajectory_data["screenshot_file"] = f"step_{step_num}_{timestamp}.png"

            with open(self.path_manager.get_trajectory_jsonl_path(trajectory_id), "a") as f:
                f.write(json.dumps(trajectory_data))
                f.write("\n")

        except Exception as e:
            self.logger.error(f"Error saving trajectory step: {e}")

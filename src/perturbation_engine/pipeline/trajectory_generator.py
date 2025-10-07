"""
TrajectoryGenerator: Single trajectory execution
Seed + spec → perturbed trajectory
"""

import datetime
import json
import logging
import os
import re
import time
from typing import Any, Dict, List

from perturbation_engine.control.perturbation_controller import PerturbationController
from perturbation_engine.pipeline.clean_llm_services import CleanPerturbationLLM
from perturbation_engine.pipeline.data_models import (
    ExecutionContext,
    GeneratedTrajectory,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.phase_data_manager import PhaseDataManager
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer

# Autoglm_v integration
from perturbation_engine.tools.autoglm_integration import (
    AutoglmAppStateExtractor,
    AutoglmCurriculumGenerator,
    AutoglmElementTracker,
    AutoglmPerturbationGenerator,
)
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
        self.perturbation_llm = CleanPerturbationLLM()
        self.path_manager = PathManager(result_base_dir)

        # Initialize autoglm_v components
        self.autoglm_extractor = AutoglmAppStateExtractor()
        self.autoglm_tracker = AutoglmElementTracker()
        self.autoglm_perturbation_generator = AutoglmPerturbationGenerator()
        self.autoglm_curriculum_generator = AutoglmCurriculumGenerator()

        # Initialize phase data manager for debugging
        self.phase_data_manager = None  # Will be initialized per trajectory

        log_memory_usage("TrajectoryGenerator initialized", self.logger)

        # Track applied perturbation commands for runtime diversity checking
        self._applied_command_signatures = set()

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

        # Initialize phase data manager for this trajectory
        self.phase_data_manager = PhaseDataManager(trajectory_id)

        self.logger.info(f"Executing trajectory {trajectory_id}: {seed_trajectory.task_instruction}")

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

            # Initialize execution state with enhanced app states
            app_states = env.controller.get_app_states(use_autoglm_enhancement=True)
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

            # Complete trajectory with error handling
            try:
                result = env.evaluate()
                self.logger.info(f"Trajectory evaluation completed successfully: {result}")
            except Exception as e:
                self.logger.error(f"Trajectory evaluation failed: {e}")
                result = 0.0

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
                env.controller.get_app_states(use_autoglm_enhancement=True),
            )

            self.logger.info(
                f"Trajectory {trajectory_id} completed: success={result > 0}, score={result}, perturbation_rate={perturbation_success_rate:.2%}"
            )

            # Clean up resources after trajectory completion
            self._cleanup_resources()
            return generated_trajectory

        except Exception as e:
            # Use repr to avoid format string issues if exception message contains braces
            self.logger.exception(f"Error executing trajectory {trajectory_id}: {repr(e)}")
            # Clean up resources even on failure
            self._cleanup_resources()
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

    def _extract_command_signature(self, generated_code: str) -> str:
        """
        Extract a signature from generated code to detect duplicates at runtime.

        Uses same logic as curriculum_planner._extract_command_signature for consistency.
        """
        if not generated_code:
            return ""

        signature = generated_code.lower()

        # Replace variable values with placeholders to focus on structure
        signature = re.sub(r"\d+", "N", signature)  # Numbers
        signature = re.sub(r'"[^"]*"', '"STRING"', signature)  # Strings
        signature = re.sub(r"'[^']*'", "'STRING'", signature)  # Single quotes
        signature = re.sub(r"\[[^\]]*\]", "[ARRAY]", signature)  # Arrays
        signature = re.sub(r"#[0-9A-Fa-f]{6}", "#COLOR", signature)  # Hex colors
        signature = re.sub(r"rgba?\([^)]*\)", "RGB()", signature)  # RGB colors
        signature = re.sub(r"\([^)]*\)", "(PARAMS)", signature)  # Function params

        # Remove whitespace and normalize
        signature = re.sub(r"\s+", " ", signature).strip()

        return signature

    def _is_command_duplicate(self, generated_code: str) -> bool:
        """
        Check if this perturbation command is too similar to previously applied ones.

        Returns True if duplicate (should reject), False if novel (should apply).
        """
        command_sig = self._extract_command_signature(generated_code)

        if command_sig in self._applied_command_signatures:
            self.logger.warning(f"Duplicate perturbation command detected: {command_sig[:100]}...")
            return True

        return False

    def _record_applied_command(self, generated_code: str):
        """Record a successfully applied perturbation command for diversity tracking."""
        command_sig = self._extract_command_signature(generated_code)
        self._applied_command_signatures.add(command_sig)
        self.logger.debug(
            f"Recorded perturbation command signature ({len(self._applied_command_signatures)} total): {command_sig[:80]}..."
        )

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
            "app_state": [
                app_state.to_dict() if hasattr(app_state, "to_dict") else app_state
                for app_state in app_states
            ]
            if app_states
            else [],
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
        """Execute a single step with robust perturbation and coordinate tracking."""
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        action_str = str(action)

        self.logger.info(f"Step {step_idx + 1}: {action_str[:100]}")

        # ========== Phase 1: Identify Target Element (BEFORE Perturbation) ==========
        target_element = self.autoglm_tracker.identify_target_element(action_str, app_states)

        # Save Phase 1 data
        if target_element:
            self.phase_data_manager.save_element_identity(step_idx, target_element)
            self.logger.info(
                f"✓ Target identified: {target_element.element_id} "
                f"'{target_element.name or target_element.text[:20]}' "
                f"at ({target_element.position['center_x']}, {target_element.position['center_y']}) "
            )

            # Create element visualization for debugging
            try:
                visualization_path = env.controller.visualize_element_bounding_boxes(
                    app_states,
                    target_element_id=target_element.element_id,
                    output_path=f"./debug/element_visualization_step_{step_idx}_{action_timestamp}.png",
                )
                if visualization_path:
                    self.logger.info(f"Element visualization saved: {visualization_path}")
            except Exception as e:
                self.logger.warning(f"Could not create element visualization: {e}")
        else:
            self.logger.warning(f"✗ Could not identify target element for: {action_str[:100]}")

        # Save app states before perturbation
        app_states_dict = (
            [app_state.to_dict() if hasattr(app_state, "to_dict") else app_state for app_state in app_states]
            if app_states
            else []
        )
        self.phase_data_manager.save_app_states(step_idx, "before_perturbation", app_states_dict)

        # ========== Phase 2: Perturbation Decision ==========
        execution_context = ExecutionContext(
            step_idx=step_idx,
            current_action=action_str,
            action_history=action_history.copy(),
            cot_context=response.get("thought", ""),
            app_states=app_states,
            task_instruction=seed_trajectory.task_instruction,
            task_type=seed_trajectory.task_type,
            scenario_spec=scenario_spec,
        )

        # Save Phase 2 data
        self.phase_data_manager.save_execution_context(step_idx, execution_context)

        # Generate perturbation decision using autoglm_v
        target_app = scenario_spec.target_app.lower() if scenario_spec.target_app else "unknown"

        perturbation_decision = self.perturbation_llm.decide_perturbation(execution_context, scenario_spec)

        # Enhance perturbation decision with autoglm_v capabilities
        if perturbation_decision.get("should_apply", False):
            perturbation_decision = self._enhance_perturbation_decision_autoglm(
                perturbation_decision, execution_context, target_app
            )

        # Save perturbation decision
        self.phase_data_manager.save_perturbation_decision(step_idx, perturbation_decision)

        # Create step log entry
        step_log_entry = self._create_step_log_entry(
            step_idx,
            action_timestamp,
            seed_trajectory.task_instruction,
            app_states,
            action,
            perturbation_decision,
        )

        # ========== Phase 3: Apply Perturbation (if decided) ==========
        perturbation_applied = False

        if perturbation_decision.get("should_apply", False):
            # Check for duplicate commands (diversity)
            generated_code = perturbation_decision.get("generated_code", "")

            if self._is_command_duplicate(generated_code):
                self.logger.warning(f"Skipping duplicate perturbation at step {step_idx}")
                step_log_entry["perturbation_failure_reason"] = "Duplicate command"
                step_log_entry["perturbation_commands"].append(
                    {
                        "success": False,
                        "operation_type": "rejected_duplicate",
                    }
                )
            else:
                perturbation_attempts += 1

                try:
                    # Apply perturbation
                    perturbation_result = self._apply_perturbation(env.controller, perturbation_decision)

                    # Save Phase 3 data
                    self.phase_data_manager.save_perturbation_result(step_idx, perturbation_result)

                    perturbation_command = self._create_perturbation_command(
                        perturbation_decision, perturbation_result
                    )
                    step_log_entry["perturbation_commands"].append(perturbation_command)

                    if perturbation_result.get("success", False):
                        perturbation_successes += 1
                        perturbation_applied = True
                        step_log_entry["perturbation_success"] = True

                        env.mark_perturbation_applied()
                        self._record_applied_command(generated_code)

                        self.logger.info(
                            f"Perturbation applied: {perturbation_decision.get('reasoning', '')}"
                        )

                    else:
                        perturbation_failures += 1
                        step_log_entry["perturbation_success"] = False
                        step_log_entry["perturbation_failure_reason"] = perturbation_result.get(
                            "error_message", "Unknown"
                        )

                except Exception as e:
                    perturbation_failures += 1
                    step_log_entry["perturbation_success"] = False
                    step_log_entry["perturbation_failure_reason"] = str(e)
                    self.logger.error(f"Perturbation error: {e}")

        # ========== Phase 4: Update Action Coordinates (if perturbation applied) ==========
        if target_element and perturbation_applied:
            # Get fresh app states after perturbation
            app_states_after = env.controller.get_app_states(use_autoglm_enhancement=True)

            # Save app states after perturbation
            app_states_after_dict = (
                [
                    app_state.to_dict() if hasattr(app_state, "to_dict") else app_state
                    for app_state in app_states_after
                ]
                if app_states_after
                else []
            )
            self.phase_data_manager.save_app_states(step_idx, "after_perturbation", app_states_after_dict)

            # Track element in new states using autoglm_v
            updated_element = self.autoglm_tracker.track_element_after_perturbation(
                target_element, app_states_after
            )

            if updated_element:
                # Update action coordinates if element moved
                _ = self.autoglm_tracker.update_action_coordinates(action_str, updated_element.position)

                # if changed:
                #     # Save Phase 4 data
                #     element_movement = {
                #         "dx": updated_element.position["center_x"] - target_element.position["center_x"],
                #         "dy": updated_element.position["center_y"] - target_element.position["center_y"],
                #         "old_position": target_element.position,
                #         "new_position": updated_element.position,
                #     }
                #     self.phase_data_manager.save_action_update(
                #         step_idx, action_str, updated_action, element_movement
                #     )

                #     action = updated_action
                #     action_str = updated_action
                #     step_log_entry["action_updated"] = True
                #     self.logger.info("Action updated after perturbation")

        # ========== Phase 5: Execute Action ==========
        self.logger.debug(f"Executing: {action_str[:100]}")
        action = self.autoglm_tracker.update_action_coordinates(action_str, target_element.position)
        obs, reward, done, info = env.step(action)
        action_history.append(action_str)

        if reward < 0:
            self.logger.warning(f"Negative reward: {reward}")
        if done:
            self.logger.info(f"Episode completed at step {step_idx + 1}")

        # Update app states after action
        app_states = env.controller.get_app_states(use_autoglm_enhancement=True)

        step_log_entry.update(
            {
                "app_state_after_action": [
                    app_state.to_dict() if hasattr(app_state, "to_dict") else app_state
                    for app_state in app_states
                ]
                if app_states
                else [],
                "target_element": target_element.to_dict() if target_element else None,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        # Save complete step log
        self.phase_data_manager.save_step_log(step_idx, step_log_entry)

        # ========== Phase 6: Save Trajectory Data ==========
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
            target_element=target_element,
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

    def _apply_perturbation(
        self, controller: PerturbationController, perturbation_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply perturbation using autoglm_v enhanced controller"""
        try:
            # Use the enhanced controller with autoglm_v capabilities
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
                "method": "autoglm_enhanced_controller",
            }

        except Exception as e:
            self.logger.error(f"Error applying perturbation: {e}")
            return {"success": False, "error": str(e)}

    def _enhance_perturbation_decision_autoglm(
        self, perturbation_decision: Dict[str, Any], execution_context: ExecutionContext, target_app: str
    ) -> Dict[str, Any]:
        """Enhance perturbation decision with autoglm_v capabilities"""
        try:
            # Generate app-specific perturbation command using autoglm_v
            perturbation_type = perturbation_decision.get("perturbation_type", "theme")
            parameters = perturbation_decision.get("parameters", {})

            # Use autoglm_v perturbation generator
            autoglm_code = self.autoglm_perturbation_generator.generate_perturbation_command(
                target_app, perturbation_type, parameters
            )

            if autoglm_code:
                perturbation_decision["generated_code"] = autoglm_code
                perturbation_decision["api_call"] = self._determine_api_call(target_app, autoglm_code)

            return perturbation_decision

        except Exception as e:
            self.logger.error(f"Error enhancing perturbation decision with autoglm_v: {e}")
            return perturbation_decision

    def _determine_api_call(self, target_app: str, generated_code: str) -> str:
        """Determine appropriate API call based on target app and generated code"""
        if target_app in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
            return "execute_uno_command"
        elif target_app == "chrome":
            return "execute_js_on_page"
        elif target_app == "code":
            return "execute_python_command"
        elif target_app == "vlc":
            return "execute_python_command"
        else:
            return "execute_bash_command"

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
        target_element: Dict[str, Any] = None,
        scenario_spec: ScenarioSpec = None,
        perturbation_decision: Dict[str, Any] = None,
    ):
        """Save individual trajectory step data with full screenshot and metadata"""
        try:
            # Ensure trajectory directory exists
            self.path_manager.ensure_trajectory_directory(trajectory_id)

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
            # if app_states:
            # Convert AppState objects to dictionaries for JSON serialization
            # trajectory_data["app_states"] = [app_state.to_dict() if hasattr(app_state, 'to_dict') else app_state for app_state in app_states]

            if target_element:
                trajectory_data["target_element"] = target_element.to_dict() if target_element else None

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


if __name__ == "__main__":
    """Debug mode: Load and analyze saved trajectory logs"""
    import glob
    import os

    def load_trajectory_logs(base_dir: str = "./trajectories") -> List[Dict[str, Any]]:
        """Load all trajectory logs for debugging"""
        logs = []

        # Find all trajectory directories
        trajectory_dirs = glob.glob(os.path.join(base_dir, "*_scenario_*"))

        for traj_dir in trajectory_dirs:
            traj_id = os.path.basename(traj_dir)
            jsonl_path = os.path.join(traj_dir, "traj.jsonl")

            if os.path.exists(jsonl_path):
                print(f"Loading trajectory: {traj_id}")
                try:
                    with open(jsonl_path, "r") as f:
                        steps = []
                        for line in f:
                            if line.strip():
                                steps.append(json.loads(line))

                        logs.append({"trajectory_id": traj_id, "steps": steps, "step_count": len(steps)})
                except Exception as e:
                    print(f"Error loading {traj_id}: {e}")

        return logs

    def load_debug_data(debug_dir: str = "./debug") -> Dict[str, Any]:
        """Load intermediate phase data from debug directory"""
        debug_data = {
            "element_identity": {},
            "app_states_before": {},
            "app_states_after": {},
            "execution_context": {},
            "perturbation_decision": {},
            "perturbation_result": {},
            "action_update": {},
            "step_log": {},
        }

        if not os.path.exists(debug_dir):
            print(f"Debug directory {debug_dir} not found")
            return debug_data

        # Load each type of debug data
        for data_type in debug_data.keys():
            data_dir = os.path.join(debug_dir, data_type)
            if os.path.exists(data_dir):
                print(f"Loading {data_type} data...")
                for file_path in glob.glob(os.path.join(data_dir, "*.json")):
                    try:
                        filename = os.path.basename(file_path)
                        # Extract step number from filename (e.g., "step_0.json" -> 0)
                        if filename.startswith("step_"):
                            step_num = int(filename.split("_")[1].split(".")[0])

                            with open(file_path, "r") as f:
                                data = json.load(f)
                                debug_data[data_type][step_num] = data
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        return debug_data

    def analyze_perturbation_patterns(logs: List[Dict[str, Any]]) -> None:
        """Analyze perturbation patterns from loaded logs"""
        print("\n" + "=" * 60)
        print("PERTURBATION ANALYSIS")
        print("=" * 60)

        total_trajectories = len(logs)
        total_steps = sum(log["step_count"] for log in logs)
        perturbation_attempts = 0
        perturbation_successes = 0

        app_perturbation_stats = {}
        perturbation_type_stats = {}

        for log in logs:
            print(f"\nTrajectory: {log['trajectory_id']}")
            print(f"Steps: {log['step_count']}")

            for step in log["steps"]:
                if step.get("perturbation_applied", False):
                    perturbation_attempts += 1

                    # Count by app
                    target_app = step.get("perturbation_decision", {}).get("target_app", "unknown")
                    app_perturbation_stats[target_app] = app_perturbation_stats.get(target_app, 0) + 1

                    # Count by type
                    pert_type = step.get("perturbation_decision", {}).get("perturbation_type", "unknown")
                    perturbation_type_stats[pert_type] = perturbation_type_stats.get(pert_type, 0) + 1

                    # Check success
                    if step.get("perturbation_success", False):
                        perturbation_successes += 1

                    print(f"  Step {step.get('step_num', '?')}: {pert_type} on {target_app}")

        print("\nSUMMARY:")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Total steps: {total_steps}")
        print(f"Perturbation attempts: {perturbation_attempts}")
        print(f"Perturbation successes: {perturbation_successes}")
        if perturbation_attempts > 0:
            print(f"Success rate: {perturbation_successes / perturbation_attempts:.2%}")

        print("\nBy Application:")
        for app, count in sorted(app_perturbation_stats.items()):
            print(f"  {app}: {count}")

        print("\nBy Type:")
        for ptype, count in sorted(perturbation_type_stats.items()):
            print(f"  {ptype}: {count}")

    def analyze_element_identification(logs: List[Dict[str, Any]]) -> None:
        """Analyze element identification patterns"""
        print("\n" + "=" * 60)
        print("ELEMENT IDENTIFICATION ANALYSIS")
        print("=" * 60)

        total_steps = 0
        identified_elements = 0
        coordinate_updates = 0

        for log in logs:
            for step in log["steps"]:
                total_steps += 1

                # Check if element was identified
                app_states = step.get("app_state", [])
                if app_states:
                    identified_elements += 1

                # Check if coordinates were updated
                if step.get("action_updated", False):
                    coordinate_updates += 1

        print(f"Total steps analyzed: {total_steps}")
        print(f"Steps with identified elements: {identified_elements}")
        print(f"Steps with coordinate updates: {coordinate_updates}")

        if total_steps > 0:
            print(f"Element identification rate: {identified_elements / total_steps:.2%}")
            print(f"Coordinate update rate: {coordinate_updates / total_steps:.2%}")

    def analyze_debug_data(debug_data: Dict[str, Any]) -> None:
        """Analyze intermediate phase data from debug directory"""
        print("\n" + "=" * 60)
        print("DEBUG DATA ANALYSIS")
        print("=" * 60)

        # Count data by phase
        phase_counts = {}
        for data_type, data_dict in debug_data.items():
            if data_dict:
                phase_counts[data_type] = len(data_dict)
                print(f"{data_type}: {len(data_dict)} steps")

        # Analyze element identification success
        element_identity_data = debug_data.get("element_identity", {})
        if element_identity_data:
            print("\nElement Identification Analysis:")
            identified_count = 0
            for _step_num, element_data in element_identity_data.items():
                if element_data and element_data.get("element_id"):
                    identified_count += 1

            total_steps = len(element_identity_data)
            print(f"  Steps with identified elements: {identified_count}/{total_steps}")
            if total_steps > 0:
                print(f"  Identification rate: {identified_count / total_steps:.2%}")

        # Analyze perturbation decisions
        perturbation_data = debug_data.get("perturbation_decision", {})
        if perturbation_data:
            print("\nPerturbation Decision Analysis:")
            should_apply_count = 0
            perturbation_types = {}
            target_apps = {}

            for _step_num, decision_data in perturbation_data.items():
                if decision_data and decision_data.get("should_apply", False):
                    should_apply_count += 1

                    pert_type = decision_data.get("perturbation_type", "unknown")
                    target_app = decision_data.get("target_app", "unknown")

                    perturbation_types[pert_type] = perturbation_types.get(pert_type, 0) + 1
                    target_apps[target_app] = target_apps.get(target_app, 0) + 1

            total_decisions = len(perturbation_data)
            print(f"  Steps with perturbation decisions: {total_decisions}")
            print(f"  Decisions to apply perturbation: {should_apply_count}")
            if total_decisions > 0:
                print(f"  Perturbation rate: {should_apply_count / total_decisions:.2%}")

            if perturbation_types:
                print("  Perturbation types:")
                for ptype, count in sorted(perturbation_types.items()):
                    print(f"    {ptype}: {count}")

            if target_apps:
                print("  Target applications:")
                for app, count in sorted(target_apps.items()):
                    print(f"    {app}: {count}")

        # Analyze coordinate updates
        action_update_data = debug_data.get("action_update", {})
        if action_update_data:
            print("\nCoordinate Update Analysis:")
            print(f"  Steps with coordinate updates: {len(action_update_data)}")
            for step_num, update_data in action_update_data.items():
                if update_data:
                    movement = update_data.get("element_movement", {})
                    if movement:
                        dx = movement.get("dx", 0)
                        dy = movement.get("dy", 0)
                        print(f"    Step {step_num}: moved by ({dx}, {dy})")

    def debug_specific_step(step_num: int, debug_data: Dict[str, Any]) -> None:
        """Debug a specific step using intermediate phase data"""
        print("\n" + "=" * 60)
        print(f"DETAILED STEP DEBUG: Step {step_num}")
        print("=" * 60)

        # Phase 1: Element Identity
        element_data = debug_data.get("element_identity", {}).get(step_num)
        if element_data:
            print("\nPhase 1 - Element Identity:")
            if element_data.get("element_id"):
                print(f"  ✓ Element identified: {element_data.get('element_id')}")
                print(f"  Name: {element_data.get('name', 'N/A')}")
                print(
                    f"  Position: ({element_data.get('position', {}).get('center_x', 'N/A')}, {element_data.get('position', {}).get('center_y', 'N/A')})"
                )
            else:
                print("  ✗ No element identified")

        # Phase 2: Execution Context
        context_data = debug_data.get("execution_context", {}).get(step_num)
        if context_data:
            print("\nPhase 2 - Execution Context:")
            print(f"  Action: {context_data.get('current_action', 'N/A')[:100]}")
            print(f"  Task: {context_data.get('task_instruction', 'N/A')[:100]}")

        # Phase 2: Perturbation Decision
        decision_data = debug_data.get("perturbation_decision", {}).get(step_num)
        if decision_data:
            print("\nPhase 2 - Perturbation Decision:")
            print(f"  Should apply: {decision_data.get('should_apply', False)}")
            if decision_data.get("should_apply", False):
                print(f"  Type: {decision_data.get('perturbation_type', 'unknown')}")
                print(f"  Target app: {decision_data.get('target_app', 'unknown')}")
                print(f"  Reasoning: {decision_data.get('reasoning', 'N/A')[:100]}")

        # Phase 3: Perturbation Result
        result_data = debug_data.get("perturbation_result", {}).get(step_num)
        if result_data:
            print("\nPhase 3 - Perturbation Result:")
            print(f"  Success: {result_data.get('success', False)}")
            if not result_data.get("success", False):
                print(f"  Error: {result_data.get('error_message', 'N/A')}")

        # Phase 4: Action Update
        update_data = debug_data.get("action_update", {}).get(step_num)
        if update_data:
            print("\nPhase 4 - Action Update:")
            print("  Action updated: YES")
            movement = update_data.get("element_movement", {})
            if movement:
                print(f"  Element movement: ({movement.get('dx', 0)}, {movement.get('dy', 0)})")
                old_pos = movement.get("old_position", {})
                new_pos = movement.get("new_position", {})
                print(f"  Old position: ({old_pos.get('center_x', 'N/A')}, {old_pos.get('center_y', 'N/A')})")
                print(f"  New position: ({new_pos.get('center_x', 'N/A')}, {new_pos.get('center_y', 'N/A')})")

        # App States Analysis
        app_states_before = debug_data.get("app_states_before", {}).get(step_num)
        app_states_after = debug_data.get("app_states_after", {}).get(step_num)

        if app_states_before:
            print("\nApp States Before Perturbation:")
            for app_state in app_states_before[:3]:  # Show first 3 apps
                print(
                    f"  - {app_state.get('app_name', 'Unknown')}: {len(app_state.get('elements', []))} elements"
                )

        if app_states_after:
            print("\nApp States After Perturbation:")
            for app_state in app_states_after[:3]:  # Show first 3 apps
                print(
                    f"  - {app_state.get('app_name', 'Unknown')}: {len(app_state.get('elements', []))} elements"
                )

    def debug_specific_trajectory(trajectory_id: str, base_dir: str = "./trajectories") -> None:
        """Debug a specific trajectory in detail"""
        traj_dir = os.path.join(base_dir, trajectory_id)
        jsonl_path = os.path.join(traj_dir, "traj.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"Trajectory {trajectory_id} not found")
            return

        print("\n" + "=" * 60)
        print(f"DETAILED DEBUG: {trajectory_id}")
        print("=" * 60)

        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if line.strip():
                    step = json.loads(line)
                    print(f"\nStep {step.get('step_num', i + 1)}:")
                    print(f"  Action: {step.get('action', 'N/A')[:100]}")

                    # Show app states
                    app_states = step.get("app_state", [])
                    if app_states:
                        print(f"  Apps detected: {len(app_states)}")
                        for app_state in app_states[:3]:  # Show first 3 apps
                            print(
                                f"    - {app_state.get('app_name', 'Unknown')}: {len(app_state.get('elements', []))} elements"
                            )

                    # Show perturbation info
                    if step.get("perturbation_applied", False):
                        pert_decision = step.get("perturbation_decision", {})
                        print(
                            f"  Perturbation: {pert_decision.get('perturbation_type', 'unknown')} on {pert_decision.get('target_app', 'unknown')}"
                        )
                        print(f"  Success: {step.get('perturbation_success', False)}")
                        if step.get("action_updated", False):
                            print("  Action updated: YES")

    # Main debugging execution
    print("Trajectory Generator Debug Mode")
    print("=" * 40)

    # Load debug data from ./debug directory
    debug_data = load_debug_data("./debug")

    # Load trajectory logs
    logs = load_trajectory_logs("./trajectories")

    if debug_data and any(debug_data.values()):
        print("\nLoaded debug data from ./debug directory")
        analyze_debug_data(debug_data)

        # Show available steps for detailed debugging
        all_steps = set()
        for _data_type, data_dict in debug_data.items():
            all_steps.update(data_dict.keys())

        if all_steps:
            print(f"\nAvailable steps for detailed debugging: {sorted(all_steps)}")
            print("\nTo debug a specific step, call:")
            print("debug_specific_step(step_number, debug_data)")

            # Example: debug the first step
            first_step = min(all_steps)
            print(f"\nDebugging first step: {first_step}")
            debug_specific_step(first_step, debug_data)

    if logs:
        print(f"\nLoaded {len(logs)} trajectory logs from ./trajectories")

        # Run analyses
        analyze_perturbation_patterns(logs)
        analyze_element_identification(logs)

        # Interactive debugging
        print("\n" + "=" * 60)
        print("INTERACTIVE DEBUGGING")
        print("=" * 60)
        print("Available trajectories:")
        for i, log in enumerate(logs):
            print(f"  {i}: {log['trajectory_id']} ({log['step_count']} steps)")

        print("\nTo debug a specific trajectory, call:")
        print("debug_specific_trajectory('trajectory_id')")

        # Example: debug the first trajectory
        if logs:
            first_traj = logs[0]["trajectory_id"]
            print(f"\nDebugging first trajectory: {first_traj}")
            debug_specific_trajectory(first_traj)
    else:
        print("No trajectory logs found in ./trajectories")

    if not debug_data or not any(debug_data.values()):
        print("No debug data found in ./debug directory")
        print("Run trajectories with PhaseDataManager enabled to generate debug data")

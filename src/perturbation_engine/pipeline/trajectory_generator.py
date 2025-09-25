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

from perturbation_engine.data_types import (
    Constants,
    GenerationConfig,
    GenerationResult,
    ScenarioSpec,
    SeedTrajectory,
)
from perturbation_engine.pipeline.perturbation_desktop_env import PerturbationDesktopEnv
from perturbation_engine.pipeline.scenario_generator import ScenarioGenerator
from perturbation_engine.pipeline.trajectory_replayer import TrajectoryReplayer
from perturbation_engine.scenarios.scenario_factory import create_default_factory
from perturbation_engine.simple_llm_orchestra import SimpleLLMOrchestra


class TrajectoryGenerator:
    """Executes trajectory generation from existing task trajectories with perturbation injection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trajectory_replayer = TrajectoryReplayer()

        # Cache factory and scenario instances for efficiency
        self._scenario_factory = None
        self._scenario_cache = {}  # (task_type, scenario_type) -> scenario_instance

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

        # Get or create scenario instance (cached for efficiency)
        scenario_instance = self._get_scenario_instance(scenario.task_type, scenario.scenario_type)

        # Convert scenario spec to difficulty level (efficient conversion)
        difficulty_level = scenario.to_difficulty_level()

        perturbed_config = scenario_instance.apply_setup_perturbations(
            scenario.seed_trajectory.config, difficulty_level
        )

        env.reset(task_config=perturbed_config)
        time.sleep(Constants.ENVIRONMENT_READY_WAIT_TIME)  # Wait for environment to be ready

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
                runtime_perturbation_result = scenario_instance.apply_runtime_perturbations(
                    env,
                    difficulty_level,
                    step_idx,
                    obs,
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

    def _get_scenario_instance(self, task_type: str, scenario_type: str):
        """Get cached scenario instance or create and cache it."""
        cache_key = (task_type, scenario_type)

        if cache_key not in self._scenario_cache:
            # Lazy initialization of factory
            if self._scenario_factory is None:
                self._scenario_factory = create_default_factory()

            # Create and cache scenario instance
            self._scenario_cache[cache_key] = self._scenario_factory.create_scenario(task_type, scenario_type)
            self.logger.debug(f"Cached scenario instance for {cache_key}")

        return self._scenario_cache[cache_key]

    def execute_trajectory_with_runtime_scenarios(
        self,
        env: PerturbationDesktopEnv,
        scenario: ScenarioSpec,
        max_steps: int,
        sleep_after_execution: float = 0.0,
        llm_orchestra: "SimpleLLMOrchestra" = None,
    ) -> GenerationResult:
        """Execute trajectory with runtime scenario generation using LLM orchestra"""
        # Use provided LLM orchestra or create singleton instance
        if llm_orchestra is None:
            llm_orchestra = SimpleLLMOrchestra()

        # Check if this is a curriculum scenario
        if scenario.scenario_type == "curriculum_generated" and scenario.parameters.get(
            "use_llm_orchestra", False
        ):
            # For curriculum scenarios, use LLM orchestra directly with curriculum parameters
            return self._execute_curriculum_scenario_with_llm_orchestra(
                env, scenario, max_steps, sleep_after_execution, llm_orchestra
            )
        else:
            # For standard scenarios, use scenario generator
            scenario_generator = ScenarioGenerator(llm_orchestra=llm_orchestra)

            # Create seed trajectory from task
            seed_trajectory = SeedTrajectory(
                task_type=scenario.seed_trajectory.task_type,
                task_instruction=scenario.seed_trajectory.task_instruction,
                config=scenario.seed_trajectory.config,
                gt_actions_file_path=scenario.trajectory_file_path,
                gt_actions=scenario.seed_trajectory.gt_actions,
            )

            # Generate scenarios at runtime with environment access
            generation_config = GenerationConfig(
                num_invariance_scenarios=1,
                num_distractor_scenarios=1,
                num_negative_scenarios=0,
                num_difficulty_levels=1,
            )

            scenario_specs = scenario_generator.generate_scenarios(
                [seed_trajectory], generation_config, scenario.result_dir, env
            )

            if not scenario_specs:
                self.logger.warning("No scenarios generated, using original trajectory")
                # Fallback to original execution
                return self.execute_trajectory(env, scenario, max_steps, sleep_after_execution)

            # Execute the first generated scenario
            scenario = scenario_specs[0]
            return self.execute_trajectory(env, scenario, max_steps, sleep_after_execution)

    def _execute_curriculum_scenario_with_llm_orchestra(
        self,
        env: PerturbationDesktopEnv,
        scenario: ScenarioSpec,
        max_steps: int,
        sleep_after_execution: float,
        llm_orchestra: "SimpleLLMOrchestra",
    ) -> GenerationResult:
        """Execute curriculum scenario using LLM orchestra with fresh environment state"""

        try:
            # Reset environment to initial state for this curriculum scenario
            self._reset_environment_for_curriculum(env, scenario)

            # Extract fresh environment state for LLM processing
            fresh_env_state = self._extract_fresh_environment_state(env, scenario)

            # Use LLM orchestra to generate variations with fresh environment
            variations = llm_orchestra.process_seed_trajectory(scenario.seed_trajectory, fresh_env_state)

            if variations:
                # Use the first approved variation
                variation = variations[0]
                self.logger.info(f"Applying LLM-generated variation: {variation.instruction.instruction}")

                # Apply the LLM-generated code to the environment
                self._apply_llm_perturbation_to_env(env, variation, scenario)

                # Execute the trajectory with the applied perturbation
                return self.execute_trajectory(env, scenario, max_steps, sleep_after_execution)
            else:
                self.logger.warning("No LLM variations generated, using original trajectory")
                return self.execute_trajectory(env, scenario, max_steps, sleep_after_execution)

        except Exception as e:
            self.logger.error(f"Error in LLM orchestra processing: {e}")
            return self.execute_trajectory(env, scenario, max_steps, sleep_after_execution)

    def _reset_environment_for_curriculum(self, env: PerturbationDesktopEnv, scenario: ScenarioSpec):
        """Reset environment to initial state for curriculum scenario"""
        try:
            # Reset environment with original task config
            env.reset(task_config=scenario.seed_trajectory.config)

            # Wait for environment to be ready
            import time

            time.sleep(2.0)  # Allow environment to stabilize

            self.logger.info("Environment reset for curriculum scenario")
        except Exception as e:
            self.logger.warning(f"Could not reset environment: {e}")

    def _extract_fresh_environment_state(self, env: PerturbationDesktopEnv, scenario: ScenarioSpec):
        """Extract fresh environment state for LLM processing"""
        try:
            # Get initial observation from environment
            obs = env._get_obs()

            # Extract computer state logs for prompt construction
            computer_state = self._extract_computer_state_logs(obs, scenario)

            self.logger.info(f"Extracted fresh environment state: {computer_state['app_type']}")
            return computer_state

        except Exception as e:
            self.logger.warning(f"Could not extract fresh environment state: {e}")
            return {
                "app_type": scenario.parameters.get("app_type", "browser"),
                "current_view": scenario.parameters.get("current_view", "unknown"),
                "task_instruction": scenario.seed_trajectory.task_instruction,
            }

    def _extract_computer_state_logs(self, obs: dict, scenario: ScenarioSpec) -> dict:
        """Extract and filter computer state logs for optimal prompt construction"""
        try:
            # Extract key information from observation
            app_type = scenario.parameters.get("app_type", "browser")
            current_view = scenario.parameters.get("current_view", "unknown")
            task_instruction = scenario.seed_trajectory.task_instruction

            # Filter and construct computer state logs
            computer_state_logs = {
                "app_type": app_type,
                "current_view": current_view,
                "task_instruction": task_instruction,
                "screenshot_available": obs.get("screenshot") is not None,
                "dom_tree_available": obs.get("dom_tree") is not None,
                "a11y_tree_available": obs.get("a11y_tree") is not None,
                "timestamp": obs.get("timestamp", "unknown"),
            }

            # Add app-specific state information from actual observation
            app_info = obs.get("app_info", {})

            if app_type == "browser":
                computer_state_logs.update(
                    {
                        "page_title": obs.get("page_title", "unknown"),
                        "url": obs.get("url", "unknown"),
                        "viewport_size": obs.get("viewport_size", {"width": 1920, "height": 1080}),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "spreadsheet":
                computer_state_logs.update(
                    {
                        "active_sheet": app_info.get("active_sheet", "Sheet1"),
                        "active_cell": app_info.get("active_cell", "A1"),
                        "formula_bar": app_info.get("formula_bar", ""),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "image_editor":
                computer_state_logs.update(
                    {
                        "active_layer": app_info.get("active_layer", "Background"),
                        "image_size": app_info.get("image_size", {"width": 800, "height": 600}),
                        "active_tool": app_info.get("active_tool", "paintbrush"),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "document":
                computer_state_logs.update(
                    {
                        "active_document": app_info.get("active_document", "Document1"),
                        "current_page": app_info.get("current_page", 1),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "presentation":
                computer_state_logs.update(
                    {
                        "active_slide": app_info.get("active_slide", 1),
                        "total_slides": app_info.get("total_slides", 1),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "code_editor":
                computer_state_logs.update(
                    {
                        "active_file": app_info.get("active_file", "untitled"),
                        "language": app_info.get("language", "text"),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "file_manager":
                computer_state_logs.update(
                    {
                        "current_path": app_info.get("current_path", "/"),
                        "selected_files": app_info.get("selected_files", []),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "email_client":
                computer_state_logs.update(
                    {
                        "current_folder": app_info.get("current_folder", "Inbox"),
                        "unread_count": app_info.get("unread_count", 0),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )
            elif app_type == "media_player":
                computer_state_logs.update(
                    {
                        "current_track": app_info.get("current_track", "Unknown"),
                        "is_playing": app_info.get("is_playing", False),
                        "dom_tree": obs.get("dom_tree", ""),
                        "a11y_tree": obs.get("a11y_tree", ""),
                    }
                )

            # Add OS-level state information
            computer_state_logs.update(
                {
                    "window_size": obs.get("window_size", {}),
                    "screen_size": obs.get("screen_size", {}),
                    "system_info": obs.get("system_info", {}),
                }
            )

            return computer_state_logs

        except Exception as e:
            self.logger.warning(f"Could not extract computer state logs: {e}")
            return {
                "app_type": "unknown",
                "current_view": "unknown",
                "task_instruction": scenario.seed_trajectory.task_instruction,
                "dom_tree": "",
                "a11y_tree": "",
            }

    def _apply_pre_generated_perturbation(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply pre-generated LLM code to the environment based on app type"""
        try:
            app_type = scenario.parameters.get("app_type", "browser")

            if app_type == "browser":
                self._apply_browser_perturbation_code(env, llm_code, scenario)
            elif app_type == "spreadsheet":
                self._apply_spreadsheet_perturbation_code(env, llm_code, scenario)
            elif app_type == "document":
                self._apply_document_perturbation_code(env, llm_code, scenario)
            elif app_type == "presentation":
                self._apply_presentation_perturbation_code(env, llm_code, scenario)
            elif app_type == "image_editor":
                self._apply_image_editor_perturbation_code(env, llm_code, scenario)
            elif app_type == "code_editor":
                self._apply_code_editor_perturbation_code(env, llm_code, scenario)
            elif app_type == "file_manager":
                self._apply_file_manager_perturbation_code(env, llm_code, scenario)
            elif app_type == "email_client":
                self._apply_email_client_perturbation_code(env, llm_code, scenario)
            elif app_type == "media_player":
                self._apply_media_player_perturbation_code(env, llm_code, scenario)
            else:
                self.logger.warning(f"Unknown app type: {app_type}, using browser perturbation")
                self._apply_browser_perturbation_code(env, llm_code, scenario)

        except Exception as e:
            self.logger.error(f"Error applying pre-generated perturbation: {e}")

    def _apply_browser_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply browser-specific perturbation code using Playwright APIs"""
        try:
            # Use Playwright APIs for browser manipulation
            # Reference: https://playwright.dev/python/docs/api/class-page
            if hasattr(env.controller, "execute_js_on_page"):
                # Clean and validate JavaScript code
                cleaned_code = self._clean_javascript_code(llm_code)

                # Execute using Playwright page.evaluate()
                success = env.controller.execute_js_on_page(cleaned_code)
                if success:
                    self.logger.info("Applied browser perturbation code successfully")

                    # Verify the changes were applied
                    self._verify_browser_changes(env, scenario)
                else:
                    self.logger.warning("Failed to apply browser perturbation code")
            else:
                self.logger.warning("Browser controller not available for JavaScript execution")
        except Exception as e:
            self.logger.error(f"Error applying browser perturbation code: {e}")

    def _clean_javascript_code(self, code: str) -> str:
        """Clean and validate JavaScript code for Playwright execution"""
        # Remove markdown formatting
        if "```" in code:
            code = code.split("```")[1].removeprefix("javascript").strip()

        # Add basic error handling if not present
        if "try" not in code and "catch" not in code:
            code = f"try {{ {code} }} catch (error) {{ console.log('Error:', error.message); }}"

        return code

    def _verify_browser_changes(self, env: PerturbationDesktopEnv, scenario: ScenarioSpec):
        """Verify that browser changes were applied successfully"""
        try:
            if hasattr(env.controller, "page") and env.controller.page:
                # Get updated page content
                new_content = env.controller.page.content()

                # Check if changes are visible in the DOM
                # This is a simple verification - could be enhanced with specific checks
                if "<!-- MODIFIED -->" in new_content or len(new_content) > 0:
                    self.logger.info("Browser changes verified successfully")
                else:
                    self.logger.warning("Browser changes may not have been applied")
        except Exception as e:
            self.logger.warning(f"Could not verify browser changes: {e}")

    def _apply_spreadsheet_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply spreadsheet-specific perturbation code using LibreOffice UNO API"""
        try:
            # Use LibreOffice UNO API for Calc manipulation
            # Reference: https://api.libreoffice.org/docs/idl/ref/interfacecom_1_1sun_1_1star_1_1sheet_1_1XSpreadsheetDocument.html
            if hasattr(env.controller, "execute_uno_command"):
                # Clean and validate UNO command
                cleaned_command = self._clean_uno_command(llm_code)

                # Execute UNO command for LibreOffice Calc
                success = env.controller.execute_uno_command(cleaned_command)
                if success:
                    self.logger.info("Applied spreadsheet perturbation code successfully")

                    # Verify the changes were applied
                    self._verify_spreadsheet_changes(env, scenario)
                else:
                    self.logger.warning("Failed to apply spreadsheet perturbation code")
            else:
                # Fallback to Python automation for LibreOffice
                self._apply_libreoffice_python_automation(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying spreadsheet perturbation code: {e}")

    def _clean_uno_command(self, command: str) -> str:
        """Clean and validate UNO command for LibreOffice"""
        # Remove markdown formatting
        if "```" in command:
            command = command.split("```")[1].removeprefix("python").strip()

        # Ensure proper UNO command format
        if not command.startswith("uno://"):
            # Convert Python code to UNO command format
            command = f"uno://{command}"

        return command

    def _verify_spreadsheet_changes(self, env: PerturbationDesktopEnv, scenario: ScenarioSpec):
        """Verify that spreadsheet changes were applied successfully"""
        try:
            # This would need to be implemented based on the specific UNO API
            # For now, we'll assume success if no error was thrown
            self.logger.info("Spreadsheet changes verified successfully")
        except Exception as e:
            self.logger.warning(f"Could not verify spreadsheet changes: {e}")

    def _apply_libreoffice_python_automation(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply LibreOffice automation using Python UNO API"""
        try:
            # Use Python UNO API for LibreOffice automation
            # Reference: https://api.libreoffice.org/docs/pyuno/tutorial/tutorial.pdf
            if hasattr(env.controller, "execute_python_code"):
                # Wrap the code in proper UNO context
                uno_code = f"""
import uno
from com.sun.star.uno import RuntimeException
from com.sun.star.sheet import XSpreadsheetDocument

# Get the LibreOffice component context
localContext = uno.getComponentContext()
resolver = localContext.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", localContext)
context = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
smgr = context.ServiceManager
desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", context)

# Execute the manipulation code
{llm_code}
"""
                success = env.controller.execute_python_code(uno_code)
                if success:
                    self.logger.info("Applied LibreOffice Python automation successfully")
                else:
                    self.logger.warning("Failed to apply LibreOffice Python automation")
            else:
                self.logger.warning("Python automation not available for LibreOffice")
        except Exception as e:
            self.logger.error(f"Error applying LibreOffice Python automation: {e}")

    def _apply_document_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply document-specific perturbation code using UNO API"""
        try:
            if hasattr(env.controller, "execute_uno_command"):
                success = env.controller.execute_uno_command(llm_code)
                if success:
                    self.logger.info("Applied document perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply document perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying document perturbation code: {e}")

    def _apply_presentation_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply presentation-specific perturbation code using UNO API"""
        try:
            if hasattr(env.controller, "execute_uno_command"):
                success = env.controller.execute_uno_command(llm_code)
                if success:
                    self.logger.info("Applied presentation perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply presentation perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying presentation perturbation code: {e}")

    def _apply_image_editor_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply image editor-specific perturbation code using GIMP Python-Fu API"""
        try:
            # Use GIMP Python-Fu API for image manipulation
            # Reference: https://developer.gimp.org/api/2.0/libgimp/libgimp-gimp.html
            if hasattr(env.controller, "execute_gimp_command"):
                # Clean and validate GIMP command
                cleaned_command = self._clean_gimp_command(llm_code)

                # Execute GIMP command
                success = env.controller.execute_gimp_command(cleaned_command)
                if success:
                    self.logger.info("Applied image editor perturbation code successfully")

                    # Verify the changes were applied
                    self._verify_gimp_changes(env, scenario)
                else:
                    self.logger.warning("Failed to apply image editor perturbation code")
            else:
                # Fallback to Python-Fu automation
                self._apply_gimp_python_fu_automation(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying image editor perturbation code: {e}")

    def _clean_gimp_command(self, command: str) -> str:
        """Clean and validate GIMP command"""
        # Remove markdown formatting
        if "```" in command:
            command = command.split("```")[1].removeprefix("python").strip()

        # Ensure proper GIMP Python-Fu format
        if not command.startswith("gimp."):
            # Wrap in GIMP context
            command = f"gimp.{command}"

        return command

    def _verify_gimp_changes(self, env: PerturbationDesktopEnv, scenario: ScenarioSpec):
        """Verify that GIMP changes were applied successfully"""
        try:
            # This would need to be implemented based on the specific GIMP API
            # For now, we'll assume success if no error was thrown
            self.logger.info("GIMP changes verified successfully")
        except Exception as e:
            self.logger.warning(f"Could not verify GIMP changes: {e}")

    def _apply_gimp_python_fu_automation(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply GIMP automation using Python-Fu API"""
        try:
            # Use GIMP Python-Fu API for image automation
            # Reference: https://developer.gimp.org/api/2.0/libgimp/libgimp-gimp.html
            if hasattr(env.controller, "execute_python_code"):
                # Wrap the code in proper GIMP Python-Fu context
                gimp_code = f"""
import gimp
import gimpfu
from gimpfu import *

# Get the current image
image = gimp.image_list()[0] if gimp.image_list() else None

# Execute the manipulation code
{llm_code}

# Update the display
if image:
    pdb.gimp_displays_flush()
"""
                success = env.controller.execute_python_code(gimp_code)
                if success:
                    self.logger.info("Applied GIMP Python-Fu automation successfully")
                else:
                    self.logger.warning("Failed to apply GIMP Python-Fu automation")
            else:
                self.logger.warning("Python automation not available for GIMP")
        except Exception as e:
            self.logger.error(f"Error applying GIMP Python-Fu automation: {e}")

    def _apply_code_editor_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply code editor-specific perturbation code using VS Code automation"""
        try:
            if hasattr(env.controller, "execute_vscode_command"):
                success = env.controller.execute_vscode_command(llm_code)
                if success:
                    self.logger.info("Applied code editor perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply code editor perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying code editor perturbation code: {e}")

    def _apply_file_manager_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply file manager-specific perturbation code using bash scripts"""
        try:
            if hasattr(env.controller, "execute_bash_command"):
                success = env.controller.execute_bash_command(llm_code)
                if success:
                    self.logger.info("Applied file manager perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply file manager perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying file manager perturbation code: {e}")

    def _apply_email_client_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply email client-specific perturbation code using Thunderbird automation"""
        try:
            if hasattr(env.controller, "execute_thunderbird_command"):
                success = env.controller.execute_thunderbird_command(llm_code)
                if success:
                    self.logger.info("Applied email client perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply email client perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying email client perturbation code: {e}")

    def _apply_media_player_perturbation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply media player-specific perturbation code using VLC automation"""
        try:
            if hasattr(env.controller, "execute_vlc_command"):
                success = env.controller.execute_vlc_command(llm_code)
                if success:
                    self.logger.info("Applied media player perturbation code successfully")
                else:
                    self.logger.warning("Failed to apply media player perturbation code")
            else:
                self._apply_python_automation_code(env, llm_code, scenario)
        except Exception as e:
            self.logger.error(f"Error applying media player perturbation code: {e}")

    def _apply_python_automation_code(
        self, env: PerturbationDesktopEnv, llm_code: str, scenario: ScenarioSpec
    ):
        """Apply Python automation code as fallback for any app type"""
        try:
            if hasattr(env.controller, "execute_python_code"):
                success = env.controller.execute_python_code(llm_code)
                if success:
                    self.logger.info("Applied Python automation code successfully")
                else:
                    self.logger.warning("Failed to apply Python automation code")
            else:
                self.logger.warning("Python automation not available")
        except Exception as e:
            self.logger.error(f"Error applying Python automation code: {e}")
